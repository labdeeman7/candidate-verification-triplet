#!/usr/bin/env python3
"""Train Qwen3-VL SFT while supervising every assistant turn in JSONL chats."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Sequence, Value, load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed: int = 42):
    """Make training as deterministic as the CUDA stack reasonably allows."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WeightedLossTrainer(Trainer):
    """Multiplies the model's loss by a per-example `loss_weight` when the
    collator includes one (see QwenJSONLMultiturnCollator) -- used for the
    loss-reweighted class-imbalance variant. A strict superset of the default
    Trainer: when every example's loss_weight is 1.0 (every pre-existing
    dataset), this is numerically identical to plain Trainer.compute_loss.

    Only exactly correct for batch_size=1 (this project's standard training
    config): HF's default loss is already mean-reduced across the whole batch,
    so a single scalar loss_weight is equivalent to true per-example weighting
    only when the batch contains exactly one example.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_weight = inputs.pop("loss_weight", None)
        outputs = model(**inputs)
        loss = outputs.loss
        if loss_weight is not None:
            if loss_weight.numel() != 1:
                raise ValueError("loss_weight-based training requires batch_size=1.")
            loss = loss * loss_weight.to(loss.device)
        return (loss, outputs) if return_outputs else loss


class EpochLoggerCallback(TrainerCallback):
    """Print clear epoch boundaries to stdout."""

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f">>> Starting Epoch {int(state.epoch or 0) + 1}/{args.num_train_epochs}")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f">>> Finished Epoch {int(state.epoch or 0)}/{args.num_train_epochs}\n")


class MemoryStatsCallback(TrainerCallback):
    """Log GPU memory at normal Trainer logging events."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(0) / 1e9
            mem_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"[MEM] Alloc: {mem_alloc:.2f} GB | Reserved: {mem_reserved:.2f} GB")


def parse_args():
    """Collect JSONL, model, vision-size, and LoRA/training arguments."""

    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Dataset root containing train/val/test split folders.",
    )
    p.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="JSONL suffix, e.g. candidate_verification for sft_train_candidate_verification.jsonl.",
    )
    p.add_argument("--train_jsonl", type=str, default=None, help="Explicit train JSONL path.")
    p.add_argument("--val_jsonl", type=str, default=None, help="Explicit validation JSONL path.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--deepspeed", type=str, default=None)

    p.add_argument("--image_max_pixels", type=int, default=768 * 28 * 28)
    p.add_argument("--image_min_pixels", type=int, default=256 * 28 * 28)

    p.add_argument("--train_bs", type=int, default=1)
    p.add_argument("--eval_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=32)

    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--freeze_visual", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--optim",
        type=str,
        default="adamw_torch_fused",
        choices=["adamw_torch", "adamw_torch_fused", "adamw_bnb_8bit"],
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    p.add_argument(
        "--eval_frequency_ratio",
        type=float,
        default=0.05,
        help="Eval every X fraction of total steps. 1.0 means every epoch; 0.05 means every 5 percent.",
    )
    p.add_argument("--early_stopping_patience", type=int, default=6)
    p.add_argument("--early_stopping_threshold", type=float, default=0.0)
    p.add_argument("--save_total_limit", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def resolve_jsonl_paths(args) -> tuple[Path, Path]:
    """Resolve train/validation JSONLs from explicit paths or an experiment suffix."""

    dataset_root = Path(args.dataset_root)
    if args.train_jsonl or args.val_jsonl:
        if not (args.train_jsonl and args.val_jsonl):
            raise ValueError("Pass both --train_jsonl and --val_jsonl, or neither.")
        train_jsonl = Path(args.train_jsonl)
        val_jsonl = Path(args.val_jsonl)
    elif args.experiment_name:
        train_jsonl = dataset_root / "train" / f"sft_train_{args.experiment_name}.jsonl"
        val_jsonl = dataset_root / "val" / f"sft_val_{args.experiment_name}.jsonl"
    else:
        train_jsonl = dataset_root / "train" / "sft_train.jsonl"
        val_jsonl = dataset_root / "val" / "sft_val.jsonl"

    for jsonl_path in (train_jsonl, val_jsonl):
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing dataset JSONL: {jsonl_path}")
    return train_jsonl, val_jsonl


def messages_to_qwen_text(processor, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    """Render chat messages exactly as Qwen expects before tokenization."""

    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def find_assistant_indices(
    messages: list[dict[str, Any]],
    supervise_mask: list[bool] | None = None,
) -> list[int]:
    """Return the indices of the assistant messages that should get loss.

    supervise_mask, when given, is aligned with the ordinal order of
    assistant messages (not raw message indices): True keeps that turn's
    span supervised, False drops it, for datasets that want a turn kept in
    the rendered context without being a loss target. None (the default)
    supervises every assistant turn.
    """

    assistant_indices = [
        i for i, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("No assistant message found in SFT example.")
    if supervise_mask is None:
        return assistant_indices
    if len(supervise_mask) != len(assistant_indices):
        raise ValueError(
            f"supervise_assistant_turns has {len(supervise_mask)} entries but "
            f"example has {len(assistant_indices)} assistant turns."
        )
    return [idx for idx, keep in zip(assistant_indices, supervise_mask) if keep]


class QwenJSONLMultiturnCollator:
    """Load images, tokenize chats, and supervise every assistant response span."""

    def __init__(self, processor):
        self.processor = processor

    def encoded_length(self, messages: list[dict[str, Any]], images, add_generation_prompt: bool) -> int:
        """Measure a single unpadded rendered conversation in processor tokens."""

        text = messages_to_qwen_text(
            self.processor,
            messages,
            add_generation_prompt=add_generation_prompt,
        )
        encoded = self.processor(
            text=[text],
            images=[images],
            padding=False,
            return_tensors="pt",
        )
        return int(encoded["attention_mask"][0].sum().item())

    def __call__(self, features):
        full_texts = []
        image_batches = []
        assistant_spans = []

        for ex in features:
            messages = ex["messages"]
            full_texts.append(messages_to_qwen_text(self.processor, messages, add_generation_prompt=False))
            images = [Image.open(p).convert("RGB") for p in ex["images"]]
            image_batches.append(images)

            spans = []
            supervise_mask = ex.get("supervise_assistant_turns")
            for assistant_idx in find_assistant_indices(messages, supervise_mask):
                # The rendered prefix includes Qwen's assistant marker; the
                # partial chat adds the ground-truth assistant JSON after it.
                start = self.encoded_length(
                    messages[:assistant_idx],
                    images,
                    add_generation_prompt=True,
                )
                end = self.encoded_length(
                    messages[:assistant_idx + 1],
                    images,
                    add_generation_prompt=False,
                )
                if end <= start:
                    raise ValueError(
                        "Assistant span has non-positive length for "
                        f"example {ex.get('id', '<unknown>')} at message {assistant_idx}"
                    )
                spans.append((start, end))
            assistant_spans.append(spans)

        batch = self.processor(
            text=full_texts,
            images=image_batches,
            padding=True,
            return_tensors="pt",
        )

        labels = torch.full_like(batch["input_ids"], -100)
        padding_side = getattr(self.processor.tokenizer, "padding_side", "right")

        for i, spans in enumerate(assistant_spans):
            sample_len = int(batch["attention_mask"][i].sum().item())
            offset = labels.shape[1] - sample_len if padding_side == "left" else 0
            for start, end in spans:
                labels[i, offset + start:offset + end] = batch["input_ids"][
                    i,
                    offset + start:offset + end,
                ]

        batch["labels"] = labels
        # Per-example loss-reweighting support (class-imbalance variant 4):
        # absent/1.0 for every pre-existing dataset, so this is a no-op unless
        # the jsonl explicitly sets "loss_weight" on a row. Uses an explicit
        # None-check, not just .get's default: when train/val jsonls are loaded
        # together via datasets.load_dataset and only one of them defines
        # "loss_weight", the library unifies the schema across splits and
        # backfills the other split's rows with an explicit null, which .get's
        # default does NOT catch (the key exists, it's just None).
        batch["loss_weight"] = torch.tensor(
            [float(w) if (w := ex.get("loss_weight")) is not None else 1.0 for ex in features],
            dtype=torch.float32,
        )
        return batch


def sanity_check_collator_labels(collator, example: dict[str, Any]):
    """Verify that every assistant JSON payload contributes supervised labels."""

    batch = collator([example])
    labels = batch["labels"][0]
    decoded = collator.processor.tokenizer.decode(
        batch["input_ids"][0][labels != -100],
        skip_special_tokens=False,
    )
    missing = []
    supervise_mask = example.get("supervise_assistant_turns")
    assistant_ordinal = -1
    for message in example["messages"]:
        if message.get("role") != "assistant":
            continue
        assistant_ordinal += 1
        if supervise_mask is not None and not supervise_mask[assistant_ordinal]:
            continue  # intentionally unsupervised turn (bucket B/C turn 1)
        target_text = "".join(
            item.get("text", "")
            for item in message.get("content", [])
            if item.get("type") == "text"
        )
        try:
            target = json.loads(target_text)
        except json.JSONDecodeError:
            target = target_text
        for key in ("instrument", "action", "target"):
            if json.dumps(key) in target_text and key not in decoded:
                missing.append(key)
        if isinstance(target, dict) and "predictions" in target and "predictions" not in decoded:
            missing.append("predictions")
    if missing:
        raise RuntimeError(f"Collator sanity check missed assistant labels: {missing}")
    print(
        "Collator sanity check passed: "
        f"{int((labels != -100).sum().item())} supervised tokens."
    )


def configure_processor(args):
    """Load the Qwen processor and apply the requested visual token budget."""

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    processor.image_processor.max_pixels = args.image_max_pixels
    processor.image_processor.min_pixels = args.image_min_pixels
    return processor


def enable_checkpointing_input_grads(model):
    """Keep the checkpointed QLoRA forward graph connected to trainable adapters."""

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    input_embeddings = model.get_input_embeddings()

    def make_inputs_require_grad(module, inputs, output):
        output.requires_grad_(True)

    input_embeddings.register_forward_hook(make_inputs_require_grad)


def load_lora_model(args):
    """Load Qwen3-VL in 4-bit and attach trainable LoRA adapters."""

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    if args.freeze_visual and hasattr(model, "model") and hasattr(model.model, "visual"):
        for p in model.model.visual.parameters():
            p.requires_grad = False
        print("Visual encoder frozen")

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    # Gradient checkpointing needs at least one tensor in the checkpointed path
    # to carry gradients. With frozen vision features plus 4-bit LoRA, this hook
    # keeps the language-model adapter path connected during backward.
    enable_checkpointing_input_grads(model)
    model.print_trainable_parameters()
    return model


def compute_eval_steps(num_train_examples: int, args) -> int:
    """Compute an eval/save cadence from dataset size and accumulation."""

    steps_per_epoch = num_train_examples // (args.train_bs * args.grad_accum)
    steps_per_epoch = max(1, steps_per_epoch)
    total_steps = steps_per_epoch * args.num_train_epochs
    if args.eval_frequency_ratio < 1.0:
        return max(1, int(total_steps * args.eval_frequency_ratio))
    return steps_per_epoch


def main():
    """Train LoRA adapters from raw JSONL chats and image paths."""

    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    train_jsonl, val_jsonl = resolve_jsonl_paths(args)
    data_files = {"train": str(train_jsonl), "validation": str(val_jsonl)}
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.cast_column("images", Sequence(Value("string")))
    print(f"Train JSONL: {train_jsonl}")
    print(f"Val JSONL: {val_jsonl}")
    print(f"Train samples: {len(dataset['train'])} | Val samples: {len(dataset['validation'])}")
    print(f"Image pixel caps: min={args.image_min_pixels} max={args.image_max_pixels}")

    processor = configure_processor(args)
    model = load_lora_model(args)
    data_collator = QwenJSONLMultiturnCollator(processor)
    sanity_check_collator_labels(data_collator, dataset["train"][0])

    eval_steps = compute_eval_steps(len(dataset["train"]), args)
    common_args = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        bf16=True,
        tf32=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=eval_steps,
        eval_steps=eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=max(1, eval_steps // 4),
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        gradient_checkpointing=True,
    )
    if args.deepspeed:
        if torch.cuda.device_count() <= 1:
            raise RuntimeError("DeepSpeed requires multiple GPUs.")
        training_args = TrainingArguments(deepspeed=args.deepspeed, **common_args)
    else:
        training_args = TrainingArguments(**common_args)

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        callbacks=[
            EpochLoggerCallback(),
            MemoryStatsCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            ),
        ],
    )

    last_checkpoint = None
    if args.resume and Path(args.output_dir).exists():
        last_checkpoint = get_last_checkpoint(args.output_dir)

    if last_checkpoint is not None:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found, starting fresh.")
        trainer.train()

    final_dir = Path(args.output_dir) / "final_checkpoint"
    trainer.save_model(str(final_dir))
    print(f"Training complete. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
