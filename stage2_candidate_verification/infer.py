#!/usr/bin/env python3
"""Evaluate the Candidate Verification Qwen3-VL LoRA checkpoint, asking one turn at a
time (instrument -> action -> target -> final) and feeding the model's own generated
response forward, rather than the single-shot generation used for Direct SFT.

Batches across examples at each turn (all examples in a batch take turn N together, then turn
N+1 together, etc.) -- confirmed via a real timing run that unbatched (one example at a time,
one turn at a time) generation takes ~46s/example, i.e. ~64 hours for a ~5000-example test set;
batching is essential to make a full test-set run practical. Uses the same left-padding pattern
as stage2_direct_sft/infer.py's batched single-shot generation, just repeated once per turn
instead of once per example.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Sequence, Value, load_dataset

from common.qwen_infer_shared import (
    add_shared_infer_args,
    configure_processor,
    get_model_device,
    load_example_images,
    load_model,
    remap_image_path,
    resolve_dataset_jsonl,
)
from common.sft_qwen_utils.multiturn_eval_helpers import (
    build_turn_user_text,
    extract_instance_ids_and_true_instruments,
    parse_field_by_id,
    safe_json_load,
)
from common.prompts import get_prompt
from common.prompts.types import MultiTurnDecompositionPrompt


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class ExampleState:
    """Per-example running state across the shared turn loop for one batch."""

    def __init__(self, ex: dict[str, Any], prompt, images: list):
        self.ex = ex
        self.images = images
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": prompt.system}]}
        ]
        self.instrument_names, self.true_instrument_by_id = extract_instance_ids_and_true_instruments(
            ex["messages"][-1]["content"][0]["text"]
        )
        self.generated_action_by_id: dict[str, str] = {}
        self.turn_texts: list[str] = []

    def append_user_turn(self, turn, turn_index: int) -> None:
        user_text = build_turn_user_text(
            turn, self.instrument_names, self.true_instrument_by_id, self.generated_action_by_id
        )
        content = []
        if turn_index == 0:
            content.append({"type": "image"})
        content.append({"type": "text", "text": user_text})
        self.messages.append({"role": "user", "content": content})

    def append_assistant_turn(self, turn, turn_text: str) -> None:
        self.turn_texts.append(turn_text)
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": turn_text}]})
        if turn.key == "action":
            self.generated_action_by_id = parse_field_by_id(turn_text, "action")

    def to_record(self, args) -> dict[str, Any]:
        final_text = self.turn_texts[-1]
        pred_obj, parse_success, parse_info = safe_json_load(final_text)
        gt_obj = json.loads(self.ex["messages"][-1]["content"][0]["text"])
        return {
            "id": self.ex["id"],
            "task": self.ex.get("task"),
            "gt": gt_obj,
            "pred": pred_obj,
            "parse_success": parse_success,
            "parse_info": parse_info,
            "raw_pred": final_text,
            "turn_texts": self.turn_texts,
            "images": [remap_image_path(path, args) for path in self.ex["images"]],
        }


def run_batch(model, processor, device, prompt, batch_examples, args) -> list[dict[str, Any]]:
    """Run the full turn sequence for a batch of examples, one turn at a time in lockstep."""

    states = [
        ExampleState(ex, prompt, load_example_images(ex["images"], args)) for ex in batch_examples
    ]

    for turn_index, turn in enumerate(prompt.turns):
        for state in states:
            state.append_user_turn(turn, turn_index)

        rendered = [
            processor.apply_chat_template(state.messages, tokenize=False, add_generation_prompt=True)
            for state in states
        ]
        images_batch = [state.images for state in states]
        batch = processor(text=rendered, images=images_batch, return_tensors="pt", padding=True)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)

        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                repetition_penalty=1.1,
            )
        # Left-padding (set in configure_processor) means every sequence in the
        # batch shares the same padded prompt length, so this single slice
        # correctly isolates each example's newly generated tokens.
        input_len = batch["input_ids"].shape[1]
        turn_texts = processor.batch_decode(
            generated[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        for state, turn_text in zip(states, turn_texts):
            state.append_assistant_turn(turn, turn_text.strip())

    return [state.to_record(args) for state in states]


def main():
    parser = argparse.ArgumentParser()
    add_shared_infer_args(parser)
    parser.add_argument(
        "--prompt_name",
        default="candidate_verification",
        choices=["candidate_verification"],
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--debug_examples", type=int, default=3)
    args = parser.parse_args()

    prompt = get_prompt(args.prompt_name)
    if not isinstance(prompt, MultiTurnDecompositionPrompt):
        raise ValueError(f"'{args.prompt_name}' is not a MultiTurnDecompositionPrompt")

    jsonl_path = resolve_dataset_jsonl(args)
    ds = load_dataset("json", data_files={"eval": str(jsonl_path)})["eval"]
    ds = ds.cast_column("images", Sequence(Value("string")))
    full_len = len(ds)
    print(f"Loaded {full_len} eval examples from {jsonl_path}")

    # Resume support: this eval has no mid-batch checkpointing of its own, and on a
    # preemptible cluster a multi-hour run can be killed and restarted many times before
    # it finishes. A sibling `.progress.jsonl` file records one line per completed example
    # as soon as its batch finishes, so a restarted run can skip everything already done
    # instead of regenerating the whole test set from scratch.
    out_path = Path(args.save_preds) if args.save_preds else None
    progress_path = out_path.with_name(out_path.name + ".progress.jsonl") if out_path else None

    saved: list[dict[str, Any]] = []
    if progress_path and progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    saved.append(json.loads(line))
        done_ids = {rec["id"] for rec in saved}
        if done_ids:
            ds = ds.filter(lambda ex: ex["id"] not in done_ids)
            print(f"Resuming from {progress_path}: {len(done_ids)} examples already done, "
                  f"{len(ds)} remaining")

    processor = configure_processor(args)
    model = load_model(args)
    device = get_model_device(model)
    debug_shown = 0

    if progress_path:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_f = progress_path.open("a", encoding="utf-8") if progress_path else None

    for start in range(0, len(ds), args.batch_size):
        batch_examples = ds.select(range(start, min(start + args.batch_size, len(ds))))
        records = run_batch(model, processor, device, prompt, batch_examples, args)

        if args.save_preds:
            saved.extend(records)
            for record in records:
                progress_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress_f.flush()
            os.fsync(progress_f.fileno())

        for record in records:
            if debug_shown < args.debug_examples:
                print("\n" + "=" * 90)
                print(f"[SAMPLE {debug_shown}] {record['id']}")
                print(f"Parse success: {record['parse_success']} | Info: {record['parse_info']}")
                for t, text in enumerate(record["turn_texts"]):
                    print(f"-- turn {t} --")
                    print(text[:800])
                print("=" * 90)
                debug_shown += 1

        processed = len(saved)
        if (start // args.batch_size + 1) % 5 == 0 or processed == full_len:
            print(f"Processed {processed}/{full_len} examples")

    if progress_f:
        progress_f.close()

    if args.save_preds:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(saved)} predictions to {out_path}")
        if progress_path.exists():
            progress_path.unlink()


if __name__ == "__main__":
    main()
