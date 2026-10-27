#!/usr/bin/env python3
"""Evaluate a Qwen3-VL LoRA checkpoint directly from SFT JSONL image paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Sequence, Value, load_dataset

from common.qwen_infer_shared import (
    add_shared_infer_args,
    build_generation_prompt,
    configure_processor,
    get_model_device,
    load_example_images,
    load_model,
    remap_image_path,
    resolve_dataset_jsonl,
    safe_json_load,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    """Run greedy generation over an SFT JSONL and optionally save predictions."""

    parser = argparse.ArgumentParser()
    add_shared_infer_args(parser)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--debug_batches", type=int, default=2)
    args = parser.parse_args()

    jsonl_path = resolve_dataset_jsonl(args)
    ds = load_dataset("json", data_files={"eval": str(jsonl_path)})["eval"]
    ds = ds.cast_column("images", Sequence(Value("string")))
    full_len = len(ds)
    print(f"Loaded {full_len} eval examples from {jsonl_path}")

    # Resume support: a sibling `.progress.jsonl` file records one line per completed
    # example as soon as it's generated, so a preempted/restarted run can skip everything
    # already done instead of regenerating the whole test set from scratch.
    out_path = Path(args.save_preds) if args.save_preds else None
    progress_path = out_path.with_name(out_path.name + ".progress.jsonl") if out_path else None

    saved = []
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

    if progress_path:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_f = progress_path.open("a", encoding="utf-8") if progress_path else None

    for start in range(0, len(ds), args.batch_size):
        batch_examples = ds.select(range(start, min(start + args.batch_size, len(ds))))
        prompts = []
        image_batches = []

        for ex in batch_examples:
            prompts.append(build_generation_prompt(processor, ex["messages"]))
            image_batches.append(load_example_images(ex["images"], args))

        batch = processor(
            text=prompts,
            images=image_batches,
            return_tensors="pt",
            padding=True,
        )
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
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

        input_len = batch["input_ids"].shape[1]
        generated_only = generated[:, input_len:]
        output_texts = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for j, ex in enumerate(batch_examples):
            gt_text = ex["messages"][-1]["content"][0]["text"]
            gt_obj = json.loads(gt_text)
            assistant_text = output_texts[j].strip()
            pred_obj, parse_success, parse_info = safe_json_load(assistant_text)

            if args.save_preds:
                record = {
                    "id": ex["id"],
                    "task": ex.get("task"),
                    "gt": gt_obj,
                    "pred": pred_obj,
                    "parse_success": parse_success,
                    "parse_info": parse_info,
                    "raw_pred": assistant_text,
                    "prompt": prompts[j],
                    "images": [remap_image_path(path, args) for path in ex["images"]],
                    "temporal_metadata": ex.get("temporal_metadata"),
                }
                saved.append(record)
                progress_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if start // args.batch_size < args.debug_batches:
                print("\n" + "=" * 90)
                print(f"[BATCH {start // args.batch_size:02d} | SAMPLE {j}] {ex['id']}")
                print(f"Images: {len(ex['images'])} | Parse success: {parse_success} | Info: {parse_info}")
                print("-" * 90)
                print(assistant_text[:2000])
                print("=" * 90)

        if args.save_preds:
            progress_f.flush()
            os.fsync(progress_f.fileno())

        if (start // args.batch_size + 1) % 5 == 0:
            print(f"Processed {len(saved)}/{full_len} examples")

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
