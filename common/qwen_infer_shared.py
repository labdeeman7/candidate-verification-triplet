#!/usr/bin/env python3
"""Shared inference helpers used by both stage2_direct_sft/infer.py and
stage2_candidate_verification/infer.py: model/processor loading, JSON-output
recovery, and dataset/image path resolution.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from common.sft_qwen_utils.json_recovery import build_recovered_reasoning_payload


def add_shared_infer_args(parser) -> None:
    """Register the CLI arguments common to both inference scripts."""

    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--lora_dir", required=True)
    parser.add_argument("--dataset_root", required=True,
                         help="Local path to your copy of the triplet-segmentation dataset.")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--dataset_jsonl", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--save_preds", default=None, help="Path to save predictions as JSON")
    parser.add_argument("--image_max_pixels", type=int, default=256 * 256)
    parser.add_argument("--image_min_pixels", type=int, default=64 * 64)
    parser.add_argument(
        "--attn_implementation", type=str, default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument(
        "--image_root_prefix", default=None,
        help="If the dataset JSONL's image paths were recorded with a different machine's "
             "absolute prefix, pass it here to rewrite it to --dataset_root at load time.",
    )


def safe_json_load(text: str) -> tuple[dict[str, Any], bool, str | None]:
    """Parse model JSON output, with a few recovery paths for common LLM mistakes."""

    raw_text = text.strip()
    raw_text = re.sub(r"```json", "", raw_text)
    raw_text = re.sub(r"```", "", raw_text).strip()

    try:
        return json.loads(raw_text), True, None
    except Exception as e:
        first_error = str(e)

    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0)), True, "recovered_from_extraction"
    except Exception:
        pass

    try:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", raw_text)
        return json.loads(cleaned), True, "recovered_trailing_commas"
    except Exception:
        pass

    recovered_reasoning = build_recovered_reasoning_payload(raw_text)
    if recovered_reasoning is not None:
        return recovered_reasoning, True, "recovered_reasoning_final_triplets"

    return {"parse_failed": True, "raw_output": raw_text}, False, first_error


def resolve_dataset_jsonl(args) -> Path:
    """Resolve the eval JSONL from an explicit path or dataset_root/split/experiment."""

    if args.dataset_jsonl:
        jsonl_path = Path(args.dataset_jsonl)
    else:
        if not args.experiment_name:
            raise ValueError("Pass --dataset_jsonl or --experiment_name.")
        jsonl_path = (
            Path(args.dataset_root)
            / args.split
            / f"sft_{args.split}_{args.experiment_name}.jsonl"
        )

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing eval JSONL: {jsonl_path}")
    return jsonl_path


def configure_processor(args):
    """Load Qwen's processor with the same image caps used for training."""

    processor = AutoProcessor.from_pretrained(
        args.model_name, trust_remote_code=True, use_fast=True,
    )
    processor.tokenizer.padding_side = "left"
    processor.image_processor.max_pixels = args.image_max_pixels
    processor.image_processor.min_pixels = args.image_min_pixels
    return processor


def load_model(args):
    """Load base Qwen and attach the trained LoRA adapter."""

    base = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()
    return model


def get_model_device(model) -> torch.device:
    """Use the first parameter device as the target device for input tensors."""

    return next(model.parameters()).device


def build_generation_prompt(processor, messages: list[dict[str, Any]]) -> str:
    """Drop the assistant target and ask Qwen to generate it."""

    return processor.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True,
    )


def remap_image_path(path: str, args) -> str:
    """Rewrite a recorded absolute image path to this machine's dataset_root, if configured."""

    if args.image_root_prefix:
        return path.replace(args.image_root_prefix, str(args.dataset_root), 1)
    return path


def load_example_images(image_paths: list[str], args) -> list[Image.Image]:
    """Load all images for one SFT example, preserving temporal order."""

    fixed_paths = [remap_image_path(path, args) for path in image_paths]
    return [Image.open(path).convert("RGB") for path in fixed_paths]
