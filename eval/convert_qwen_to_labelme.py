import json
from pathlib import Path
import sys
from copy import deepcopy

from collections import defaultdict
from common.sft_qwen_utils.converters import instrument_sft_to_labelme, verb_sft_to_labelme, target_sft_to_labelme
from common.sft_qwen_utils.json_recovery import build_recovered_reasoning_payload
from common.sft_qwen_utils.triplet_validator import validate_triplets_against_registry, _get_label_maps
from common.sft_qwen_utils.ontology import NOT_AN_INSTRUMENT


def parse_instance_key(instance_key: str):

    class_name, group_id = instance_key.rsplit("_", 1)
    return class_name, int(group_id)

def parse_image_id_from_instance_key(instance_key: str):
    """Extract the dataset split and canonical ``t50_VIDxx_frame`` image id."""
    if instance_key.endswith("_reasoning"):
        instance_key = instance_key[:-len("_reasoning")]

    parts = instance_key.split("_")

    # Some evaluation methods append a suffix such as `_candidate_verification`
    # after the frame id. Anchor on the VID token so those method suffixes do not
    # become part of the annotation filename.
    for vid_idx, part in enumerate(parts):
        if part.startswith("VID") and vid_idx + 1 < len(parts):
            image_start = max(0, vid_idx - 1)
            split_name = "_".join(parts[:image_start])
            image_id = "_".join(parts[image_start:vid_idx + 2])
            return split_name, image_id

    return parts[0], "_".join(parts[-3:])
    

def initialise_labelme_template(gt_json: dict):
    """Copy non-shape metadata if needed."""
    return {
        "version": gt_json.get("version", "5.0.1"),
        "flags": {},
        "shapes": [],
        "imagePath": gt_json.get("imagePath", ""),
        "imageHeight": gt_json.get("imageHeight", ""),
        "imageWidth": gt_json.get("imageWidth", ""),
    }


DIRECT_TASK = "multi_instance_triplet"
REASONING_TASK = "multi_instance_triplet_reasoning"


def get_prediction_kind(full_image_pred_dict: dict):
    task = full_image_pred_dict.get("task")
    pred = full_image_pred_dict.get("pred", {})

    if task == REASONING_TASK or "reasoning_predictions" in pred:
        return "reasoning"
    if task == DIRECT_TASK or "predictions" in pred:
        return "direct"
    return "unknown"


def normalise_predictions(full_image_pred_dict: dict):
    """
    Return plain triplet predictions regardless of whether the model emitted
    direct triplets or reasoning + final_triplet objects.
    """
    pred = full_image_pred_dict.get("pred", {})
    kind = get_prediction_kind(full_image_pred_dict)

    if kind == "direct":
        return deepcopy(pred.get("predictions", []))

    if kind == "reasoning":
        if "reasoning_predictions" not in pred and pred.get("raw_output"):
            recovered_pred = build_recovered_reasoning_payload(pred["raw_output"])
            if recovered_pred is not None:
                pred = recovered_pred

        normalised = []
        for reasoning_pred in pred.get("reasoning_predictions", []):
            final_triplet = reasoning_pred.get("final_triplet", {})
            normalised.append({
                "id": reasoning_pred.get("id"),
                "instrument": final_triplet.get("instrument"),
                "action": final_triplet.get("action"),
                "target": final_triplet.get("target"),
            })
        return normalised

    return []


def triplet_signature(pred: dict):
    return (
        pred.get("instrument"),
        pred.get("action"),
        pred.get("target"),
    )


def predictions_to_compare_map(full_image_pred_dict: dict):
    return {
        pred.get("id"): triplet_signature(pred)
        for pred in normalise_predictions(full_image_pred_dict)
        if pred.get("id")
    }


def write_direct_reasoning_failures(
    qwen_pred_list: list,
    out_pred_ann_dir: Path,
):
    direct_by_image = {}
    reasoning_by_image = {}

    for item in qwen_pred_list:
        image_id = item.get("id")
        if image_id is None:
            continue

        _, image_id_no_split = parse_image_id_from_instance_key(image_id)
        kind = get_prediction_kind(item)

        if kind == "direct":
            direct_by_image[image_id_no_split] = item
        elif kind == "reasoning":
            reasoning_by_image[image_id_no_split] = item

    if not direct_by_image or not reasoning_by_image:
        return

    failures = []
    all_image_ids = sorted(set(direct_by_image) | set(reasoning_by_image))

    for image_id in all_image_ids:
        direct_item = direct_by_image.get(image_id)
        reasoning_item = reasoning_by_image.get(image_id)

        if direct_item is None:
            failures.append({
                "image_id": image_id,
                "failure_type": "missing_direct",
                "direct": None,
                "reasoning": reasoning_item,
            })
            continue

        if reasoning_item is None:
            failures.append({
                "image_id": image_id,
                "failure_type": "missing_reasoning",
                "direct": direct_item,
                "reasoning": None,
            })
            continue

        direct_map = predictions_to_compare_map(direct_item)
        reasoning_map = predictions_to_compare_map(reasoning_item)

        if direct_map != reasoning_map:
            failures.append({
                "image_id": image_id,
                "failure_type": "triplet_mismatch",
                "direct_final_triplets": direct_map,
                "reasoning_final_triplets": reasoning_map,
                "direct": direct_item,
                "reasoning": reasoning_item,
            })

    failure_path = out_pred_ann_dir.parent / f"{out_pred_ann_dir.name}_direct_reasoning_failures.json"
    with open(failure_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved {len(failures)} direct/reasoning comparison failures to {failure_path}")


def split_predictions_by_kind(qwen_pred_list: list):
    direct_items = []
    reasoning_items = []
    unknown_items = []

    for item in qwen_pred_list:
        kind = get_prediction_kind(item)
        if kind == "direct":
            direct_items.append(item)
        elif kind == "reasoning":
            reasoning_items.append(item)
        else:
            unknown_items.append(item)

    return direct_items, reasoning_items, unknown_items



def convert_qwen_preds_to_labelme(
    qwen_pred_list: list,
    first_stage_prediction_ann_dir: Path,
    out_pred_ann_dir: Path,
):
    """
    Converts Qwen predictions into LabelMe-style annotations.

    first_stage_prediction_ann_dir:
        Output of stage-1 (currently GT, later a segmentation model).
    """
    out_pred_ann_dir.mkdir(parents=True, exist_ok=True)
    inst_class_to_id, trip_class_to_id = _get_label_maps()
    
    total_preds = 0
    total_invalid = 0
    total_rejected_fp = 0
    all_invalid_triplets = set()


    image_to_labelme = {}
    shape_cache = {}  # Cache to avoid duplicate loading

    for full_image_pred_dict in qwen_pred_list:
        image_id = full_image_pred_dict["id"]
        _, image_id_no_split = parse_image_id_from_instance_key(image_id)
        
        predictions = normalise_predictions(full_image_pred_dict)
        total_preds += len(predictions)

        # V2 (correct + reject) training can predict NOT_AN_INSTRUMENT to actively
        # reject a Stage-1 false-positive detection -- there's no real instrument class
        # or ground-truth triplet for this instance, so it's dropped here, before
        # validate_triplets_against_registry (which itself calls
        # instrument_sft_to_labelme and would raise on it, same as the per-prediction
        # conversion loop below would) -- same convention as dropping "snare".
        n_rejected = sum(1 for p in predictions if p.get("instrument") == NOT_AN_INSTRUMENT)
        total_rejected_fp += n_rejected
        predictions = [p for p in predictions if p.get("instrument") != NOT_AN_INSTRUMENT]

        # --- VALIDATE BEFORE CONVERSION ---
        predictions, invalid_set = validate_triplets_against_registry(
            predictions, trip_class_to_id, inst_class_to_id, warn_on_unknown=True
        )
        total_invalid += len(invalid_set)
        all_invalid_triplets.update(invalid_set)
        # ----------------------------------

        for pred in predictions:
            class_name_instance_key = pred["id"]
            stage_1_class_name, gt_group_id = parse_instance_key(class_name_instance_key)

            pred_instrument_sft = pred["instrument"]
            pred_verb_sft = pred["action"]
            pred_target_sft = pred["target"]
            
            pred_instrument_labelme = instrument_sft_to_labelme(pred_instrument_sft)
            pred_verb_labelme = verb_sft_to_labelme(pred_verb_sft)
            pred_target_labelme = target_sft_to_labelme(pred_target_sft)
            
            if pred_instrument_labelme != stage_1_class_name:
                print(f"Warning: Instrument mismatch for {class_name_instance_key}")
            
            stage1_path = first_stage_prediction_ann_dir / f"{image_id_no_split}.json"
            if not stage1_path.exists():
                raise FileNotFoundError(f"Missing stage-1 annotation: {stage1_path}")

            if image_id_no_split not in shape_cache:
                with open(stage1_path) as f:
                    stage1_json = json.load(f)
                shape_cache[image_id_no_split] = stage1_json["shapes"]
                image_to_labelme[image_id_no_split] = initialise_labelme_template(stage1_json)

            for shape in shape_cache[image_id_no_split]:
                if shape["label"] == stage_1_class_name and shape["group_id"] == gt_group_id:
                    new_shape = {
                        "label": stage_1_class_name,
                        "group_id": gt_group_id,
                        "points": shape["points"],
                        "shape_type": shape.get("shape_type", "polygon"),
                        "verb": pred_verb_labelme,
                        "target": pred_target_labelme,
                        "score": pred.get("score", 1.0),
                        "verb_score": pred.get("verb_score", 1.0),
                        "target_score": pred.get("target_score", 1.0),
                    }

                    image_to_labelme[image_id_no_split]["shapes"].append(new_shape)

    for img_id, lm_json in image_to_labelme.items():
        out_path = out_pred_ann_dir / f"{img_id}.json"
        with open(out_path, "w") as f:
            json.dump(lm_json, f, indent=2)
            


    if total_rejected_fp:
        print(f"Dropped {total_rejected_fp} NOT_AN_INSTRUMENT predictions (Stage-1 FP rejections) before scoring.")

    print(f'There are {len(all_invalid_triplets)} triplets not found in the predefined triplet list:')
    for triplet_name in all_invalid_triplets:
        print(triplet_name)

    parent_dir = out_pred_ann_dir.parent
    # get the folder name (last part of the path)
    folder_name = out_pred_ann_dir.name
    
    invalid_path = parent_dir / f"{folder_name}_invalid_triplets.txt"
    with open(invalid_path, "w") as f:
        for triplet in all_invalid_triplets:
            f.write(f"{triplet}\n")


import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen predictions to LabelMe-style annotations "
                    "using stage-1 instance geometry."
    )

    parser.add_argument(
        "--qwen_pred_json",
        type=Path,
        required=True,
        help="Path to Qwen prediction JSON file",
    )

    parser.add_argument(
        "--first_stage_prediction_ann_dir",
        type=Path,
        required=True,
        help="Directory containing stage-1 LabelMe annotations "
             "(currently GT, later segmentation model output)",
    )

    parser.add_argument(
        "--out_pred_ann_dir",
        type=Path,
        required=True,
        help="Output directory for direct generated LabelMe prediction files",
    )

    parser.add_argument(
        "--out_reasoning_pred_ann_dir",
        type=Path,
        default=None,
        help="Output directory for reasoning generated LabelMe prediction files. "
             "Defaults to '<out_pred_ann_dir>_reasoning' when both tasks are present.",
    )

    args = parser.parse_args()

    # Load Qwen predictions
    with open(args.qwen_pred_json) as f:
        qwen_pred_list = json.load(f)

    direct_items, reasoning_items, unknown_items = split_predictions_by_kind(qwen_pred_list)

    if unknown_items:
        print(f"Warning: skipped {len(unknown_items)} predictions with unknown schema/task.")

    if direct_items:
        convert_qwen_preds_to_labelme(
            qwen_pred_list=direct_items,
            first_stage_prediction_ann_dir=args.first_stage_prediction_ann_dir,
            out_pred_ann_dir=args.out_pred_ann_dir,
        )

    if reasoning_items:
        if direct_items:
            reasoning_out_dir = args.out_reasoning_pred_ann_dir
            if reasoning_out_dir is None:
                reasoning_out_dir = args.out_pred_ann_dir.parent / f"{args.out_pred_ann_dir.name}_reasoning"
        else:
            reasoning_out_dir = args.out_reasoning_pred_ann_dir or args.out_pred_ann_dir

        convert_qwen_preds_to_labelme(
            qwen_pred_list=reasoning_items,
            first_stage_prediction_ann_dir=args.first_stage_prediction_ann_dir,
            out_pred_ann_dir=reasoning_out_dir,
        )

    write_direct_reasoning_failures(
        qwen_pred_list=qwen_pred_list,
        out_pred_ann_dir=args.out_pred_ann_dir,
    )

    print(
        f"[OK] Converted {len(direct_items)} direct and {len(reasoning_items)} reasoning "
        "image predictions into LabelMe files."
    )


if __name__ == "__main__":
    main()

