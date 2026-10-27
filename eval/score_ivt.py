#!/usr/bin/env python3
"""Score predicted LabelMe annotations against ground truth with the real
ivtmetrics AP_IVT (and AP_I/V/T/IV/IT) protocol.

Loops over the **ground-truth** directory, not the prediction directory: any
GT test frame missing a corresponding prediction file is scored as an
explicit empty prediction (so a real GT instance in that frame correctly
counts as a false negative) rather than being silently excluded from the
metric. An earlier version of this script looped over the prediction
directory instead, which meant a frame where the pipeline produced zero
output (detector found nothing, or every proposal was rejected) was invisible
to the scorer -- missing a real instance in such a frame could score as a
perfect 100 AP instead of being penalized. This version is the fix.
"""
import argparse
import json
import os
from functools import lru_cache
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ivtmetrics import Segmentation as ivtmetricsSegmentation
from common.tripletseg_dataset_utils.dataset_variables import TripletSegmentationVariables


def read_from_json(json_file_path: str):
    with open(json_file_path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _get_label_maps():
    inst_id_to_class = TripletSegmentationVariables.categories["instrument"]
    inst_class_to_id = {cls: int(i) for i, cls in inst_id_to_class.items()}

    trip_id_to_class = TripletSegmentationVariables.categories["triplet"]
    trip_class_to_id = {cls: int(i) for i, cls in trip_id_to_class.items()}

    return inst_class_to_id, trip_class_to_id


def _group_shapes_by_instance(json_dict: Dict[str, Any]):
    """Groups LabelMe shapes into {(class_name, instance_id): [shape_info, ...]}."""

    grouped = {}
    for s in json_dict.get("shapes", []):
        class_name = s["label"]
        instance_id = s["group_id"]

        info = {
            "contour_pts": s["points"],
            "score": s.get("score", 1.0),
            "verb": s["verb"],
            "verb_score": s.get("verb_score", 1.0),
            "target": s["target"],
            "target_score": s.get("target_score", 1.0),
        }

        grouped.setdefault((class_name, instance_id), []).append(info)

    return grouped


def _polygons_to_mask(polygons_xy: List[np.ndarray], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, polygons_xy, color=1)
    return mask


def get_triplets_for_segmentation_metric_calculation_in_img(
    json_dict: Dict[str, Any],
    image_shape: Optional[Tuple[int, int]] = None,
    warn_on_unknown_triplet: bool = True,
):
    """Converts a LabelMe json dict into the list format ivtmetrics expects."""

    inst_class_to_id, trip_class_to_id = _get_label_maps()

    if image_shape is None:
        h = TripletSegmentationVariables.height
        w = TripletSegmentationVariables.width
    else:
        h, w = image_shape

    grouped = _group_shapes_by_instance(json_dict)

    does_not_exist = set()
    out = []
    for (class_name, instance_id), infos in grouped.items():
        score = infos[0]["score"]
        verb = infos[0]["verb"]
        target = infos[0]["target"]

        triplet_name = f"{class_name},{verb},{target}"

        if triplet_name not in trip_class_to_id:
            if warn_on_unknown_triplet:
                print(f"triplet_name - {triplet_name} that does not exist")
                does_not_exist.add(triplet_name)
            triplet_name = f"{class_name},null_verb,null_target"

        triplet_id = int(trip_class_to_id[triplet_name]) - 1
        instrument_id = str(int(inst_class_to_id[class_name]) - 1)

        polygons = []
        for info in infos:
            pts = np.array(info["contour_pts"], dtype=np.int32).reshape((-1, 1, 2))
            polygons.append(pts)

        mask = _polygons_to_mask(polygons, h=h, w=w)

        out.append({"triplet": triplet_id, "instrument": [instrument_id, score, mask]})

    return out, does_not_exist


def run_ivt_metric_object_for_folder_segmentation(ivt_metric_object, pred_ann_dir, gt_ann_dir):
    """Scores every GT test frame, defaulting to an empty prediction when a
    frame has no corresponding prediction file (see module docstring)."""

    ivt_metric_object.reset_global()
    gt_ann_list = sorted(os.listdir(gt_ann_dir))
    pred_files_available = set(os.listdir(pred_ann_dir))

    n_missing = 0
    global_does_not_exist = set()

    for i, filename in enumerate(gt_ann_list):
        gt_ann_path = join(gt_ann_dir, filename)

        basename = filename.split(".")[0]
        basename = basename.replace("t50_", "")
        video_name, _ = basename.split("_")

        if i == 0:
            video_name_tracker = video_name
        if i > 0 and video_name_tracker != video_name:
            ivt_metric_object.video_end()
            video_name_tracker = video_name

        if filename in pred_files_available:
            json_dict_pred = read_from_json(join(pred_ann_dir, filename))
        else:
            n_missing += 1
            json_dict_pred = {"shapes": []}

        list_pred, does_not_exist_pred = get_triplets_for_segmentation_metric_calculation_in_img(json_dict_pred)
        global_does_not_exist = global_does_not_exist.union(does_not_exist_pred)

        json_dict_gt = read_from_json(gt_ann_path)
        list_gt, _ = get_triplets_for_segmentation_metric_calculation_in_img(json_dict_gt)

        ivt_metric_object.update(targets=[list_gt], predictions=[list_pred], format="dict")

    if n_missing:
        print(f"{n_missing}/{len(gt_ann_list)} GT frames had no prediction file "
              f"-- scored as empty predictions (not silently excluded).")

    return global_does_not_exist


def run_ivt_segmentation_evaluation(pred_ann_dir: str, gt_ann_dir: str, num_classes: int = 100):
    segmentation_object = ivtmetricsSegmentation(num_class=num_classes)

    global_does_not_exist = run_ivt_metric_object_for_folder_segmentation(
        ivt_metric_object=segmentation_object, pred_ann_dir=pred_ann_dir, gt_ann_dir=gt_ann_dir,
    )

    results = {key: segmentation_object.compute_video_AP(key) for key in ["i", "v", "t", "iv", "it", "ivt"]}
    return results, global_does_not_exist


def print_ivt_results(results: dict):
    for key in ["i", "v", "t", "iv", "it", "ivt"]:
        mAP = results[key]["mAP"] * 100
        print(f"results_{key}: {mAP:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate IVT segmentation mAP")
    parser.add_argument("--pred_ann_dir", type=str, required=True)
    parser.add_argument("--gt_ann_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=100)
    args = parser.parse_args()

    results, global_does_not_exist = run_ivt_segmentation_evaluation(
        pred_ann_dir=args.pred_ann_dir, gt_ann_dir=args.gt_ann_dir, num_classes=args.num_classes,
    )

    print_ivt_results(results)

    print(f"There are {len(global_does_not_exist)} triplets not found in the predefined triplet list:")
    for triplet_name in global_does_not_exist:
        print(triplet_name)


if __name__ == "__main__":
    main()
