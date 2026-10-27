"""Evaluate a trained ResNet triplet classifier, GT-grounded or complete-pipeline.

GT-grounded (``--first_stage_prediction_ann_dir`` points at the real
``test/ann_dir``): crops are built from real ground-truth instance masks.
Complete-pipeline (``--first_stage_prediction_ann_dir`` points at a Stage-1
predicted-grounding split's ``ann_dir``, e.g.
``mask2former_predicted_grounding_swin_s_hires_v1/ann_dir``): crops are built
from Stage 1's own predicted instance masks, exactly mirroring how Direct
SFT/Candidate Verification's complete-pipeline numbers are produced.

``--label_mode triplet`` (default): the classifier predicts the full
(instrument, verb, target) triplet per instance, regardless of what Stage 1's
own label said. ``--label_mode vt`` (the interaction classifier reported in
the paper): the classifier predicts only (verb, target); the instrument
field is always taken from Stage 1's own grounding, never re-predicted --
the same convention ``eval/convert_qwen_to_labelme.py`` uses for the
VLM-based methods, making this the fair two-stage comparison point.

Writes a LabelMe-format ``pred_ann_dir`` directly (no intermediate conversion
step needed, since there's no free-text parsing involved), ready for
``eval/score_ivt.py`` to score with real ivtmetrics.

Run (from inside baseline_resnet/):
  python infer.py \
      --checkpoint ./runs/resnet_vt_seed42/best_model.pt \
      --label_mode vt \
      --first_stage_prediction_ann_dir /path/to/test/ann_dir \
      --img_dir /path/to/test/img_dir \
      --out_pred_ann_dir ./eval_results/resnet_vt_seed42_gt/pred_ann_dir
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from train import build_model
from common.resnet_triplet_classifier.dataset import build_eval_crops
from common.resnet_triplet_classifier.triplet_classes import (
    NUM_CLASSES,
    NUM_VT_CLASSES,
    index_to_triplet,
    index_to_vt,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--first_stage_prediction_ann_dir", required=True,
                    help="GT test/ann_dir for GT-grounded, or a predicted-grounding "
                         "split's ann_dir for complete-pipeline.")
    p.add_argument("--img_dir", required=True)
    p.add_argument("--out_pred_ann_dir", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--label_mode", choices=["triplet", "vt"], default="triplet",
                    help="Must match the checkpoint's training label_mode. 'vt': only "
                         "verb/target are predicted; the instrument is kept from the "
                         "input shape's own label (GT or Stage-1 grounding), never "
                         "re-predicted -- same convention as convert_qwen_to_labelme.py.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = NUM_VT_CLASSES if args.label_mode == "vt" else NUM_CLASSES
    model = build_model(num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    in_dir = Path(args.first_stage_prediction_ann_dir)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_pred_ann_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_files, n_instances = 0, 0
    with torch.no_grad():
        for ann_path in sorted(in_dir.glob("*.json")):
            d, crops = build_eval_crops(ann_path, img_dir)
            n_files += 1
            if not crops:
                (out_dir / ann_path.name).write_text(json.dumps(d), encoding="utf-8")
                continue

            batch = torch.stack([c for _, c in crops]).to(device)
            preds = []
            for i in range(0, batch.size(0), args.batch_size):
                logits = model(batch[i:i + args.batch_size])
                preds.extend(logits.argmax(dim=1).cpu().tolist())

            new_shapes = []
            for (orig_shape, _), pred_idx in zip(crops, preds):
                shape = copy.deepcopy(orig_shape)
                if args.label_mode == "vt":
                    # Instrument inherited from Stage 1's own grounding (GT or
                    # predicted), never re-predicted -- matches how every other
                    # Stage-2 method's instrument field is handled.
                    verb, target = index_to_vt(pred_idx)
                else:
                    instrument, verb, target = index_to_triplet(pred_idx)
                    shape["label"] = instrument
                shape["verb"] = verb
                shape["target"] = target
                new_shapes.append(shape)
                n_instances += 1

            out_d = dict(d)
            out_d["shapes"] = new_shapes
            (out_dir / ann_path.name).write_text(json.dumps(out_d), encoding="utf-8")

    print(f"Done: {n_files} files, {n_instances} instances classified -> {out_dir}")


if __name__ == "__main__":
    main()
