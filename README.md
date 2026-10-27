# Surgical Action Triplet Detection via Candidate Verification

Code release for the IPCAI paper on a two-stage pipeline for surgical
action-triplet (instrument, verb, target) detection in laparoscopic
cholecystectomy video: **Stage 1** (Mask2Former) segments instrument
instances; **Stage 2** (Qwen3-VL-8B, QLoRA fine-tuned) predicts each
instance's verb and target via a 4-turn "Candidate Verification" dialogue
(instrument -> action -> target -> final triplet), optionally rejecting a
Stage-1 false positive outright via a `NOT_AN_INSTRUMENT` pseudo-class.

This repo contains the headline method (Candidate Verification), the Direct
SFT single-shot baseline, and the ResNet-50 interaction-classifier negative
control used as contrast points in the paper. It does not contain the full
experimental history (ablations, abandoned directions, in-progress work) —
only what's needed to train and evaluate the three reported methods.

## Headline results (AP$_{IVT}$, CholecTriplet-Seg test split)

| Method | GT-grounded | Complete pipeline |
|---|---|---|
| **Candidate Verification (ours)** | 16.51 | **15.22** |
| Direct SFT (single-shot baseline) | 13.94 ± 0.49 | 14.20 ± 1.08 |
| ResNet-50 interaction classifier (negative control) | 13.24 ± 1.40 | 11.84 ± 0.90 |
| TargetFusionNet (single-stage joint baseline) | - | 13.51 |

GT-grounded = Stage 2 shown ground-truth instrument masks (isolates Stage
2's semantic prediction quality). Complete pipeline = Stage 2 shown Stage
1's actual predicted masks (the real deployed system). Direct SFT / ResNet
numbers are 3-seed mean ± sample std.

## Repository layout

```
stage1_mask2former/            # Stage-1 detector: config + documentation (standard MMDetection Mask2Former)
stage2_candidate_verification/ # headline method
  train.py                     # QLoRA fine-tuning (shared trainer, also used by stage2_direct_sft)
  infer.py                     # 4-turn batched inference
stage2_direct_sft/              # single-shot baseline
  train.py
  infer.py
baseline_resnet/                # ResNet-50 negative-control baseline
  train.py
  infer.py
eval/
  convert_qwen_to_labelme.py    # turn a Qwen prediction JSON into scoreable LabelMe annotations
  score_ivt.py                  # real AP_I/V/T/IV/IT/IVT scoring (ivtmetrics)
common/                         # shared ontology, label-space, and prompt code
ivtmetrics/                     # vendored CAMMA ivtmetrics (https://github.com/CAMMA-public/ivtmetrics)
```

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD   # every script uses absolute imports rooted at the repo root
```

### Dataset format

Each split directory (`train/`, `val/`, `test/`) should contain:
- `img_dir/` — frame images
- `ann_dir/` — one LabelMe-style JSON per frame (see
  `stage1_mask2former/README.md` for the exact schema)

Training/eval JSONLs (one example per line, Qwen chat-message format) are
built from these annotations; this release does not include the
dataset-construction scripts, since they're specific to the internal data
layout — point `common/sft_qwen_utils` at your own JSONL-building step, or
open an issue if you need the exact format documented further.

## Training

```bash
# Candidate Verification (headline method)
python stage2_candidate_verification/train.py \
  --dataset_root /path/to/dataset \
  --train_jsonl /path/to/sft_train_candidate_verification.jsonl \
  --val_jsonl /path/to/sft_val_candidate_verification.jsonl \
  --output_dir ./runs/candidate_verification

# Direct SFT (baseline)
python stage2_direct_sft/train.py \
  --dataset_root /path/to/dataset \
  --train_jsonl /path/to/sft_train_baseline.jsonl \
  --val_jsonl /path/to/sft_val_baseline.jsonl \
  --output_dir ./runs/direct_sft

# ResNet-50 interaction classifier (negative control)
python baseline_resnet/train.py \
  --dataset_root /path/to/dataset \
  --label_mode vt \
  --output_dir ./runs/resnet_vt --seed 42 --epochs 15
```

Default hyperparameters match the paper: QLoRA r=64/alpha=128, effective
batch size 32 (bs=1 x grad_accum=32), AdamW-fused, lr=2e-4, linear decay with
3% warmup, 5 epochs with early stopping.

## Inference

```bash
python stage2_candidate_verification/infer.py \
  --lora_dir ./runs/candidate_verification/final_checkpoint \
  --dataset_root /path/to/dataset --dataset_jsonl /path/to/test.jsonl \
  --save_preds ./eval_results/cv/test_predictions.json --batch_size 8

python stage2_direct_sft/infer.py \
  --lora_dir ./runs/direct_sft/final_checkpoint \
  --dataset_root /path/to/dataset --dataset_jsonl /path/to/test.jsonl \
  --save_preds ./eval_results/direct/test_predictions.json --batch_size 8

python baseline_resnet/infer.py \
  --checkpoint ./runs/resnet_vt/best_model.pt --label_mode vt \
  --first_stage_prediction_ann_dir /path/to/test/ann_dir \
  --img_dir /path/to/test/img_dir \
  --out_pred_ann_dir ./eval_results/resnet_vt/pred_ann_dir
```

Point `--first_stage_prediction_ann_dir` (for `baseline_resnet`) or the
Stage-1 ann_dir used to build a Qwen test JSONL at either the real
`test/ann_dir` (GT-grounded) or a Stage-1 predicted-grounding split
(complete pipeline).

## Scoring

```bash
# Qwen-based methods: convert the raw prediction JSON to LabelMe first
python eval/convert_qwen_to_labelme.py \
  --qwen_pred_json ./eval_results/cv/test_predictions.json \
  --first_stage_prediction_ann_dir /path/to/stage1/ann_dir \
  --out_pred_ann_dir ./eval_results/cv/pred_ann_dir

# Then score any method's pred_ann_dir against GT
python eval/score_ivt.py \
  --pred_ann_dir ./eval_results/cv/pred_ann_dir \
  --gt_ann_dir /path/to/test/ann_dir
```

`score_ivt.py` scores every GT test frame, including any frame your
pipeline produced no output for (detector found nothing, or every proposal
was rejected) — these are scored as empty predictions, not silently
excluded. See the module's docstring for why this matters.

## Citation

If you use this code, please cite the paper (citation details added on
publication).

## License

MIT (see `LICENSE`).
