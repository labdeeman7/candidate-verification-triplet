# Stage 1: instrument instance segmentation

Stage 1 is a standard [MMDetection](https://github.com/open-mmlab/mmdetection)
Mask2Former, predicting per-frame instrument instance masks + classes. It is
not a custom architecture, so this directory is documentation rather than a
forked copy of MMDetection.

## Configuration

- **Backbone**: Swin-S.
- **Input**: 1280x1280, large-scale-jitter `RandomResize` (ratio 0.1-2.0) +
  `RandomCrop`, square.
- **Optimizer**: AdamW, eps=1e-8, betas=(0.9, 0.999); backbone learning rate
  multiplier 0.1 (effective backbone LR ~1e-5). Base LR 1e-4, weight decay
  0.05, gradient norm clipped to 0.01.
- **Schedule**: `MultiStepLR`, two decay milestones at ~89%/96.5% of total
  scheduled iterations, gamma=0.1. No warmup.
- **Effective batch size**: 4 (2 per GPU x 2 gradient-accumulation steps, 1
  GPU).
- **Classes**: 6 instrument classes (no "stuff"/background classes).

## Score threshold + NMS (important for reproducing the paper's numbers)

Proposals are filtered with:
- **Score threshold**: 0.4
- **Mask-NMS IoU threshold**: 1.0 (i.e. NMS is effectively disabled;
  duplicate suppression relies on the score threshold alone)

A frame where every detection falls below 0.4 produces **zero** Stage-1
proposals. `eval/score_ivt.py` handles this correctly (scores it as an
all-empty prediction against GT), but if you write your own Stage-1 export
step, make sure every GT frame still gets *some* output file — even an
empty one — or you will reproduce the silent-exclusion bug this release
already fixes (see `eval/score_ivt.py`'s module docstring).

## A gotcha worth not rediscovering

**Do not trust a detector's config file for its output label-index
ordering.** Two Mask2Former checkpoints trained on the same classes can use
genuinely different index-to-class orderings internally. Verify empirically
(decode raw predicted labels from a saved inference output and cross-check
per-index prediction counts against real per-class GT frequency) before
wiring a new detector's raw output into the class-order assumed by
`eval/convert_qwen_to_labelme.py` and Stage 2.

## Output format Stage 2 expects

Whatever detector you use, Stage 2 expects one LabelMe-style JSON per frame:

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "grasper",
      "group_id": 1,
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "polygon",
      "score": 0.91
    }
  ],
  "imagePath": "...", "imageHeight": 480, "imageWidth": 854
}
```

`group_id` is a per-class, 1-indexed instance id (assigned in
highest-score-first order among surviving instances of that class). A frame
with zero surviving instances should still get a file written, with an
empty `"shapes"` list — not be skipped.

## Training

Use MMDetection directly with a Mask2Former Swin-S config matching the
hyperparameters above, pointed at your own COCO-format instrument
segmentation annotations. See the MMDetection docs for the standard
training/inference commands (`tools/train.py`, `tools/test.py`).
