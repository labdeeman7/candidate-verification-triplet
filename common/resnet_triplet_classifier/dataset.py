"""Per-instance crop dataset for the ResNet triplet classifier.

Builds one example per annotated instrument instance: a padded-bbox crop of the
instance (enough surrounding tissue context for verb/target to be inferable,
not just a tight crop of the instrument itself -- a tight crop would silently
cripple the baseline and defeat the point of testing it fairly) plus its
100-class triplet label, for ground-truth-grounded training/eval.

For complete-pipeline eval, the same crop logic runs against Stage 1's own
predicted instances (predicted polygon, no verb/target) instead of GT ones --
see `build_instances_for_eval` with `source="predicted"`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .triplet_classes import triplet_to_index, vt_to_index

IMG_H = 480
IMG_W = 854
CROP_SIZE = 224
# Fraction of the instance bbox's larger dimension added as padding on each
# side, so the crop includes nearby tissue -- verb/target depend on what the
# instrument is touching, not just the instrument's own appearance.
PAD_FRAC = 0.6

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = T.Compose(
    [
        T.Resize((CROP_SIZE, CROP_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

EVAL_TRANSFORM = T.Compose(
    [
        T.Resize((CROP_SIZE, CROP_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def _bbox_from_points(points: list[list[float]]) -> tuple[int, int, int, int]:
    arr = np.array(points)
    x0, y0 = arr[:, 0].min(), arr[:, 1].min()
    x1, y1 = arr[:, 0].max(), arr[:, 1].max()
    return x0, y0, x1, y1


def _padded_crop_box(points: list[list[float]]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _bbox_from_points(points)
    w, h = x1 - x0, y1 - y0
    pad = PAD_FRAC * max(w, h)
    x0 = max(0, int(x0 - pad))
    y0 = max(0, int(y0 - pad))
    x1 = min(IMG_W, int(x1 + pad))
    y1 = min(IMG_H, int(y1 + pad))
    # Clamping each edge independently can invert the box when a polygon's
    # padded bbox extends entirely past one side of the image (e.g. a
    # near-top-edge annotation with enough padding to push both y0 and y1
    # negative before clamping) -- a handful of real annotations do this.
    # Enforce a minimum 1px box as a final safety net rather than silently
    # dropping or crashing on these.
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return x0, y0, x1, y1


class TripletCropDataset(Dataset):
    """One example per GT instance: (cropped image, class index).

    ``label_mode="triplet"`` (default) classifies the full 100-way
    (instrument, verb, target) triplet from the crop. ``label_mode="vt"``
    classifies only the 56-way (verb, target) interaction, dropping the
    instrument -- the fair two-stage framing, matching how Stage 1 supplies
    the instrument for every other Stage-2 method in this project.
    """

    def __init__(self, ann_dir: str, img_dir: str, train: bool = True, label_mode: str = "triplet"):
        assert label_mode in ("triplet", "vt")
        self.img_dir = Path(img_dir)
        self.label_mode = label_mode
        self.samples = []  # list of (ann_path, shape_index)
        ann_dir = Path(ann_dir)
        for ann_path in sorted(ann_dir.glob("*.json")):
            d = json.loads(ann_path.read_text(encoding="utf-8"))
            for i, shape in enumerate(d.get("shapes", [])):
                instrument = shape.get("label")
                verb = shape.get("verb")
                target = shape.get("target")
                if not instrument or not verb or not target:
                    continue
                try:
                    if label_mode == "vt":
                        vt_to_index(verb, target)
                    else:
                        triplet_to_index(instrument, verb, target)
                except KeyError:
                    continue
                self.samples.append((ann_path, i))
        self.transform = TRAIN_TRANSFORM if train else EVAL_TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ann_path, shape_idx = self.samples[idx]
        d = json.loads(ann_path.read_text(encoding="utf-8"))
        shape = d["shapes"][shape_idx]
        img_name = ann_path.stem + ".png"
        img = Image.open(self.img_dir / img_name).convert("RGB")
        x0, y0, x1, y1 = _padded_crop_box(shape["points"])
        crop = img.crop((x0, y0, x1, y1))
        if self.label_mode == "vt":
            label = vt_to_index(shape["verb"], shape["target"])
        else:
            label = triplet_to_index(shape["label"], shape["verb"], shape["target"])
        return self.transform(crop), label


def build_eval_crops(ann_path: Path, img_dir: Path):
    """Yield (group_id, crop_tensor) for every instance in one ann file, for
    inference. Works for both GT ann (verb/target present, ignored here) and
    predicted ann (verb/target are None) -- only `points`/`group_id`/`label`
    (instrument, used only for crop geometry, not as the predicted label) are
    needed to build the crop.
    """
    d = json.loads(ann_path.read_text(encoding="utf-8"))
    img_name = ann_path.stem + ".png"
    img = Image.open(img_dir / img_name).convert("RGB")
    out = []
    for shape in d.get("shapes", []):
        x0, y0, x1, y1 = _padded_crop_box(shape["points"])
        crop = img.crop((x0, y0, x1, y1))
        out.append((shape, EVAL_TRANSFORM(crop)))
    return d, out
