"""Train a plain ResNet-50 triplet classifier on ground-truth-grounded
instrument-instance crops -- the simplest possible Stage-2 design, used as a
deliberate negative/contrast point: does a careful decomposition+alignment
design (Candidate Verification + Stage-1 alignment) actually matter, or would
any two-stage pipeline do about as well?

``--label_mode triplet`` (default, 100-way) mirrors Direct SFT's framing
(predict the full triplet per localized instance in one shot) but with a
generic CNN classifier instead of a fine-tuned VLM, and no decomposition at
all -- including re-predicting the instrument from the crop, unlike every
other Stage-2 method in this project. ``--label_mode vt`` (56-way) fixes
this: the instrument is assumed given by Stage 1, matching how Direct SFT and
Candidate Verification are actually evaluated, and the classifier only
predicts the interaction.

Checkpointing/resume: saves a single rolling checkpoint each epoch
(``checkpoint_latest.pt``, model+optimizer+epoch) and resumes from it
automatically if present, so a preemption on a shared cluster costs at most
one epoch rather than restarting training from scratch.

Run (from inside baseline_resnet/):
  python train.py \
      --dataset_root /path/to/triplet_segmentation_dataset \
      --label_mode vt \
      --output_dir ./runs/resnet_vt_seed42 \
      --seed 42 --epochs 15
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from common.resnet_triplet_classifier.dataset import TripletCropDataset
from common.resnet_triplet_classifier.triplet_classes import NUM_CLASSES, NUM_VT_CLASSES


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--label_mode", choices=["triplet", "vt"], default="triplet",
                    help="'triplet': 100-way (instrument,verb,target), re-predicting the "
                         "instrument. 'vt': 56-way (verb,target) only, assuming the "
                         "instrument is supplied by Stage 1, matching every other "
                         "Stage-2 method in this project.")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = NUM_VT_CLASSES if args.label_mode == "vt" else NUM_CLASSES

    train_ds = TripletCropDataset(
        ann_dir=f"{args.dataset_root}/train/ann_dir",
        img_dir=f"{args.dataset_root}/train/img_dir",
        train=True,
        label_mode=args.label_mode,
    )
    val_ds = TripletCropDataset(
        ann_dir=f"{args.dataset_root}/val/ann_dir",
        img_dir=f"{args.dataset_root}/val/img_dir",
        train=False,
        label_mode=args.label_mode,
    )
    print(f"Train instances: {len(train_ds)} | Val instances: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ckpt_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
    start_epoch = 0
    best_val_acc = 0.0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resuming from checkpoint: epoch {start_epoch}/{args.epochs}, best_val_acc={best_val_acc:.4f}")
    else:
        print("No checkpoint found, starting fresh.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_total += images.size(0)
        train_loss = total_loss / n_total
        train_acc = n_correct / n_total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)
        val_acc = val_correct / max(val_total, 1)
        print(f">>> Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        best_val_acc = max(best_val_acc, val_acc)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "val_acc": val_acc,
            },
            ckpt_path,
        )
        if val_acc == best_val_acc:
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))

    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    print(f"Training complete. Final model saved to {args.output_dir}/final_model.pt")


if __name__ == "__main__":
    main()
