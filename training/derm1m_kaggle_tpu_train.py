"""
derm1m_kaggle_tpu_train.py — Derm1M Skin Disease Classifier (TPU v3-8 / Kaggle)
==================================================================================

Designed for Kaggle TPU v3-8 (8 cores) using PyTorch/XLA + WebDataset.
Streams .tar shards directly from a GCS bucket so there is no I/O bottleneck.

The val split is derived from the training shards themselves: the last
VAL_SHARD_FRACTION (default 10%) of shards are held out for validation.
No separate val tar files are needed.

Setup on Kaggle
---------------
1. Upload the .tar shards (from scripts/tar_conversion.py) and class_mapping.json
   to a GCS bucket, or attach them as a Kaggle dataset.
2. Set GCS_SHARD_URLS below to all your training shards.
3. Set CLASS_MAPPING_PATH to wherever class_mapping.json lives.
4. Enable TPU in the Kaggle notebook accelerator settings.
5. Install requirements in the notebook cell before running:
       !pip install timm webdataset albumentations torch_xla -q

Notes
-----
- Effective batch size = BATCH_SIZE (per core) × 8 cores.
- bfloat16 is handled natively by XLA; no GradScaler is needed.
- Only the master core (rank 0) prints logs and saves checkpoints.
"""

import json
import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import webdataset as wds
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

# ── XLA imports ───────────────────────────────────────────────────────────────
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit before running
# ─────────────────────────────────────────────────────────────────────────────

import glob

# Path to the folder containing your uploaded .tar shards on Kaggle.
# When you add a Kaggle dataset, your files appear under /kaggle/input/<dataset-name>/
SHARD_INPUT_DIR = "/kaggle/input/derm1m-shards"

# Auto-discover all train shards from the input directory.
# Sorted so the shard split is deterministic across runs.
SHARD_URLS = sorted(glob.glob(f"{SHARD_INPUT_DIR}/derm1m_train_*.tar"))

# Fraction of shards reserved for validation (0.1 = last 10% of shards)
VAL_SHARD_FRACTION = 0.10

# Path to the class_mapping.json saved by tar_conversion.py
CLASS_MAPPING_PATH = f"{SHARD_INPUT_DIR}/class_mapping.json"

# Training hyper-parameters
IMG_SIZE        = 380
BATCH_SIZE      = 32          # Per TPU core; effective = 32 × 8 = 256
EPOCHS          = 40
LEARNING_RATE   = 1e-4        # Good baseline for effective batch of 256
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
PATIENCE        = 8

# Adjust to (total_train_images // effective_batch_size).
# With 10% held-out for val, ~90% of ~410K images are used for training.
STEPS_PER_EPOCH     = 1440    # ~369 000 / 256  ≈ 1 440
VAL_STEPS_PER_EPOCH = 160     # ~41  000 / 256  ≈  160

CHECKPOINT_DIR = "/kaggle/working/"

# ─────────────────────────────────────────────────────────────────────────────



# ── Load class mapping ────────────────────────────────────────────────────────

def load_class_mapping(path: str) -> tuple[dict, int]:
    with open(path) as f:
        mapping = json.load(f)   # {str(idx): class_name}
    return mapping, len(mapping)


# ── Transforms (built once — NOT inside the per-sample function) ──────────────

def _make_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                             val_shift_limit=20, p=0.4),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(
            num_holes_range=(4, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            p=0.3,
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _make_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── WebDataset sample decoders ────────────────────────────────────────────────

# Build transforms once at module level (not per sample!)
_TRAIN_TRANSFORM = _make_train_transform()
_VAL_TRANSFORM   = _make_val_transform()


def _decode_label(raw) -> int:
    """Convert bytes b'42', string '42', or int 42 → int 42."""
    if isinstance(raw, (bytes, bytearray)):
        return int(raw.decode().strip())
    return int(raw)


def train_sample_transform(sample):
    """Applied to each (image_np, label_raw) tuple from WebDataset."""
    image, label_raw = sample
    image = _TRAIN_TRANSFORM(image=image)["image"]
    return image, _decode_label(label_raw)


def val_sample_transform(sample):
    image, label_raw = sample
    image = _VAL_TRANSFORM(image=image)["image"]
    return image, _decode_label(label_raw)


# ── Model ─────────────────────────────────────────────────────────────────────

class Derm1MModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=True, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ── Training / Validation loops ───────────────────────────────────────────────

def train_one_epoch(loader, model, criterion, optimizer, scheduler,
                    device, epoch):
    model.train()
    tracker  = xm.RateTracker()
    total_loss = correct = total = 0

    for step, (images, labels) in enumerate(loader):
        if step >= STEPS_PER_EPOCH:
            break

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # CRITICAL: syncs gradients across all 8 cores before weight update
        xm.optimizer_step(optimizer)
        tracker.add(BATCH_SIZE)

        _, preds  = torch.max(outputs, 1)
        correct   += (preds == labels).sum().item()
        total     += labels.size(0)
        total_loss += loss.item()

        if step % 50 == 0:
            xm.master_print(
                f"[Epoch {epoch:02d} | Step {step:04d}] "
                f"loss={total_loss/(step+1):.4f}  "
                f"acc={correct/max(total,1):.4f}  "
                f"rate={tracker.rate():.1f} img/s"
            )

    scheduler.step()
    return total_loss / max(STEPS_PER_EPOCH, 1), correct / max(total, 1)


@torch.no_grad()
def validate(loader, model, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    for step, (images, labels) in enumerate(loader):
        if step >= VAL_STEPS_PER_EPOCH:
            break
        outputs    = model(images)
        loss       = criterion(outputs, labels)
        _, preds   = torch.max(outputs, 1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        total_loss += loss.item()

    # Reduce accuracy across all TPU cores so every core reports the same value
    acc_tensor = torch.tensor(correct / max(total, 1), device=device)
    acc_tensor = xm.all_reduce(xm.REDUCE_SUM, acc_tensor) / xm.xrt_world_size()

    return total_loss / max(VAL_STEPS_PER_EPOCH, 1), acc_tensor.item()


# ── Multiprocessing entry point (runs once per TPU core) ─────────────────────

def _mp_fn(index, flags):
    device = xm.xla_device()

    # Load class mapping (all cores read it; identical content)
    class_mapping, num_classes = load_class_mapping(CLASS_MAPPING_PATH)
    xm.master_print(f"Classes: {num_classes}")

    # Save class mapping immediately at startup — don't risk losing it to a timeout
    mapping_out = os.path.join(CHECKPOINT_DIR, "derm1m_class_mapping.json")
    if xm.is_master_ordinal():
        with open(mapping_out, "w") as f:
            json.dump(class_mapping, f, indent=2)
    xm.master_print(f"Class mapping saved → {mapping_out}")

    # ── Model, loss, optimiser, scheduler ─────────────────────────────────────
    model     = Derm1MModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Shard-level train / val split ──────────────────────────────────────────
    # Split the shard list deterministically: last VAL_SHARD_FRACTION = val.
    all_shards  = list(SHARD_URLS)
    split_idx   = max(1, int(len(all_shards) * (1.0 - VAL_SHARD_FRACTION)))
    train_shards = all_shards[:split_idx]
    val_shards   = all_shards[split_idx:]
    xm.master_print(
        f"Shards — train: {len(train_shards)}  |  val: {len(val_shards)}"
    )

    # ── WebDataset pipelines ──────────────────────────────────────────────────
    def make_loader(shards, transform_fn, shuffle_buf=1000, resample=True):
        dataset = (
            wds.WebDataset(shards, resampled=resample,
                           nodesplitter=wds.split_by_node)
            .shuffle(shuffle_buf)
            .decode("rgb8")           # decodes jpg → uint8 numpy (H,W,3)
            .to_tuple("jpg", "cls")   # matches keys written by tar_conversion.py
            .map(transform_fn)
            .batched(BATCH_SIZE, partial=False)
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=2
        )

    train_loader_raw = make_loader(train_shards, train_sample_transform,
                                   shuffle_buf=2000, resample=True)
    val_loader_raw   = make_loader(val_shards,   val_sample_transform,
                                   shuffle_buf=200,  resample=False)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        xm.master_print(f"\n{'='*56}\nEpoch {epoch}/{EPOCHS}\n{'='*56}")

        # Wrap with ParallelLoader to pre-fetch data to the TPU device
        train_pl = pl.ParallelLoader(train_loader_raw, [device])
        val_pl   = pl.ParallelLoader(val_loader_raw,   [device])

        train_loss, train_acc = train_one_epoch(
            train_pl.per_device_loader(device),
            model, criterion, optimizer, scheduler, device, epoch,
        )
        xm.master_print(f"Train — loss: {train_loss:.4f}  acc: {train_acc:.4f}")

        val_loss, val_acc = validate(
            val_pl.per_device_loader(device),
            model, criterion, device,
        )
        xm.master_print(f"Val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Checkpoint (master core only to avoid file corruption)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR,
                                     f"derm1m_best_ep{epoch:02d}.pth")
            xm.save(model.state_dict(), ckpt_path)
            xm.master_print(f"✓ Saved best checkpoint → {ckpt_path}")
        else:
            patience_ctr += 1
            xm.master_print(f"No improvement ({patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                xm.master_print("Early stopping.")
                break

    # Save final model weights
    xm.save(model.state_dict(),
            os.path.join(CHECKPOINT_DIR, "derm1m_final.pth"))

    xm.master_print(f"\nTraining complete! Best val acc: {best_val_acc:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FLAGS = {}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method="spawn")
