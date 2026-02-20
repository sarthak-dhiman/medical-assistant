"""
train_derm1m.py  —  Derm1M Skin Disease Classifier
====================================================

Trains an EfficientNet-B4 classifier on the sorted Derm1M dataset.

Expected dataset layout (produced by scripts/sort_derm1m.py):
    Dataset/Derm1M/sorted_by_class/
        train/
            acne vulgaris/
            psoriasis/
            eczema/
            ...
        val/
            acne vulgaris/
            ...

Classes are auto-discovered from the train/ sub-directories, so no
manual CLASS_NAMES list is needed. The model output layer is sized
to however many classes are found on disk.

Outputs (saved to saved_models/):
    derm1m_classifier.pth       — best model weights
    derm1m_class_mapping.json   — {index: class_name}
    derm1m_metadata.json        — training run summary
"""

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import cv2
import numpy as np
import timm
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE        = 380
BATCH_SIZE      = 8          # Lower if you hit OOM
ACCUM_STEPS     = 4          # Effective batch = BATCH_SIZE * ACCUM_STEPS
EPOCHS          = 40
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
PATIENCE        = 8          # Early-stopping patience (epochs)
NUM_WORKERS     = 2
LABEL_SMOOTHING = 0.1

# Minimum images a class must have in train/ to be included in training.
# Classes below this threshold are skipped (avoids single-image classes).
MIN_IMAGES_PER_CLASS = 5

DATASET_ROOT = Path("Dataset/Derm1M/sorted_by_class")
OUTPUT_DIR   = Path("saved_models")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AMP scaler for mixed-precision training
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
torch.backends.cudnn.benchmark = True
# ─────────────────────────────────────────────────────────────────────────────


# ── Dataset ──────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class Derm1MDataset(Dataset):
    """ImageFolder-style dataset that reads class names from directory names."""

    def __init__(self, root_dir: Path, class_names: list[str], transform=None):
        self.root_dir    = root_dir
        self.class_names = class_names
        self.transform   = transform
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        self.image_paths: list[Path] = []
        self.labels: list[int]       = []

        for class_name in class_names:
            class_dir = root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTS:
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        if not self.image_paths:
            raise ValueError(f"No images found in: {root_dir}")

        split_name = root_dir.name.capitalize()
        print(f"  {split_name}: {len(self.image_paths):,} images across "
              f"{len(class_names)} classes")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            # Return a blank image rather than crashing on a corrupt file
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(is_train: bool):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if is_train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(
                num_holes_range=(4, 8),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                p=0.3,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


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


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, accum_steps):
    model.train()
    running_loss = correct = total = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        _, preds = torch.max(outputs, 1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)
        pbar.set_postfix(loss=f"{running_loss/(batch_idx+1):.4f}",
                         acc=f"{correct/total:.4f}")

    return running_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0

    for images, labels in tqdm(loader, desc="Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)

    return running_loss / len(loader), correct / total


# ── Class discovery ───────────────────────────────────────────────────────────

def discover_classes(train_dir: Path, min_images: int) -> list[str]:
    """
    Walk train_dir and return sorted class names that have >= min_images.
    Skips __unclassified__ and __invalid__ folders.
    """
    SKIP = {"__unclassified__", "__invalid__"}
    classes = []
    for d in sorted(train_dir.iterdir()):
        if not d.is_dir() or d.name in SKIP:
            continue
        count = sum(
            1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )
        if count >= min_images:
            classes.append(d.name)
        else:
            print(f"  [skip] '{d.name}' — only {count} image(s) "
                  f"(< {min_images} threshold)")
    return classes


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  Derm1M Skin Disease Classifier — Training")
    print("=" * 64)
    print(f"  Device    : {DEVICE}")
    print(f"  Backbone  : EfficientNet-B4")

    train_dir = DATASET_ROOT / "train"
    val_dir   = DATASET_ROOT / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Train directory not found: {train_dir}\n"
            "Run scripts/sort_derm1m.py first to build the dataset structure."
        )

    # ── Discover classes ──────────────────────────────────────────────────────
    print(f"\n  Discovering classes (min {MIN_IMAGES_PER_CLASS} images each) …")
    class_names = discover_classes(train_dir, MIN_IMAGES_PER_CLASS)
    num_classes = len(class_names)
    print(f"  → {num_classes} classes selected.\n")

    if num_classes < 2:
        raise ValueError("Need at least 2 classes to train. "
                         "Run sort_derm1m.py first.")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    print("  Loading datasets …")
    train_dataset = Derm1MDataset(train_dir, class_names, get_transforms(True))
    val_dataset   = Derm1MDataset(val_dir,   class_names, get_transforms(False)) \
                    if val_dir.exists() else None

    # Weighted sampler to handle extreme class imbalance
    label_counts   = Counter(train_dataset.labels)
    class_weights  = {c: 1.0 / max(n, 1) for c, n in label_counts.items()}
    sample_weights = [class_weights[l] for l in train_dataset.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    ) if val_dataset else None

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = Derm1MModel(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Classes    : {num_classes}")
    print(f"  Train imgs : {len(train_dataset):,}")
    if val_dataset:
        print(f"  Val imgs   : {len(val_dataset):,}")
    else:
        print("  Val imgs   : (no val/ directory found — skipping validation)")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc      = 0.0
    best_epoch        = 0
    patience_counter  = 0

    print("\n" + "=" * 64)
    for epoch in range(EPOCHS):
        print(f"\n  Epoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, ACCUM_STEPS
        )
        print(f"  Train — loss: {train_loss:.4f}  acc: {train_acc:.4f}")

        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
            print(f"  Val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}")
            monitor = val_acc
        else:
            monitor = train_acc

        scheduler.step()

        if monitor > best_val_acc:
            best_val_acc     = monitor
            best_epoch       = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(),
                       OUTPUT_DIR / "derm1m_classifier.pth")
            print(f"  ✓ Saved best model  (acc={monitor:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1}.")
            break

    # ── Save artefacts ────────────────────────────────────────────────────────
    class_mapping = {str(i): name for i, name in enumerate(class_names)}

    with open(OUTPUT_DIR / "derm1m_class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)

    metadata = {
        "model"         : "Derm1M Skin Disease Classifier",
        "architecture"  : "EfficientNet-B4",
        "num_classes"   : num_classes,
        "classes"       : class_mapping,
        "image_size"    : IMG_SIZE,
        "best_val_acc"  : round(float(best_val_acc), 4),
        "best_epoch"    : best_epoch,
        "epochs_trained": epoch + 1,
    }
    with open(OUTPUT_DIR / "derm1m_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 64)
    print(f"  Training Complete!")
    print(f"  Best accuracy : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"  Model saved   : {OUTPUT_DIR / 'derm1m_classifier.pth'}")
    print(f"  Mapping saved : {OUTPUT_DIR / 'derm1m_class_mapping.json'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
