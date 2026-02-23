"""
derm1m_kaggle_gpu_train.py — Derm1M Skin Disease Classifier (GPU)
=====================================================================

Designed for Kaggle T4 x2 or P100 GPUs using PyTorch + WebDataset.
Automatically detects multiple GPUs and uses DataParallel.

Setup on Kaggle
---------------
1. Upload the .tar shards (from scripts/tar_conversion.py) and class_mapping.json
   to a Kaggle dataset and attach it to your notebook.
2. Ensure SHARD_INPUT_DIR points to that dataset directory.
3. Enable "GPU T4 x2" or "GPU P100" in the Kaggle notebook accelerator settings.
4. Install requirements in the notebook cell before running:
       !pip install timm webdataset albumentations -q
"""

import json
import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import webdataset as wds
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Path to the folder containing your uploaded .tar shards on Kaggle.
SHARD_INPUT_DIR = "/kaggle/input/derm1m-shards"
SHARD_URLS = sorted(glob.glob(f"{SHARD_INPUT_DIR}/derm1m_train_*.tar"))
VAL_SHARD_FRACTION = 0.10
CLASS_MAPPING_PATH = f"{SHARD_INPUT_DIR}/class_mapping.json"
CHECKPOINT_DIR = "/kaggle/working/"

IMG_SIZE        = 380
BATCH_SIZE      = 64          # Effective batch size. T4x2 can handle ~64 total.
EPOCHS          = 40
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
PATIENCE        = 8

# Steps for ~410K total images split 90/10. Adjust if dataset size changes.
STEPS_PER_EPOCH     = 5760    # ~369,000 / 64 ≈ 5760
VAL_STEPS_PER_EPOCH = 640     # ~41,000 / 64 ≈ 640

NUM_WORKERS = 4               # Dataloader workers per split

# ─────────────────────────────────────────────────────────────────────────────

def load_class_mapping(path: str) -> tuple[dict, int]:
    with open(path) as f:
        mapping = json.load(f)
    return mapping, len(mapping)

def _make_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def _make_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

_TRAIN_TRANSFORM = _make_train_transform()
_VAL_TRANSFORM   = _make_val_transform()

def _decode_label(raw) -> int:
    if hasattr(raw, 'decode'):
        return int(raw.decode().strip())
    return int(raw)

def train_sample_transform(sample):
    image, label_raw = sample
    image = _TRAIN_TRANSFORM(image=image)["image"]
    return image, _decode_label(label_raw)

def val_sample_transform(sample):
    image, label_raw = sample
    image = _VAL_TRANSFORM(image=image)["image"]
    return image, _decode_label(label_raw)

class Derm1MModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0)
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


def train_one_epoch(loader, model, criterion, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss = correct = total = 0
    t0 = time.time()

    for step, (images, labels) in enumerate(loader):
        if step >= STEPS_PER_EPOCH:
            break

        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        
        # Gradient clipping for stability with many classes
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

        if step % 50 == 0:
            elapsed = time.time() - t0
            rate = (step + 1) * BATCH_SIZE / elapsed
            print(f"[Epoch {epoch:02d} | Step {step:04d}/{STEPS_PER_EPOCH}] "
                  f"loss={total_loss/(step+1):.4f}  acc={correct/total:.4f}  "
                  f"rate={rate:.1f} img/s")

    scheduler.step()
    return total_loss / STEPS_PER_EPOCH, correct / total


@torch.no_grad()
def validate(loader, model, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    for step, (images, labels) in enumerate(loader):
        if step >= VAL_STEPS_PER_EPOCH:
            break

        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    return total_loss / VAL_STEPS_PER_EPOCH, correct / total


def main():
    if not SHARD_URLS:
        raise ValueError(f"No .tar files found in {SHARD_INPUT_DIR}!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        print(f"GPUs available: {n_gpus} ({torch.cuda.get_device_name(0)})")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    class_mapping, num_classes = load_class_mapping(CLASS_MAPPING_PATH)
    
    # Save class mapping to working dir immediately
    map_out = os.path.join(CHECKPOINT_DIR, "derm1m_class_mapping.json")
    with open(map_out, "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Classes: {num_classes} | Mapping saved → {map_out}")

    model = Derm1MModel(num_classes)
    
    # Use DataParallel if multiple GPUs (e.g., Kaggle T4x2)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()  # For Mixed Precision (AMP)

    # ── Shard splitting ───────────────────────────────────────────────────────
    split_idx = max(1, int(len(SHARD_URLS) * (1.0 - VAL_SHARD_FRACTION)))
    train_shards = SHARD_URLS[:split_idx]
    val_shards = SHARD_URLS[split_idx:]
    print(f"Shards — train: {len(train_shards)} | val: {len(val_shards)}")

    # ── WebDataset Pipelines ──────────────────────────────────────────────────
    def make_loader(shards, transform_fn, shuffle_buf=2000, resample=True):
        dataset = (
            wds.WebDataset(shards, resampled=resample, nodesplitter=wds.split_by_node)
            .shuffle(shuffle_buf)
            .decode("rgb8")
            .to_tuple("jpg", "cls")
            .map(transform_fn)
            .batched(BATCH_SIZE, partial=False)
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=NUM_WORKERS, pin_memory=True
        )

    train_loader = make_loader(train_shards, train_sample_transform, shuffle_buf=3000, resample=True)
    val_loader   = make_loader(val_shards, val_sample_transform, shuffle_buf=0, resample=False)

    best_val_acc = 0.0
    patience_ctr = 0

    print("\nStarting Training...\n" + "="*50)
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            train_loader, model, criterion, optimizer, scheduler, scaler, device, epoch
        )
        print(f"[Result] Train — loss: {train_loss:.4f}  acc: {train_acc:.4f}")

        val_loss, val_acc = validate(val_loader, model, criterion, device)
        print(f"[Result] Val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Best Model Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"derm1m_best_ep{epoch:02d}.pth")
            
            # Save unwrapped model if DataParallel is used
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckpt_path)
            print(f"✓ Saved best checkpoint → {ckpt_path}")
        else:
            patience_ctr += 1
            print(f"No improvement ({patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                print("Early stopping triggered.")
                break
                
        # Periodic Checkpoint (every 5 epochs) against timeouts
        if epoch % 5 == 0:
            periodic_path = os.path.join(CHECKPOINT_DIR, f"derm1m_periodic_ep{epoch:02d}.pth")
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, periodic_path)
            print(f"• Saved periodic checkpoint → {periodic_path}")

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "derm1m_final.pth")
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, final_path)
    print(f"\nTraining complete! Best val acc: {best_val_acc:.4f} | Saved final model.")

if __name__ == "__main__":
    main()
