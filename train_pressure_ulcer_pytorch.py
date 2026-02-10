"""
Training Script for Pressure Ulcer Staging Model
4-Class Ordinal Classification: Stage 1, 2, 3, 4
Architecture: EfficientNet-B4
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import timm
from tqdm import tqdm
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
IMG_SIZE = 380
BATCH_SIZE = 16  # Smaller batch for limited dataset
EPOCHS = 60
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
PATIENCE = 15

# Paths
DATASET_ROOT = Path("Dataset/pressure ulcers")
OUTPUT_DIR = Path("saved_models")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"🩹 Pressure Ulcer Staging Training")
print(f"Device: {DEVICE}")


class PressureUlcerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        return img, label


def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),  # Moderate rotation to preserve wound characteristics
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def load_dataset():
    all_images = []
    all_labels = []
    
    for stage in [1, 2, 3, 4]:
        stage_dir = DATASET_ROOT / str(stage)
        images = list(stage_dir.glob("*.jpg")) + list(stage_dir.glob("*.png"))
        all_images.extend(images)
        all_labels.extend([stage - 1] * len(images))  # 0-indexed (0, 1, 2, 3)
        print(f"  Stage {stage}: {len(images)} images")
    
    print(f"📊 Total images: {len(all_images)}")
    
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.33, stratify=temp_labels, random_state=42
    )
    
    print(f"  Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")
    
    return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels)


class PressureUlcerModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    return running_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / len(loader), correct / total


def main():
    (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels) = load_dataset()
    
    train_dataset = PressureUlcerDataset(train_imgs, train_labels, get_transforms(True))
    val_dataset = PressureUlcerDataset(val_imgs, val_labels, get_transforms(False))
    test_dataset = PressureUlcerDataset(test_imgs, test_labels, get_transforms(False))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = PressureUlcerModel(num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for small dataset
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\n📍 Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "pressure_ulcer_pytorch.pth")
            print(f"✅ Saved best model (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping triggered")
            break
    
    print("\n🧪 Testing...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "pressure_ulcer_pytorch.pth"))
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    metadata = {
        "model": "Pressure Ulcer Staging",
        "architecture": "EfficientNet-B4",
        "classes": {
            "0": "Stage 1",
            "1": "Stage 2",
            "2": "Stage 3",
            "3": "Stage 4"
        },
        "image_size": IMG_SIZE,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "epochs_trained": epoch + 1
    }
    
    with open(OUTPUT_DIR / "pressure_ulcer_mapping.json", 'w') as f:
        json.dump(metadata["classes"], f, indent=2)
    
    with open(OUTPUT_DIR / "pressure_ulcer_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n🎉 Training Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
