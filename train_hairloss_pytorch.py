"""
Training Script for Hairloss Detection Model
Binary Classification: Bald vs Not Bald
Architecture: EfficientNet-B3
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
IMG_SIZE = 224  # Smaller for scalp images
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
PATIENCE = 10

# Paths
DATASET_ROOT = Path("Dataset/Hairloss")
OUTPUT_DIR = Path("saved_models")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"💇 Hairloss Detection Training")
print(f"Device: {DEVICE}")


class HairlossDataset(Dataset):
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
            A.Rotate(limit=15, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
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
    bald_dir = DATASET_ROOT / "bald"
    notbald_dir = DATASET_ROOT / "notbald"
    
    bald_images = list(bald_dir.glob("*.jpg")) + list(bald_dir.glob("*.png"))
    notbald_images = list(notbald_dir.glob("*.jpg")) + list(notbald_dir.glob("*.png"))
    
    print(f"📊 Dataset Statistics:")
    print(f"  Bald images: {len(bald_images)}")
    print(f"  Not Bald images: {len(notbald_images)}")
    
    # Labels: 0 = not bald, 1 = bald
    all_images = notbald_images + bald_images
    all_labels = [0] * len(notbald_images) + [1] * len(bald_images)
    
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.33, stratify=temp_labels, random_state=42
    )
    
    print(f"  Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")
    
    return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class HairlossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
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
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
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
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / len(loader), correct / total


def main():
    (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels) = load_dataset()
    
    train_dataset = HairlossDataset(train_imgs, train_labels, get_transforms(True))
    val_dataset = HairlossDataset(val_imgs, val_labels, get_transforms(False))
    test_dataset = HairlossDataset(test_imgs, test_labels, get_transforms(False))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = HairlossModel().to(DEVICE)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)  # Handle imbalance
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
            torch.save(model.state_dict(), OUTPUT_DIR / "hairloss_pytorch.pth")
            print(f"✅ Saved best model (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping triggered")
            break
    
    print("\n🧪 Testing...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "hairloss_pytorch.pth"))
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    metadata = {
        "model": "Hairloss Detection",
        "architecture": "EfficientNet-B3",
        "classes": {"0": "Not Bald", "1": "Bald"},
        "image_size": IMG_SIZE,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "epochs_trained": epoch + 1
    }
    
    with open(OUTPUT_DIR / "hairloss_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n🎉 Training Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
