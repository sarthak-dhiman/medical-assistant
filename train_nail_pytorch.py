"""
Training Script for Nail Disease Detection Model
8-Class Classification: Multiple nail pathologies
Architecture: EfficientNet-B4
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
import timm
from tqdm import tqdm
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
IMG_SIZE = 380
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
PATIENCE = 10

# Paths
DATASET_ROOT = Path("Dataset/nail/nail_dataset")
OUTPUT_DIR = Path("saved_models")
OUTPUT_DIR.mkdir(exist_ok=True)

# Class mapping
CLASS_NAMES = [
    "Acral_Lentiginous_Melanoma",
    "Blue_Finger",
    "Clubbing",
    "Healthy_Nail",
    "Nail_Psoriasis",
    "Onychogryphosis",
    "Onychomycosis",
    "Pitting"
]

print(f"💅 Nail Disease Detection Training")
print(f"Device: {DEVICE}")
print(f"Classes: {len(CLASS_NAMES)}")


class NailDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load all images from class folders
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                self.image_paths.extend(images)
                self.labels.extend([class_idx] * len(images))
        
        print(f"  {split.capitalize()}: {len(self.image_paths)} images")
        
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
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(max_holes=6, max_height=24, max_width=24, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class NailDiseaseModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
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
    print("📊 Loading dataset...")
    train_dataset = NailDataset(DATASET_ROOT, split='train', transform=get_transforms(True))
    test_dataset = NailDataset(DATASET_ROOT, split='test', transform=get_transforms(False))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = NailDiseaseModel(num_classes=len(CLASS_NAMES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\n📍 Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, test_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "nail_disease_pytorch.pth")
            print(f"✅ Saved best model (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping triggered")
            break
    
    print("\n🧪 Final Testing...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "nail_disease_pytorch.pth"))
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Save class mapping
    class_mapping = {str(i): name.replace("_", " ") for i, name in enumerate(CLASS_NAMES)}
    with open(OUTPUT_DIR / "nail_disease_mapping.json", 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    metadata = {
        "model": "Nail Disease Detection",
        "architecture": "EfficientNet-B4",
        "num_classes": len(CLASS_NAMES),
        "classes": class_mapping,
        "image_size": IMG_SIZE,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "epochs_trained": epoch + 1
    }
    
    with open(OUTPUT_DIR / "nail_disease_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n🎉 Training Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
