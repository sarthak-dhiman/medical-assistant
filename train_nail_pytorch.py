"""
Training Script for Nail Disease Detection Model
8-Class Classification: Multiple nail pathologies
Architecture: EfficientNet-B4
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
BATCH_SIZE = 4   # Reduced further to avoid OOM
ACCUM_STEPS = 8  # Increased accumulation (effective batch size = 32)
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0  # Reduced workers to minimize overhead
PATIENCE = 10

# Initialize AMP Scaler
scaler = torch.cuda.amp.GradScaler()

# Optimize cuDNN
torch.backends.cudnn.benchmark = True

# Paths
DATASET_ROOT = Path("Dataset/nail")
OUTPUT_DIR = Path("saved_models")
OUTPUT_DIR.mkdir(exist_ok=True)

# Class mapping
CLASS_NAMES = [
    "Acral_Lentiginous_Melanoma", "Alopecia_Areata", "Beaus_Lines", "Blue_Finger",
    "Clubbing", "Dariers_Disease", "Eczema", "Half_And_Half_Nail",
    "Healthy_Nail", "Koilonychia", "Leukonychia", "Lunula_Red",
    "Muehrckes_Lines", "Nail_Pale", "Nail_Psoriasis", "Onychogryphosis",
    "Onycholysis", "Onychomycosis", "Pitting", "Splinter_Hemorrhage",
    "Terrys_Nails", "Nail_White", "Nails_Yellow"
]

print(f"Nail Disease Detection Training")
print(f"Device: {DEVICE}")
print(f"Classes: {len(CLASS_NAMES)}")


class NailDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")
            
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load all images from class folders
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
                self.image_paths.extend(images)
                self.labels.extend([class_idx] * len(images))
            else:
                print(f"  Warning: Class directory {class_name} not found in {split}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.root_dir}. Please check your dataset structure.")
            
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
            A.CoarseDropout(num_holes_range=(4, 6), hole_height_range=(12, 24), hole_width_range=(12, 24), p=0.3),
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


def train_epoch(model, loader, criterion, optimizer, device, accum_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Use AMP autocast
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accum_steps  # Scale loss
            
        # Scale loss and backward
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accum_steps
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}', 'acc': f'{correct/total:.4f}'})
    
    return running_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / len(loader), correct / total


def main():
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("Loading dataset...")
    train_dataset = NailDataset(DATASET_ROOT, split='train', transform=get_transforms(True))
    test_dataset = NailDataset(DATASET_ROOT, split='test', transform=get_transforms(False))
    
    # Calculate weights for imbalance handling (ensure Healthy and minority classes are sampled)
    labels = torch.tensor(train_dataset.labels)
    class_sample_count = torch.tensor([(labels == t).sum() for t in range(len(CLASS_NAMES))])
    
    # Avoid division by zero if a class is missing (though mapping should find them now)
    class_sample_count = torch.clamp(class_sample_count, min=1)
    
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in labels])
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
    
    # Use sampler for train_loader (shuffle must be False when using a sampler)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = NailDiseaseModel(num_classes=len(CLASS_NAMES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, ACCUM_STEPS)
        val_loss, val_acc = validate(model, test_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "nail_disease_pytorch.pth")
            print(f"Saved best model (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered")
            break
    
    print("\nFinal Testing...")
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
    
    print(f"\n Training Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
