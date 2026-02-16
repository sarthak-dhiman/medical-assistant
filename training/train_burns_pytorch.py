"""
Training Script for Burns Detection Model
Binary Classification: Burn vs Healthy Skin
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
BATCH_SIZE = 8   # Further reduced to avoid cuDNN execution failure
ACCUM_STEPS = 4  # Gradient accumulation (effective batch size = 32)
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0  # Reduced workers to minimize overhead
PATIENCE = 10  # Early stopping

# Optimize cuDNN
torch.backends.cudnn.benchmark = True

# Paths
DATASET_ROOT = Path("Dataset/Burns_Skin")
OUTPUT_DIR = Path("saved_models")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Burns Detection Training")
print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_ROOT}")


class BurnsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
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
            A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
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
    """Load burns dataset with class balancing"""
    burn_dir = DATASET_ROOT / "burns"
    healthy_dir = DATASET_ROOT / "healthy" / "healthy"
    
    # Collect all images
    burn_images = list(burn_dir.glob("*.jpg")) + list(burn_dir.glob("*.png"))
    healthy_images = list(healthy_dir.glob("*.jpg")) + list(healthy_dir.glob("*.png"))
    
    print(f"Dataset Statistics:")
    print(f"Burn images: {len(burn_images)}")
    print(f"Healthy images: {len(healthy_images)}")
    
    # Create labels (0 = healthy, 1 = burn)
    all_images = healthy_images + burn_images
    all_labels = [0] * len(healthy_images) + [1] * len(burn_images)
    
    # Split: 70% train, 20% val, 10% test
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.33, stratify=temp_labels, random_state=42
    )
    
    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")
    
    return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels)


class BurnsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
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
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / accum_steps  # Scale loss for gradient accumulation
        loss.backward()
        
        # Update weights every accum_steps batches
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accum_steps
        preds = (torch.sigmoid(outputs) > 0.5).float()
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
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / len(loader), correct / total


def main():
    # Load dataset
    (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels) = load_dataset()
    
    # Calculate class weights for imbalanced data
    num_healthy = sum(1 for l in train_labels if l == 0)
    num_burn = sum(1 for l in train_labels if l == 1)
    pos_weight = torch.tensor([num_healthy / num_burn]).to(DEVICE)
    print(f"Class Weight (Burn): {pos_weight.item():.2f}")
    
    # Create datasets
    train_dataset = BurnsDataset(train_imgs, train_labels, get_transforms(True))
    val_dataset = BurnsDataset(val_imgs, val_labels, get_transforms(False))
    test_dataset = BurnsDataset(test_imgs, test_labels, get_transforms(False))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Create model
    model = BurnsModel().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, ACCUM_STEPS)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "burns_pytorch.pth")
            print(f"Saved best model (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Test evaluation
    print("\nTesting on held-out test set...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "burns_pytorch.pth"))
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Save metadata
    metadata = {
        "model": "Burns Detection",
        "architecture": "EfficientNet-B4",
        "classes": {"0": "Healthy", "1": "Burn"},
        "image_size": IMG_SIZE,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "epochs_trained": epoch + 1
    }
    
    with open(OUTPUT_DIR / "burns_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTraining Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
