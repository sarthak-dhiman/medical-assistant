# Google Colab TPU Training Script for Nail Disease Model
# Hardware: TPU v5e-1 (or any TPU node)
# Dataset: Non-zipped folder (via Google Drive Mount)

# ==========================================
# 1. SETUP & IMPORTS
# ==========================================
# Copy-paste this block into a Colab cell to install dependencies
# !pip install -U torch-xla cloud-tpu-client https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl
# !pip install timm albumentations opencv-python tqdm

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
import timm
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

# TPU Specific Imports
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    print("✅ TPU detected and libraries imported.")
except ImportError:
    print("❌ TPU libraries not found. Using CPU/GPU if available.")
    # Fallback for CPU/GPU debugging
    class XMDummy:
        def xla_device(self): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        def optimizer_step(self, optimizer): optimizer.step()
        def save(self, obj, path): torch.save(obj, path)
        def master_print(self, msg): print(msg)
        def xrt_world_size(self): return 1
        def get_ordinal(self): return 0
    xm = XMDummy()

# ==========================================
# 2. CONFIGURATION
# ==========================================
# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

# UPDATE THIS PATH to where your unzipped 'nail_dataset' folder is located on Drive
DATASET_ROOT = Path("/content/drive/MyDrive/Disease Prediction/Dataset/nail/nail_dataset")
OUTPUT_DIR = Path("/content/drive/MyDrive/Disease Prediction/saved_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Class Mapping
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

IMG_SIZE = 380
BATCH_SIZE = 16  # Per core batch size (BS=128 effective on 8 cores)
EPOCHS = 30
LEARNING_RATE = 1e-4

# ==========================================
# 3. DATASET
# ==========================================
class NailDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.png")) + \
                         list(class_dir.glob("*.jpeg"))
                self.image_paths.extend(images)
                self.labels.extend([class_idx] * len(images))
        
        xm.master_print(f"  {split.capitalize()}: {len(self.image_paths)} images found.")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            # Handle corrupt images gracefully
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), label
            
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

# ==========================================
# 4. MODEL
# ==========================================
class NailDiseaseModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # EfficientNet-B4 is a good balance for this task
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

# ==========================================
# 5. TRAINING LOOP (TPU Optimized)
# ==========================================
def train_model(index, flags):
    # Set device
    device = xm.xla_device()
    
    # Create datasets
    train_dataset = NailDataset(DATASET_ROOT, split='train', transform=get_transforms(True))
    val_dataset = NailDataset(DATASET_ROOT, split='test', transform=get_transforms(False))
    
    # Distributed Sampler for multi-core TPU
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    
    # DataLoaders
    # Note: drop_last=True is important for TPUs to keep shapes consistent
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=4,
        drop_last=False
    )
    
    # Model Setup
    model = NailDiseaseModel(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Scale learning rate by world size
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE * xm.xrt_world_size(), weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training Phase
        model.train()
        # Use ParallelLoader for TPU data prefetching
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in para_train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            xm.optimizer_step(optimizer) # TPU optimize step
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        para_val_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in para_val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        # Log results (only on master process)
        xm.master_print(f"Epoch {epoch+1}/{EPOCHS} - "
                        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                        f"Time: {time.time() - start_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            xm.save(model.state_dict(), OUTPUT_DIR / "nail_disease_tpu.pth")
            xm.master_print(f"✅ Saved best model (Val Acc: {val_acc:.4f})")

# ==========================================
# 6. EXECUTION
# ==========================================
if __name__ == '__main__':
    # Launch multi-processing on TPU
    # Use 'start_method'='fork' for Colab environments usually
    try:
        xmp.spawn(train_model, args=({},), nprocs=8, start_method='fork')
    except Exception as e:
        print(f"TPU Spawn failed: {e}")
