import os
import cv2
import time
import numpy as np
import yaml
import copy
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm

# --- Configuration ---
IMG_SIZE = (380, 380) # EfficientNetB4 Resolution
SCLERA_SIZE = (64, 64)
BATCH_SIZE = 8 # Fit on 3050 GPU
EPOCHS = 25
LR = 1e-4

# Paths
DATASET_ROOT = Path(r"D:\Disease Prediction\Dataset\combined_sclera")
IMG_DIR = DATASET_ROOT / "JPEGImages"
MASK_DIR = DATASET_ROOT / "SegmentationClass"
SAVE_DIR = Path("saved_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = SAVE_DIR / "jaundice_with_sclera_torch.pth"

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Utils ---
def get_label(filename):
    """
    Infers label from filename.
    Jaundice files usually have 'jaundice' in the name.
    """
    name = filename.lower()
    if 'jaundice' in name:
        return 1.0
    return 0.0

# --- Dataset ---
class JaundiceScleraDataset(Dataset):
    def __init__(self, image_files, mask_dir, augment=True):
        self.image_files = image_files
        self.mask_dir = mask_dir
        self.augment = augment
        
    def __len__(self):
        return len(self.image_files)
    
    def apply_augmentation(self, img, sclera):
        """
        Simple photometric augmentations.
        Using cv2 for speed.
        """
        # Flip H
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            # Sclera is cropped, so flipping it might make it look like the other eye
            # but for texture analysis it's fine.
            sclera = cv2.flip(sclera, 1)

        # Brightness/Contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            sclera = cv2.convertScaleAbs(sclera, alpha=alpha, beta=beta)
            
        return img, sclera

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # 1. Load Image
        img = cv2.imread(str(img_path))
        if img is None:
            # Fallback for corrupted image
            return self.__getitem__((idx + 1) % len(self))
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Label
        label = get_label(img_path.name)
        
        # 3. Load Mask & Crop Sclera
        stem = img_path.stem
        mask_path = self.mask_dir / f"{stem}.png"
        
        sclera_crop = np.zeros((*SCLERA_SIZE, 3), dtype=np.uint8)
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is not None:
                # Handle resizing
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                # Binary Mask
                if len(mask.shape) == 3: mask = mask[:,:,0]
                binary_mask = (mask > 0).astype(np.uint8)
                
                # Bounding Box
                points = cv2.findNonZero(binary_mask)
                if points is not None:
                    x, y, w, h = cv2.boundingRect(points)
                    masked_img = cv2.bitwise_and(img, img, mask=binary_mask)
                    crop = masked_img[y:y+h, x:x+w]
                    
                    if crop.size > 0:
                        sclera_crop = cv2.resize(crop, SCLERA_SIZE)

        # 4. Augmentation
        if self.augment:
            img, sclera_crop = self.apply_augmentation(img, sclera_crop)
            
        # 5. Preprocess (Normalization & Tensors)
        # Skin Branch: Resize -> Normalize 0-1 -> CHW
        img_resized = cv2.resize(img, IMG_SIZE)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        
        # Sclera Branch: Normalize 0-1 -> CHW
        sclera_tensor = torch.from_numpy(sclera_crop).permute(2, 0, 1).float() / 255.0
        
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return {
            'skin': img_tensor,
            'sclera': sclera_tensor,
            'label': label_tensor
        }

# --- Model ---
class JaundiceModel(nn.Module):
    def __init__(self):
        super(JaundiceModel, self).__init__()
        
        # 1. Skin Branch: EfficientNetB4
        # 'tf_efficientnet_b4' matches the Keras weights closely
        self.skin_backbone = timm.create_model('tf_efficientnet_b4', pretrained=True, num_classes=0)
        self.skin_dim = self.skin_backbone.num_features # Usually 1792 for B4
        
        # 2. Sclera Branch: Custom CNN
        self.sclera_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling -> 64
        )
        self.sclera_dim = 64
        
        # 3. Fusion
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.skin_dim + self.sclera_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Logits
        )
        
    def forward(self, skin, sclera):
        # Skin Feature
        x1 = self.skin_backbone(skin) # (B, 1792)
        
        # Sclera Feature
        x2 = self.sclera_branch(sclera) # (B, 64, 1, 1)
        x2 = x2.view(x2.size(0), -1)    # (B, 64)
        
        # Fusion
        concat = torch.cat((x1, x2), dim=1)
        logits = self.classifier(concat)
        return logits

# --- Training Loop ---
def train_model():
    # 1. Prepare Data
    print("Gathering images...")
    all_images = list(IMG_DIR.glob("*"))
    valid_exts = ['.jpg', '.jpeg', '.png']
    all_images = [f for f in all_images if f.suffix.lower() in valid_exts]
    
    if not all_images:
        print("No images found!")
        return

    labels = [get_label(f.name) for f in all_images]
    print(f"Found {len(all_images)} images. Jaundice: {sum(labels)}, Normal: {len(labels)-sum(labels)}")
    
    train_files, val_files = train_test_split(all_images, test_size=0.2, stratify=labels, random_state=42)
    
    train_dataset = JaundiceScleraDataset(train_files, MASK_DIR, augment=True)
    val_dataset = JaundiceScleraDataset(val_files, MASK_DIR, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # workers=0 for Windows safety
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. Init Model
    print("Initializing EfficientNetB4 + Sclera CNN...")
    model = JaundiceModel().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 3. Validation Helper
    def validate(loader):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                skin = batch['skin'].to(device)
                sclera = batch['sclera'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(skin, sclera)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * skin.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return running_loss / total, correct / total

    # 4. Train Loop
    best_acc = 0.0
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            skin = batch['skin'].to(device)
            sclera = batch['sclera'].to(device)
            labels = batch['label'].to(device)
            
            # Forward
            outputs = model(skin, sclera)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats
            running_loss += loss.item() * skin.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        
        train_loss = running_loss / train_total
        train_acc = train_correct / train_total
        
        val_loss, val_acc = validate(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"Saving new best model... ({best_acc:.4f})")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print("Training Complete.")

if __name__ == '__main__':
    train_model()
