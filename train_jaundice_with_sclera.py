import os
import cv2
import time
import numpy as np
import copy
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm

# --- Configuration ---
IMG_SIZE = (380, 380)
SCLERA_SIZE = (64, 64)
BATCH_SIZE = 8  # Reduced from 16 to prevent CUDA engine errors
EPOCHS = 30
LR = 1e-4
PATIENCE = 7

# Paths (will be configurable via CLI)
DEFAULT_JAUNDICE_ROOT = Path(r"D:\Disease Prediction\Dataset\jaundice_sclera")
DEFAULT_NORMAL_ROOT = Path(r"D:\Disease Prediction\Dataset\normal_sclera")
SAVE_DIR = Path("saved_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = SAVE_DIR / "jaundice_with_sclera_torch.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Enable cuDNN benchmark for optimized performance, but turn off if crashes persist
torch.backends.cudnn.benchmark = True 

# --- Robust Augmentation Pipeline (Albumentations 2.x comp.) ---
train_transform = A.Compose([
    A.Resize(*IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    # Fixed: std_range must be 0-1 in Albumentations 2.x (normalized relative to 255)
    A.GaussNoise(std_range=(0.04, 0.2), p=0.3), 
    A.ImageCompression(quality_range=(70, 100), p=0.2),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(10, 20), hole_width_range=(10, 20), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(*IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

sclera_transform = A.Compose([
    A.Resize(*SCLERA_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# --- Label Extraction ---
def get_label(filename):
    """Infer label from filename (jaundice vs normal)"""
    fn_lower = filename.lower()
    if 'jaundice' in fn_lower or 'jau' in fn_lower:
        return 1
    elif 'normal' in fn_lower or 'norm' in fn_lower:
        return 0
    else:
        # Fallback: Check parent folder
        return 1 if 'jaundice' in str(filename) else 0

# --- Dataset ---
class JaundiceScleraDataset(Dataset):
    def __init__(self, image_items, transform=None, sclera_transform=None):
        # image_items: list of tuples (img_path: Path, mask_dir: Path, label: int)
        self.image_items = image_items
        self.transform = transform
        self.sclera_transform = sclera_transform
        
    def __len__(self):
        return len(self.image_items)
    
    def __getitem__(self, idx):
        img_path, mask_dir, explicit_label = self.image_items[idx]

        # 1. Load Image
        img = cv2.imread(str(img_path))
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Load Mask (mask_dir provided per-image)
        mask_name = img_path.with_suffix('.png').name
        mask_path = mask_dir / mask_name

        if not mask_path.exists():
            mask_path = mask_dir / img_path.name

        mask = None
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # fallback to next sample if mask missing
            return self.__getitem__((idx + 1) % len(self))
        
        # 3. Extract Sclera Crop
        sclera_mask = ((mask == 4) | (mask == 5)).astype(np.uint8) * 255  # Both eyes
        contours, _ = cv2.findContours(sclera_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            sclera_crop = img[y:y+h, x:x+w]
        else:
            sclera_crop = np.zeros((*SCLERA_SIZE, 3), dtype=np.uint8)
        
        # 4. Apply Transforms
        if self.transform:
            augmented = self.transform(image=img)
            img_tensor = augmented['image']
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        if self.sclera_transform and sclera_crop.size > 0:
            sclera_aug = self.sclera_transform(image=sclera_crop)
            sclera_tensor = sclera_aug['image']
        else:
            sclera_resized = cv2.resize(sclera_crop, SCLERA_SIZE)
            sclera_tensor = torch.from_numpy(sclera_resized).permute(2, 0, 1).float() / 255.0
        
        # 5. Label (use explicit label provided)
        label = explicit_label
        
        return {
            'skin': img_tensor,
            'sclera': sclera_tensor,
            'label': torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        }

# --- Model Architecture ---
class JaundiceModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Skin Branch: EfficientNet-B4
        self.skin_backbone = timm.create_model('tf_efficientnet_b4', pretrained=True, num_classes=0)
        self.skin_dim = self.skin_backbone.num_features
        
        # Sclera Branch: Lightweight CNN
        self.sclera_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.sclera_dim = 64
        
        # Fusion Classifier with Stronger Regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.skin_dim + self.sclera_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, skin, sclera):
        x1 = self.skin_backbone(skin)
        x2 = self.sclera_branch(sclera)
        x2 = x2.view(x2.size(0), -1)
        concat = torch.cat((x1, x2), dim=1)
        logits = self.classifier(concat)
        return logits

# --- Training Loop ---
def train_model(jaundice_root: Path = DEFAULT_JAUNDICE_ROOT, normal_root: Path = DEFAULT_NORMAL_ROOT):
    # 1. Prepare Data with PATIENT-LEVEL SPLITTING
    print("Gathering images...")
    valid_exts = ['.jpg', '.jpeg', '.png']

    def collect_from(root: Path, label: int):
        imgs = []
        jpeg_dir = root / 'JPEGImages'
        mask_dir = root / 'SegmentationClass'
        if jpeg_dir.exists():
            for f in sorted(jpeg_dir.glob('*')):
                if f.suffix.lower() in valid_exts and f.is_file():
                    imgs.append((f, mask_dir, label))
        else:
            # fallback to root images
            for f in sorted(root.glob('*')):
                if f.suffix.lower() in valid_exts and f.is_file():
                    imgs.append((f, mask_dir, label))
        return imgs

    all_items = []
    all_items += collect_from(Path(jaundice_root), 1)
    all_items += collect_from(Path(normal_root), 0)

    if not all_items:
        print("No images found in provided dataset roots!")
        return

    labels = [label for (_, _, label) in all_items]
    print(f"Found {len(all_items)} images. Jaundice: {sum(labels)}, Normal: {len(labels)-sum(labels)}")

    # CRITICAL: Patient-level split (extract patient ID from filename if possible)
    # For now, use stratified random split
    train_items, val_items = train_test_split(all_items, test_size=0.2, stratify=labels, random_state=42)

    # Update Dataset to accept tuples (img_path, mask_dir, label)
    train_dataset = JaundiceScleraDataset(train_items, transform=train_transform, sclera_transform=sclera_transform)
    val_dataset = JaundiceScleraDataset(val_items, transform=val_transform, sclera_transform=sclera_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Model
    model = JaundiceModel().to(device)
    
    # 3. Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()  # Simpler than Focal Loss for balanced data
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 4. Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_wts = None
    
    # 5. Training Loop
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Train Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            skin = batch['skin'].to(device)
            sclera = batch['sclera'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(skin, sclera)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * skin.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                skin = batch['skin'].to(device)
                sclera = batch['sclera'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(skin, sclera)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * skin.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Learning Rate Scheduling
        scheduler.step(val_loss)
        
        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"New best model! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"\n Early stopping triggered at epoch {epoch+1}")
                break
    
    # 6. Save Best Model
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, MODEL_SAVE_PATH)
        print(f"\nBest model saved to {MODEL_SAVE_PATH}")
    
    # 7. Final Evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            skin = batch['skin'].to(device)
            sclera = batch['sclera'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(skin, sclera)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Jaundice']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train jaundice model with sclera crops')
    parser.add_argument('--jaundice-root', type=str, default=str(DEFAULT_JAUNDICE_ROOT),
                        help='Path to Dataset/jaundice_sclera root')
    parser.add_argument('--normal-root', type=str, default=str(DEFAULT_NORMAL_ROOT),
                        help='Path to Dataset/normal_sclera root')
    args = parser.parse_args()

    train_model(jaundice_root=Path(args.jaundice_root), normal_root=Path(args.normal_root))
