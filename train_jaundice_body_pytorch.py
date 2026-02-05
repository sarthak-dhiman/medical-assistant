# Jaundice Body Detection: PyTorch Local Training Script
# Architecture: EfficientNet-B4 (Matches Eye Model)
# Hardware: RTX 3050 (Local GPU)

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 1. CONFIGURATION ---
IMG_SIZE = 380 
BATCH_SIZE = 8 # B4 is large for 4GB VRAM
EPOCHS = 35
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = Path("saved_models") / "jaundice_body_pytorch.pth"
os.makedirs("saved_models", exist_ok=True)

print(f"🚀 Training on: {DEVICE}")

# --- 2. DATA PREPARATION ---
workspace_root = Path.cwd()
candidates_jaundice = [workspace_root / "Dataset" / "Dataset_Cleaned" / "jaundice"]
jaundice_dir = next((p for p in candidates_jaundice if p.exists()), None)

if not jaundice_dir:
    print("❌ ERROR: Jaundice dataset not found at expected location.")
    sys.exit(1)

normal_dir = jaundice_dir.parent / "normal"
if not normal_dir.exists():
    normal_dir = jaundice_dir.parent / "normal_processed"

def list_images(directory):
    return [str(p) for p in Path(directory).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

jaundice_imgs = list_images(jaundice_dir)
normal_imgs = list_images(normal_dir)

from sklearn.model_selection import GroupShuffleSplit

df = pd.DataFrame({
    "path": jaundice_imgs + normal_imgs,
    "label": [1] * len(jaundice_imgs) + [0] * len(normal_imgs)
})

# --- FIX DATA LEAKAGE: Sorted Block Split (The "Nuclear" Option) ---
# Previous method failed because "normal (1).jpg" and "normal (2).jpg" were treated as different patients.
# If they are video frames, this causes leakage.
# SOLUTION: Sort by filename -> Cut dataset at 80%. 
# This guarantees sequential frames stay together.

# 1. Separate Dataframes
df_jaundice = df[df["label"] == 1].sort_values("path")
df_normal = df[df["label"] == 0].sort_values("path")

# 2. Split each class independently by index (No Shuffle yet)
def block_split(sub_df, split_ratio=0.8):
    cut_idx = int(len(sub_df) * split_ratio)
    return sub_df.iloc[:cut_idx], sub_df.iloc[cut_idx:]

train_j, val_j = block_split(df_jaundice)
train_n, val_n = block_split(df_normal)

# 3. Combine and Shuffle Train only (Keep Val pure)
train_df = pd.concat([train_j, train_n]).sample(frac=1, random_state=42).reset_index(drop=True)
val_df = pd.concat([val_j, val_n]).reset_index(drop=True)

print(f"✅ Split Complete (Sorted Block Split):")
print(f"   - Train: {len(train_df)} images")
print(f"   - Val:   {len(val_df)} images")
print(f"   - Val Samples: {[Path(p).name for p in val_df['path'].head(3)]} ...")

class JaundiceDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["path"]
        image = Image.open(img_path).convert("RGB")
        label = self.df.iloc[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20), # increased rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Add Color Jitter
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Calculate Class Weights for Imbalanced Loss
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
class_weight_tensor = torch.tensor(class_weights[1], dtype=torch.float32).to(DEVICE) # Weight for Positive Class (Jaundice)
print(f"⚖️ Class Weight for Jaundice: {class_weight_tensor.item():.4f}")

train_loader = DataLoader(JaundiceDataset(train_df, train_tf), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(JaundiceDataset(val_df, val_tf), batch_size=BATCH_SIZE)

# --- 3. MODEL BUILDING ---
class JaundiceBodyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use EfficientNet-B4 to match the quality of the Eye model
        self.backbone = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0)
        self.num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased Dropout
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

model = JaundiceBodyModel().to(DEVICE)
# Use pos_weight for imbalanced dataset
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- 4. TRAINING LOOP ---
from sklearn.metrics import balanced_accuracy_score

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix({"loss": train_loss/len(train_loader)})
    
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels_gpu = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            outputs = model(imgs)
            val_loss += criterion(outputs, labels_gpu).item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Use Balanced Accuracy
    b_acc = balanced_accuracy_score(all_labels, all_preds) * 100
    
    print(f"📊 Val Balanced Acc: {b_acc:.2f}% | Loss: {avg_val_loss:.4f}")
    
    if b_acc > best_acc:
        best_acc = b_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"⭐ Saved Best Model to {MODEL_SAVE_PATH}")

print(f"✅ DONE. Best Balanced Acc: {best_acc:.2f}%")
