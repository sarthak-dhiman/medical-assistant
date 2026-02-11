"""
Jaundice Body Detection Training Script (Simplified & Practical)
Key Features:
- Sorted Block Split (prevents video frame leakage)
- Basic augmentation (flip, brightness, rotation)
- Standard training loop
- Early stopping
"""
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
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import cv2

# ============================================================================
# DATASET
# ============================================================================

class JaundiceBodyDataset(Dataset):
    def __init__(self, dataframe, transform=None, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.augment = augment
    
    def __len__(self):
        return len(self.df)
    
    def augment_image(self, img):
        """Simple augmentation"""
        img = np.array(img)
        
        # Flip
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        
        # Brightness/Contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
        
        return Image.fromarray(img)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["path"]
        image = Image.open(img_path).convert("RGB")
        
        if self.augment:
            image = self.augment_image(image)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.df.iloc[idx]["label"]
        return image, torch.tensor(label, dtype=torch.float32)


# ============================================================================
# MODEL
# ============================================================================

class JaundiceBodyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.classifier(self.backbone(x))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Config
    IMG_SIZE = 380
    BATCH_SIZE = 8
    EPOCHS = 35
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MODEL_SAVE_PATH = Path("saved_models") / "jaundice_body_pytorch.pth"
    os.makedirs("saved_models", exist_ok=True)
    
    print(f"Training on: {DEVICE}")
    
    # Load data
    DATASET_ROOT = Path(r"D:\Disease Prediction\Dataset\baby_body_clean")
    JAUNDICE_DIR = DATASET_ROOT / "jaundice"
    NORMAL_DIR = DATASET_ROOT / "normal"
    
    if not JAUNDICE_DIR.exists() or not NORMAL_DIR.exists():
        print(f"ERROR: Clean baby dataset not found at {DATASET_ROOT}")
        print("Please run prepare_baby_dataset.py first!")
        sys.exit(1)
    
    def list_images(directory):
        if not directory.exists():
            return []
        return [str(p) for p in directory.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    jaundice_imgs = list_images(JAUNDICE_DIR)
    normal_imgs = list_images(NORMAL_DIR)
    
    print(f"Found {len(jaundice_imgs)} Jaundice, {len(normal_imgs)} Normal images")
    
    if not jaundice_imgs or not normal_imgs:
        print("ERROR: Dataset not found!")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame({
        "path": jaundice_imgs + normal_imgs,
        "label": [1] * len(jaundice_imgs) + [0] * len(normal_imgs)
    })
    
    # CRITICAL: Sorted Block Split (prevents video frame leakage)
    print("Using Sorted Block Split...")
    
    df_jaundice = df[df["label"] == 1].sort_values("path").reset_index(drop=True)
    df_normal = df[df["label"] == 0].sort_values("path").reset_index(drop=True)
    
    # Split 80/20
    cut_j = int(len(df_jaundice) * 0.8)
    cut_n = int(len(df_normal) * 0.8)
    
    train_df = pd.concat([
        df_jaundice.iloc[:cut_j],
        df_normal.iloc[:cut_n]
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    val_df = pd.concat([
        df_jaundice.iloc[cut_j:],
        df_normal.iloc[cut_n:]
    ]).reset_index(drop=True)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Data loaders
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = JaundiceBodyDataset(train_df, transform=tf, augment=True)
    val_dataset = JaundiceBodyDataset(val_df, transform=tf, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Model setup
    model = JaundiceBodyModel().to(DEVICE)
    
    # Calculate class weights for imbalanced dataset
    # We have ~3.6x more jaundice images than normal
    # So we weight normal class higher (pos_weight = jaundice_count / normal_count)
    jaundice_count = len(jaundice_imgs)
    normal_count = len(normal_imgs)
    # Cap weight to avoid over-punishing (stability)
    # limit max weight to 3.0 to prevent exploding loss on "Normal" misclassification
    raw_weight = jaundice_count / normal_count
    capped_weight = min(raw_weight, 3.0)
    pos_weight = torch.tensor([capped_weight]).to(DEVICE)
    
    print(f"\n Class Weighting:")
    print(f"Jaundice: {jaundice_count} images")
    print(f"Normal:   {normal_count} images")
    print(f"Pos Weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_acc = 0.0
    patience = 8
    patience_counter = 0
    
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
        
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_acc = (np.array(all_preds).flatten() == np.array(all_labels).flatten()).mean()
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved (Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping")
            break
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).unsqueeze(1)
            outputs = model(imgs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    print(f"\nBest Accuracy: {best_acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Jaundice']))
    
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
