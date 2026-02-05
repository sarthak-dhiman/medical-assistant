import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from pathlib import Path

import sys

# --- Configuration ---
IMG_SIZE = (380, 380)
SCLERA_SIZE = (64, 64)

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "saved_models" / "jaundice_with_sclera_torch.pth"
# AUTO DETECT GPU:
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEBUG: PyTorch will use device: {DEVICE}", flush=True)

# --- Model Arch (Must Match Training) ---
class JaundiceModel(nn.Module):
    def __init__(self):
        super(JaundiceModel, self).__init__()
        self.skin_backbone = timm.create_model('tf_efficientnet_b4', pretrained=False, num_classes=0)
        self.skin_dim = self.skin_backbone.num_features
        
        self.sclera_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.sclera_dim = 64
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.skin_dim + self.sclera_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, skin, sclera):
        x1 = self.skin_backbone(skin)
        x2 = self.sclera_branch(sclera)
        x2 = x2.view(x2.size(0), -1)
        concat = torch.cat((x1, x2), dim=1)
        logits = self.classifier(concat)
        return logits

# --- Resource Loading ---
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
        
    print(f"DEBUG: Checking Model Path: {MODEL_PATH}", flush=True)
    if not MODEL_PATH.exists():
        print(f"⚠️ Model not found at {MODEL_PATH}", flush=True)
        return None
        
    print(f"DEBUG: Loading PyTorch Model from {MODEL_PATH}...", flush=True)
    try:
        print("DEBUG: Initializing Architecture...", flush=True)
        model = JaundiceModel()
        print(f"DEBUG: Moving to device {DEVICE}...", flush=True)
        model = model.to(DEVICE)
        
        print("DEBUG: Loading State Dict...", flush=True)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        
        print("DEBUG: Setting to Eval...", flush=True)
        model.eval()
        _model = model
        print("✅ Model loaded successfully.", flush=True)
        return _model
    except Exception as e:
        print(f"❌ Error loading model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def predict_jaundice(skin_img, sclera_crop=None):
    """
    skin_img: BGR numpy array
    sclera_crop: BGR numpy array (optional, will be black if None)
    """
    model = get_model()
    if model is None:
        return "Model Missing", 0.0
        
    # Preprocess Skin
    skin_resized = cv2.resize(skin_img, IMG_SIZE)
    skin_rgb = cv2.cvtColor(skin_resized, cv2.COLOR_BGR2RGB)
    skin_tensor = torch.from_numpy(skin_rgb).permute(2, 0, 1).float() / 255.0
    skin_tensor = skin_tensor.unsqueeze(0).to(DEVICE) # (1, 3, 380, 380)
    
    # Preprocess Sclera
    if sclera_crop is None or sclera_crop.size == 0:
        sclera_crop = np.zeros((*SCLERA_SIZE, 3), dtype=np.uint8)
    else:
        sclera_crop = cv2.resize(sclera_crop, SCLERA_SIZE)
        
    sclera_rgb = cv2.cvtColor(sclera_crop, cv2.COLOR_BGR2RGB)
    sclera_tensor = torch.from_numpy(sclera_rgb).permute(2, 0, 1).float() / 255.0
    sclera_tensor = sclera_tensor.unsqueeze(0).to(DEVICE) # (1, 3, 64, 64)
    
    # Inference
    with torch.no_grad():
        logits = model(skin_tensor, sclera_tensor)
        prob = torch.sigmoid(logits).item()
        
    label = "Jaundice" if prob > 0.5 else "Normal"
    # Confidence: distance from 0.5
    confidence = prob if prob > 0.5 else 1.0 - prob
    
    return label, confidence
