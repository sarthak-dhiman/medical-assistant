import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from pathlib import Path
import sys
import json

# --- Configuration ---
IMG_SIZE = (380, 380) # EfficientNet-B4 Native Resolution
SCLERA_SIZE = (64, 64)

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

# AUTO DETECT GPU:
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEBUG: PyTorch will use device: {DEVICE}", flush=True)

# --- Model Architectures (Must Match Training) ---

# 1. Jaundice Eye Model (EfficientNet-B4 + Sclera Branch)
# 1. Jaundice Eye Model (EfficientNet-B4 + Sclera Branch)
class JaundiceModel(nn.Module):
    def __init__(self):
        super(JaundiceModel, self).__init__()
        # Skin Branch: EfficientNet-B4
        self.skin_backbone = timm.create_model('tf_efficientnet_b4', pretrained=False, num_classes=0)
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
        
        # Fusion Classifier with Stronger Regularization (Matches training script)
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

# 2. Jaundice Body Model (EfficientNet-B4)
class JaundiceBodyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use simple B4 backbone matching local training script
        self.backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.classifier(self.backbone(x))

# 3. Skin Disease Model (EfficientNet-B4)
class SkinDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
        self.num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# --- Resource Loading ---
_eye_model = None
_body_model = None
_skin_model = None
_skin_classes = {}

def get_eye_model():
    global _eye_model
    if _eye_model: return _eye_model
    path = BASE_DIR / "saved_models" / "jaundice_with_sclera_torch.pth"
    if not path.exists(): return None
    try:
        model = JaundiceModel().to(DEVICE)
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        _eye_model = model
        print("✅ Eye Model Loaded (PyTorch)", flush=True)
        return _eye_model
    except Exception as e:
        print(f"❌ Failed to load Eye Model: {e}", flush=True)
        return None

def get_body_model():
    global _body_model
    if _body_model: return _body_model
    path = BASE_DIR / "saved_models" / "jaundice_body_pytorch.pth"
    if not path.exists(): return None
    try:
        model = JaundiceBodyModel().to(DEVICE)
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        _body_model = model
        print("✅ Body Model Loaded (PyTorch)", flush=True)
        return _body_model
    except Exception as e:
        print(f"❌ Failed to load Body Model: {e}", flush=True)
        return None

def get_skin_model():
    global _skin_model, _skin_classes
    if _skin_model: return _skin_model
    
    model_path = BASE_DIR / "saved_models" / "skin_disease_pytorch.pth"
    map_path = BASE_DIR / "saved_models" / "skin_disease_mapping.json"
    
    if not model_path.exists() or not map_path.exists(): 
        print(f"⚠️ Skin model files missing: {model_path} or {map_path}", flush=True)
        return None
    
    try:
        # Load mapping first
        with open(map_path, 'r') as f:
            raw_map = json.load(f)
            
        # Ensure mapping is int -> string
        # Some JSON libs load '0' as string key. We need int keys for lookup if we predict by index.
        # But wait - usually we just get argmax index.
        # Let's handle both { "0": "Acne" } and { "Acne": 0 }
        
        # Heuristic: if values are ints, it's {Class: Index}. Invert it.
        first_val = list(raw_map.values())[0]
        if isinstance(first_val, int):
            _skin_classes = {v: k for k, v in raw_map.items()}
        else:
            # It's {"0": "Acne"}. Convert keys to int.
            _skin_classes = {int(k): v for k, v in raw_map.items()}
            
        num_classes = len(_skin_classes)
        
        model = SkinDiseaseModel(num_classes=num_classes).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        _skin_model = model
        print(f"✅ Skin Model Loaded (PyTorch) - {num_classes} classes", flush=True)
        return _skin_model
    except Exception as e:
        print(f"❌ Failed to load Skin Model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

# --- Prediction Functions ---

def _preprocess_img(img_bgr, size=(380,380)):
    """Standard EfficientNet preprocessing: Resize -> RGB -> Normalize -> Tensor"""
    # Resize
    img_resized = cv2.resize(img_bgr, size)
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize (Manual implementation instead of torchvision transforms to keep inference light)
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    img_float = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_norm = (img_float - mean) / std
    
    # HWC -> CHW
    img_chw = np.transpose(img_norm, (2, 0, 1))
    
    # Batch dim
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return torch.tensor(img_batch, dtype=torch.float32).to(DEVICE)

def predict_jaundice_eye(skin_img, sclera_crop=None):
    model = get_eye_model()
    if not model: return "Model Missing", 0.0
    
    # Preprocess Skin - MATCH NEW TRAINING: ImageNet normalization
    skin_resized = cv2.resize(skin_img, IMG_SIZE)
    skin_rgb = cv2.cvtColor(skin_resized, cv2.COLOR_BGR2RGB)
    skin_float = skin_rgb.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    skin_norm = (skin_float - mean) / std
    
    skin_chw = np.transpose(skin_norm, (2, 0, 1))
    skin_t = torch.tensor(np.expand_dims(skin_chw, axis=0), dtype=torch.float32).to(DEVICE)
    
    # Preprocess Sclera (64x64) - MATCH NEW TRAINING: ImageNet normalization
    if sclera_crop is None or sclera_crop.size == 0:
        sclera_crop = np.zeros((*SCLERA_SIZE, 3), dtype=np.uint8)
    else:
        sclera_crop = cv2.resize(sclera_crop, SCLERA_SIZE)
        
    sclera_rgb = cv2.cvtColor(sclera_crop, cv2.COLOR_BGR2RGB)
    sclera_float = sclera_rgb.astype(np.float32) / 255.0
    sclera_norm = (sclera_float - mean) / std
    sclera_chw = np.transpose(sclera_norm, (2, 0, 1))
    sclera_t = torch.tensor(np.expand_dims(sclera_chw, axis=0), dtype=torch.float32).to(DEVICE)
        
    with torch.no_grad():
        logits = model(skin_t, sclera_t)
        prob = torch.sigmoid(logits).item()
    
    # Standard threshold after retraining with balanced data
    threshold = 0.5
    label = "Jaundice" if prob > threshold else "Normal"
    
    if prob > threshold:
        conf = prob  # Confidence in Jaundice
    else:
        conf = 1.0 - prob  # Confidence in Normal
    
    return label, conf

def predict_jaundice_body(img_bgr):
    model = get_body_model()
    if not model: return "Model Missing", 0.0
    
    img_t = _preprocess_img(img_bgr, size=IMG_SIZE)
    with torch.no_grad():
        logits = model(img_t)
        prob = torch.sigmoid(logits).item()
    
    # STRICTER THRESHOLD: Model trained on babies, less reliable on adults
    # Require 70% confidence instead of 50% to reduce false positives
    threshold = 0.70
    label = "Jaundice" if prob > threshold else "Normal"
    
    # Return actual probability as confidence
    if prob > threshold:
        conf = prob  # Confidence in Jaundice prediction
    else:
        conf = 1.0 - prob  # Confidence in Normal prediction
    
    return label, conf

def predict_skin_disease_torch(img_bgr):
    model = get_skin_model()
    if not model: return "Model Missing", 0.0
    
    img_t = _preprocess_img(img_bgr, size=IMG_SIZE)
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)
        
    idx = idx.item()
    conf = conf.item()
    
    # Default to "Normal" if confidence < 80% to reduce false positives
    if conf < 0.80:
        label = "Normal"
    else:
        label = _skin_classes.get(idx, f"Unknown Class {idx}")
    
    return label, conf
