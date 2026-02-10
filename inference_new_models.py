"""
Inference functions for new disease detection models
Burns, Hairloss, Nail Disease, Pressure Ulcer
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from pathlib import Path
import json
import threading
import sys

# Configuration
IMG_SIZE_LARGE = (380, 380)  # Burns, Nail, Pressure Ulcer
IMG_SIZE_SMALL = (224, 224)  # Hairloss

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global model variables
_burns_model = None
_hairloss_model = None
_nail_model = None
_pressure_ulcer_model = None
_nail_classes = None
_pressure_ulcer_classes = None

# Thread-safe locks
_model_locks = {
    'burns': threading.Lock(),
    'hairloss': threading.Lock(),
    'nail': threading.Lock(),
    'pressure_ulcer': threading.Lock()
}


# --- Model Architectures ---

class BurnsModel(nn.Module):
    """Binary classification: Burn vs Healthy"""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
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


class HairlossModel(nn.Module):
    """Binary classification: Bald vs Not Bald"""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class NailDiseaseModel(nn.Module):
    """8-class classification for nail pathologies"""
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
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


class PressureUlcerModel(nn.Module):
    """4-class ordinal classification: Stage 1-4"""
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# --- Model Loading Functions (Thread-Safe) ---

def get_burns_model():
    global _burns_model
    if _burns_model:
        return _burns_model
    
    with _model_locks['burns']:
        if _burns_model:
            return _burns_model
        
        path = BASE_DIR / "saved_models" / "burns_pytorch.pth"
        if not path.exists():
            return None
        
        try:
            print("🔄 Loading Burns Model (Thread-Safe)...", flush=True)
            model = BurnsModel().to(DEVICE)
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            _burns_model = model
            print("✅ Burns Model Loaded", flush=True)
            return _burns_model
        except Exception as e:
            print(f"❌ Failed to load Burns Model: {e}", flush=True)
            return None


def get_hairloss_model():
    global _hairloss_model
    if _hairloss_model:
        return _hairloss_model
    
    with _model_locks['hairloss']:
        if _hairloss_model:
            return _hairloss_model
        
        path = BASE_DIR / "saved_models" / "hairloss_pytorch.pth"
        if not path.exists():
            return None
        
        try:
            print("🔄 Loading Hairloss Model (Thread-Safe)...", flush=True)
            model = HairlossModel().to(DEVICE)
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            _hairloss_model = model
            print("✅ Hairloss Model Loaded", flush=True)
            return _hairloss_model
        except Exception as e:
            print(f"❌ Failed to load Hairloss Model: {e}", flush=True)
            return None


def get_nail_model():
    global _nail_model, _nail_classes
    if _nail_model:
        return _nail_model
    
    with _model_locks['nail']:
        if _nail_model:
            return _nail_model
        
        model_path = BASE_DIR / "saved_models" / "nail_disease_pytorch.pth"
        map_path = BASE_DIR / "saved_models" / "nail_disease_mapping.json"
        
        if not model_path.exists() or not map_path.exists():
            print(f"⚠️ Nail model files missing", flush=True)
            return None
        
        try:
            print("🔄 Loading Nail Disease Model (Thread-Safe)...", flush=True)
            
            with open(map_path, 'r') as f:
                raw_map = json.load(f)
            
            # Convert to int keys
            _nail_classes = {int(k): v for k, v in raw_map.items()}
            num_classes = len(_nail_classes)
            
            model = NailDiseaseModel(num_classes=num_classes).to(DEVICE)
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            _nail_model = model
            print(f"✅ Nail Disease Model Loaded - {num_classes} classes", flush=True)
            return _nail_model
        except Exception as e:
            print(f"❌ Failed to load Nail Model: {e}", flush=True)
            return None


def get_pressure_ulcer_model():
    global _pressure_ulcer_model, _pressure_ulcer_classes
    if _pressure_ulcer_model:
        return _pressure_ulcer_model
    
    with _model_locks['pressure_ulcer']:
        if _pressure_ulcer_model:
            return _pressure_ulcer_model
        
        model_path = BASE_DIR / "saved_models" / "pressure_ulcer_pytorch.pth"
        map_path = BASE_DIR / "saved_models" / "pressure_ulcer_mapping.json"
        
        if not model_path.exists() or not map_path.exists():
            print(f"⚠️ Pressure ulcer model files missing", flush=True)
            return None
        
        try:
            print("🔄 Loading Pressure Ulcer Model (Thread-Safe)...", flush=True)
            
            with open(map_path, 'r') as f:
                raw_map = json.load(f)
            
            _pressure_ulcer_classes = {int(k): v for k, v in raw_map.items()}
            num_classes = len(_pressure_ulcer_classes)
            
            model = PressureUlcerModel(num_classes=num_classes).to(DEVICE)
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            _pressure_ulcer_model = model
            print(f"✅ Pressure Ulcer Model Loaded - {num_classes} stages", flush=True)
            return _pressure_ulcer_model
        except Exception as e:
            print(f"❌ Failed to load Pressure Ulcer Model: {e}", flush=True)
            return None


# --- Preprocessing Functions ---

def preprocess_image(img_bgr, target_size):
    """Resize and normalize image for inference"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    # Convert to tensor (CHW format)
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor.to(DEVICE)


# --- Inference Functions ---

def predict_burns(img_bgr, debug=False):
    """
    Predict burn detection
    Returns: (label, confidence, debug_info)
    """
    model = get_burns_model()
    if model is None:
        return "Model Not Loaded", 0.0, {"error": "Burns model not available"}
    
    debug_info = {}
    
    try:
        img_tensor = preprocess_image(img_bgr, IMG_SIZE_LARGE)
        
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
        
        # 0 = healthy, 1 = burn
        if prob > 0.5:
            label = "Burn"
            conf = prob
        else:
            label = "Healthy"
            conf = 1 - prob
        
        debug_info["raw_probability"] = float(prob)
        
        return label, conf, debug_info
        
    except Exception as e:
        return "Error", 0.0, {"error": str(e)}


def predict_hairloss(img_bgr, debug=False):
    """
    Predict hairloss detection
    Returns: (label, confidence, debug_info)
    """
    model = get_hairloss_model()
    if model is None:
        return "Model Not Loaded", 0.0, {"error": "Hairloss model not available"}
    
    debug_info = {}
    
    try:
        img_tensor = preprocess_image(img_bgr, IMG_SIZE_SMALL)
        
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
        
        # 0 = not bald, 1 = bald
        if prob > 0.5:
            label = "Bald"
            conf = prob
        else:
            label = "Not Bald"
            conf = 1 - prob
        
        debug_info["raw_probability"] = float(prob)
        
        return label, conf, debug_info
        
    except Exception as e:
        return "Error", 0.0, {"error": str(e)}


def predict_nail_disease(img_bgr, debug=False):
    """
    Predict nail disease
    Returns: (label, confidence, debug_info)
    """
    model = get_nail_model()
    if model is None or _nail_classes is None:
        return "Model Not Loaded", 0.0, {"error": "Nail disease model not available"}
    
    debug_info = {}
    
    try:
        img_tensor = preprocess_image(img_bgr, IMG_SIZE_LARGE)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, dim=1)
            conf = conf.item()
            idx = idx.item()
        
        label = _nail_classes.get(idx, f"Unknown Class {idx}")
        
        # Top-3 predictions
        topk_conf, topk_idx = torch.topk(probs, min(3, len(_nail_classes)))
        topk_conf = topk_conf[0].cpu().numpy()
        topk_idx = topk_idx[0].cpu().numpy()
        
        top_3 = []
        for i in range(len(topk_idx)):
            class_name = _nail_classes.get(int(topk_idx[i]), f"Class {topk_idx[i]}")
            score = float(topk_conf[i])
            top_3.append({"label": class_name, "confidence": score})
        
        debug_info["top_3"] = top_3
        
        return label, conf, debug_info
        
    except Exception as e:
        return "Error", 0.0, {"error": str(e)}


def predict_pressure_ulcer(img_bgr, debug=False):
    """
    Predict pressure ulcer stage
    Returns: (label, confidence, debug_info)
    """
    model = get_pressure_ulcer_model()
    if model is None or _pressure_ulcer_classes is None:
        return "Model Not Loaded", 0.0, {"error": "Pressure ulcer model not available"}
    
    debug_info = {}
    
    try:
        img_tensor = preprocess_image(img_bgr, IMG_SIZE_LARGE)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, dim=1)
            conf = conf.item()
            idx = idx.item()
        
        label = _pressure_ulcer_classes.get(idx, f"Stage {idx+1}")
        
        # All stage probabilities
        stage_probs = {}
        for i in range(len(_pressure_ulcer_classes)):
            stage_name = _pressure_ulcer_classes.get(i, f"Stage {i+1}")
            stage_probs[stage_name] = float(probs[0][i].item())
        
        debug_info["stage_probabilities"] = stage_probs
        
        return label, conf, debug_info
        
    except Exception as e:
        return "Error", 0.0, {"error": str(e)}
