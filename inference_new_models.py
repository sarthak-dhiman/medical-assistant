
"""
Inference functions for new disease detection models
Burns, Hairloss, Nail Disease
"""

import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from pathlib import Path
import json
import threading
import base64
import sys
from vis_utils import GradCAM, generate_heatmap_overlay

# Configuration
IMG_SIZE_LARGE = (380, 380)  # Burns, Nail
IMG_SIZE_SMALL = (224, 224)  # Hairloss

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

# Lazy Device Loading
_DEVICE = None

def get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _DEVICE

# Global model variables
_burns_model = None
_nail_model = None
_nail_classes = None

# Thread-safe locks
_model_locks = {
    'burns': threading.Lock(),
    'nail': threading.Lock()
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
            print("Loading Burns Model (Thread-Safe)...", flush=True)
            model = BurnsModel().to(get_device())
            state_dict = torch.load(path, map_location=get_device())
            model.load_state_dict(state_dict)
            model.eval()
            _burns_model = model
            print("Burns Model Loaded", flush=True)
            return _burns_model
        except Exception as e:
            print(f"Failed to load Burns Model: {e}", flush=True)
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
            print(f"Nail model files missing", flush=True)
            return None
        
        try:
            print("Loading Nail Disease Model (Thread-Safe)...", flush=True)
            
            with open(map_path, 'r') as f:
                raw_map = json.load(f)
            
            # Convert to int keys
            _nail_classes = {int(k): v for k, v in raw_map.items()}
            num_classes = len(_nail_classes)
            
            model = NailDiseaseModel(num_classes=num_classes).to(get_device())
            state_dict = torch.load(model_path, map_location=get_device())
            model.load_state_dict(state_dict)
            model.eval()
            _nail_model = model
            print(f"Nail Disease Model Loaded - {num_classes} classes", flush=True)
            return _nail_model
        except Exception as e:
            print(f"Failed to load Nail Model: {e}", flush=True)
            return None





# --- Preprocessing Functions ---

def preprocess_image(img_bgr, target_size):
    """Resize and normalize image for inference"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_normalized - mean) / std
    
    # Convert to tensor (CHW format)
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor.to(get_device()), img_resized


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
        print("Preprocessing Burns Image...", flush=True)
        img_tensor, img_resized = preprocess_image(img_bgr, IMG_SIZE_LARGE)
        
        # Setup Grad-CAM if debug mode is on
        context = torch.enable_grad() if debug else torch.no_grad()
        grad_cam = None
        
        if debug:
            if hasattr(model.backbone, 'conv_head'):
                target_layer = model.backbone.conv_head
            elif hasattr(model.backbone, 'blocks'):
                target_layer = model.backbone.blocks[-1]
            else:
                target_layer = list(model.backbone.children())[-1]
            grad_cam = GradCAM(model, target_layer)

        print("Running Burns Model Forward Pass...", flush=True)
        with context:
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
            
            if debug and grad_cam:
                # Backward pass for Grad-CAM
                model.zero_grad()
                output.backward(retain_graph=False)
                heatmap = grad_cam.generate()
                
                if heatmap is not None:
                    overlay = generate_heatmap_overlay(heatmap, img_resized)
                    debug_info["grad_cam"] = overlay
            
        print(f"Burns Prob: {prob}", flush=True)
        
        # 0 = healthy, 1 = burn
        if prob > 0.5:
            label = "Burns Detected"
            conf = prob
        else:
            label = "Healthy/Normal"
            conf = 1 - prob
        
        debug_info["raw_probability"] = float(prob)
        
        # Add Debug Image (Input Crop)
        if debug:
            _, buffer = cv2.imencode('.jpg', img_resized)
            debug_info["debug_image"] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        
        if grad_cam:
            grad_cam.remove_hooks()
            model.zero_grad()
            
        return label, conf, debug_info
        
    except Exception as e:
        print(f"Error in predict_burns: {e}")
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
        img_tensor, img_resized = preprocess_image(img_bgr, IMG_SIZE_LARGE)
        
        # Setup Grad-CAM if debug mode is on
        context = torch.enable_grad() if debug else torch.no_grad()
        grad_cam = None
        
        if debug:
            if hasattr(model.backbone, 'conv_head'):
                target_layer = model.backbone.conv_head
            elif hasattr(model.backbone, 'blocks'):
                target_layer = model.backbone.blocks[-1]
            else:
                target_layer = list(model.backbone.children())[-1]
            grad_cam = GradCAM(model, target_layer)

        with context:
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, dim=1)
            conf = conf.item()
            idx = idx.item()
            
            if debug and grad_cam:
                # Backward pass for Grad-CAM
                score = output[0, idx]
                model.zero_grad()
                score.backward(retain_graph=False)
                heatmap = grad_cam.generate()
                
                if heatmap is not None:
                    overlay = generate_heatmap_overlay(heatmap, img_resized)
                    debug_info["grad_cam"] = overlay
            
        label = _nail_classes.get(idx, f"Unknown Class {idx}")
        
        # Top-3 predictions
        topk_conf, topk_idx = torch.topk(probs, min(3, len(_nail_classes)))
        topk_conf = topk_conf[0].cpu().numpy()
        topk_idx = topk_idx[0].cpu().numpy()
        
        top3 = []
        for i in range(len(topk_idx)): # Iterate up to the actual number of topk results
            class_name = _nail_classes.get(int(topk_idx[i]), f"Class {topk_idx[i]}")
            top3.append({"label": class_name, "confidence": float(topk_conf[i])})
        
        debug_info["top_3"] = top3
        
        # Add Debug Image (Input Crop)
        if debug:
            _, buffer = cv2.imencode('.jpg', img_resized)
            debug_info["debug_image"] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
            
        if grad_cam:
            grad_cam.remove_hooks()
            model.zero_grad()
        
        return label, conf, debug_info
        
    except Exception as e:
        return "Error", 0.0, {"error": str(e)}


