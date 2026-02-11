import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from pathlib import Path
import sys
import json
import threading
import base64
from vis_utils import GradCAM, generate_heatmap_overlay

# --- Configuration ---
IMG_SIZE = (380, 380)  # EfficientNet-B4 Native Resolution
SCLERA_SIZE = (64, 64)
MIN_IMAGE_SIZE = 10  # Minimum acceptable image dimension
MAX_IMAGE_SIZE = 4096  # Maximum acceptable image dimension
CONFIDENCE_THRESHOLD_JAUNDICE_EYE = 0.5
CONFIDENCE_THRESHOLD_JAUNDICE_BODY = 0.70
CONFIDENCE_THRESHOLD_SKIN_DISEASE = 0.80

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
        print(f"DEBUG: PyTorch using device: {_DEVICE}", flush=True)
    return _DEVICE

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

# Thread-safe model loading locks
_model_locks = {
    'eye': threading.Lock(),
    'body': threading.Lock(),
    'skin': threading.Lock()
}

def get_eye_model():
    global _eye_model
    # Fast path: model already loaded
    if _eye_model: 
        return _eye_model
    
    # Thread-safe loading with double-checked locking
    with _model_locks['eye']:
        # Double-check after acquiring lock
        if _eye_model:
            return _eye_model
            
        path = BASE_DIR / "saved_models" / "jaundice_with_sclera_torch.pth"
        if not path.exists(): 
            return None
            
        try:
            print("Loading Eye Model (Thread-Safe)...", flush=True)
            model = JaundiceModel().to(get_device())
            state_dict = torch.load(path, map_location=get_device())
            model.load_state_dict(state_dict)
            model.eval()
            _eye_model = model
            print("Eye Model Loaded (PyTorch)", flush=True)
            return _eye_model
        except Exception as e:
            print(f"Failed to load Eye Model: {e}", flush=True)
            return None

def get_body_model():
    global _body_model
    # Fast path: model already loaded
    if _body_model: 
        return _body_model
    
    # Thread-safe loading with double-checked locking
    with _model_locks['body']:
        # Double-check after acquiring lock
        if _body_model:
            return _body_model
            
        path = BASE_DIR / "saved_models" / "jaundice_body_pytorch.pth"
        if not path.exists(): 
            return None
            
        try:
            print("🔄 Loading Body Model (Thread-Safe)...", flush=True)
            model = JaundiceBodyModel().to(get_device())
            state_dict = torch.load(path, map_location=get_device())
            model.load_state_dict(state_dict)
            model.eval()
            _body_model = model
            print("Body Model Loaded (PyTorch)", flush=True)
            return _body_model
        except Exception as e:
            print(f"Failed to load Body Model: {e}", flush=True)
            return None

def get_skin_model():
    global _skin_model, _skin_classes
    # Fast path: model already loaded
    if _skin_model: 
        return _skin_model
    
    # Thread-safe loading with double-checked locking
    with _model_locks['skin']:
        # Double-check after acquiring lock
        if _skin_model:
            return _skin_model
            
        model_path = BASE_DIR / "saved_models" / "skin_disease_pytorch.pth"
        map_path = BASE_DIR / "saved_models" / "skin_disease_mapping.json"
        
        if not model_path.exists() or not map_path.exists(): 
            print(f"Skin model files missing: {model_path} or {map_path}", flush=True)
            return None
        
        try:
            print("Loading Skin Model (Thread-Safe)...", flush=True)
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
            
            model = SkinDiseaseModel(num_classes=num_classes).to(get_device())
            state_dict = torch.load(model_path, map_location=get_device())
            model.load_state_dict(state_dict)
            model.eval()
            _skin_model = model
            print(f"Skin Model Loaded (PyTorch) - {num_classes} classes", flush=True)
            return _skin_model
        except Exception as e:
            print(f"Failed to load Skin Model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

# --- Input Validation ---
def validate_image(img_bgr, context="image"):
    """Comprehensive image validation with detailed error messages."""
    errors = []
    
    # Check if image is None
    if img_bgr is None:
        return False, f"{context} is None"
    
    # Check if image is numpy array
    if not isinstance(img_bgr, np.ndarray):
        return False, f"{context} is not a numpy array, got {type(img_bgr)}"
    
    # Check if image is empty
    if img_bgr.size == 0:
        return False, f"{context} is empty (size=0)"
    
    # Check dimensions
    if len(img_bgr.shape) < 2:
        return False, f"{context} has insufficient dimensions: {img_bgr.shape}"
    
    h, w = img_bgr.shape[:2]
    
    # Check minimum size
    if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
        return False, f"{context} too small: {w}x{h} (min: {MIN_IMAGE_SIZE})"
    
    # Check maximum size
    if h > MAX_IMAGE_SIZE or w > MAX_IMAGE_SIZE:
        return False, f"{context} too large: {w}x{h} (max: {MAX_IMAGE_SIZE})"
    
    # Check for NaN or Inf values
    if np.isnan(img_bgr).any():
        return False, f"{context} contains NaN values"
    
    if np.isinf(img_bgr).any():
        return False, f"{context} contains Inf values"
    
    # Check for uniform image (all pixels same color)
    if img_bgr.std() < 1.0:
        errors.append(f"Warning: {context} appears to be uniform (no variation)")
    
    return True, "; ".join(errors) if errors else "Valid"


def ensure_3channel(img):
    """Ensure image has 3 channels (BGR format)."""
    if len(img.shape) == 2:
        # Grayscale to BGR
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        # Single channel to BGR
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        # Already BGR
        return img
    elif img.shape[2] == 4:
        # RGBA to BGR
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] > 4:
        # Take first 3 channels
        return img[:, :, :3]
    else:
        # Fallback: convert to gray then BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] >= 3 else img
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# --- Prediction Functions ---

def _preprocess_img(img_bgr, size=(380, 380)):
    """Standard EfficientNet preprocessing: Resize -> RGB -> Normalize -> Tensor
    
    Edge cases handled:
    - Empty or None images
    - Wrong number of channels
    - Extreme aspect ratios
    - NaN/Inf values
    - Memory allocation errors
    """
    try:
        # Validate input
        is_valid, msg = validate_image(img_bgr, "preprocess input")
        if not is_valid:
            print(f"Preprocess validation failed: {msg}", flush=True)
            return None
        
        # Ensure 3-channel BGR
        img_bgr = ensure_3channel(img_bgr)
        
        # Resize with aspect ratio preservation (letterboxing if needed)
        h, w = img_bgr.shape[:2]
        target_h, target_w = size
        
        # Calculate scaling factor to fit within target while preserving aspect ratio
        scale = min(target_w / w, target_h / h)
        
        if scale < 1.0 or (w != target_w and h != target_h):
            new_w = int(w * scale)
            new_h = int(h * scale)
            new_w = max(new_w, 1)
            new_h = max(new_h, 1)
            
            resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Create letterboxed image
            img_resized = np.full((target_h, target_w, 3), 128, dtype=np.uint8)  # Gray padding
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            img_resized[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        else:
            img_resized = cv2.resize(img_bgr, size, interpolation=cv2.INTER_LINEAR)
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize (Manual implementation instead of torchvision transforms to keep inference light)
        # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Clip values to prevent numerical issues
        img_float = np.clip(img_float, 0.0, 1.0)
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_norm = (img_float - mean) / std
        
        # Check for NaN/Inf after normalization
        if np.isnan(img_norm).any() or np.isinf(img_norm).any():
            print("Warning: NaN/Inf detected after normalization, replacing with zeros", flush=True)
            img_norm = np.nan_to_num(img_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # HWC -> CHW
        img_chw = np.transpose(img_norm, (2, 0, 1))
        
        # Batch dim
        img_batch = np.expand_dims(img_chw, axis=0)
        
        # Convert to tensor and move to device
        tensor = torch.tensor(img_batch, dtype=torch.float32).to(get_device())
        
        return tensor
        
    except Exception as e:
        print(f"Preprocessing error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Preprocessing error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

# GradCAM and generate_heatmap_overlay moved to vis_utils.py

def predict_jaundice_eye(skin_img, sclera_crop=None, debug=False):
    """Predict jaundice from eye image with comprehensive error handling.
    
    Returns: (label, confidence, debug_info)
    """
    model = get_eye_model()
    if not model: 
        return "Model Missing", 0.0, {"error": "Model not loaded"}
    
    # Validate skin image
    is_valid, msg = validate_image(skin_img, "skin image")
    if not is_valid:
        return "Invalid Input", 0.0, {"error": msg}
    
    # Ensure 3-channel BGR for skin
    skin_img = ensure_3channel(skin_img)
    
    # Preprocess Skin
    skin_resized = cv2.resize(skin_img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    skin_rgb = cv2.cvtColor(skin_resized, cv2.COLOR_BGR2RGB)
    skin_float = skin_rgb.astype(np.float32) / 255.0
    skin_float = np.clip(skin_float, 0.0, 1.0)
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    skin_norm = (skin_float - mean) / std
    
    # Handle NaN/Inf
    if np.isnan(skin_norm).any() or np.isinf(skin_norm).any():
        skin_norm = np.nan_to_num(skin_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    skin_chw = np.transpose(skin_norm, (2, 0, 1))
    skin_t = torch.tensor(np.expand_dims(skin_chw, axis=0), dtype=torch.float32).to(get_device())
    
    # Preprocess Sclera (64x64)
    if sclera_crop is None or sclera_crop.size == 0:
        sclera_crop = np.zeros((*SCLERA_SIZE, 3), dtype=np.uint8)
    else:
        # Validate sclera crop
        is_valid, msg = validate_image(sclera_crop, "sclera crop")
        if not is_valid:
            print(f"Sclera validation failed: {msg}, using zeros", flush=True)
            sclera_crop = np.zeros((*SCLERA_SIZE, 3), dtype=np.uint8)
        else:
            sclera_crop = ensure_3channel(sclera_crop)
            sclera_crop = cv2.resize(sclera_crop, SCLERA_SIZE, interpolation=cv2.INTER_LINEAR)
    
    sclera_rgb = cv2.cvtColor(sclera_crop, cv2.COLOR_BGR2RGB)
    sclera_float = sclera_rgb.astype(np.float32) / 255.0
    sclera_float = np.clip(sclera_float, 0.0, 1.0)
    sclera_norm = (sclera_float - mean) / std
    
    if np.isnan(sclera_norm).any() or np.isinf(sclera_norm).any():
        sclera_norm = np.nan_to_num(sclera_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    sclera_chw = np.transpose(sclera_norm, (2, 0, 1))
    sclera_t = torch.tensor(np.expand_dims(sclera_chw, axis=0), dtype=torch.float32).to(get_device())
    
    debug_info = {}
    grad_cam = None
    
    try:
        # Context Manager for Gradients
        context = torch.enable_grad() if debug else torch.no_grad()
        
        if debug:
            # Hook the last conv layer of the Skin Backbone (EfficientNet)
            if hasattr(model.skin_backbone, 'conv_head'):
                 target_layer = model.skin_backbone.conv_head
            elif hasattr(model.skin_backbone, 'blocks'):
                 target_layer = model.skin_backbone.blocks[-1]
            else:
                 print("DEBUG: Could not find target layer for Grad-CAM", flush=True)
                 target_layer = list(model.skin_backbone.children())[-1]
                 
            grad_cam = GradCAM(model, target_layer)
        
        with context:
            logits = model(skin_t, sclera_t)
            
            # Check for numerical issues
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Warning: NaN/Inf in model output, returning default", flush=True)
                return "Error", 0.0, {"error": "Model produced NaN/Inf output"}
            
            prob = torch.sigmoid(logits).item()
            
            if debug and grad_cam:
                # Backward for Grad-CAM
                score = logits[0]  # Single output
                model.zero_grad()
                score.backward(retain_graph=False)
                
                heatmap = grad_cam.generate()
                if heatmap is not None:
                     # Overlay on Skin Image (Original Resize)
                     overlay = generate_heatmap_overlay(heatmap, skin_resized)
                     debug_info["grad_cam"] = overlay
                     
    except torch.cuda.OutOfMemoryError:
        print("CRITICAL: GPU Out Of Memory in Jaundice Eye Inference!", flush=True)
        torch.cuda.empty_cache()
        return "GPU_OOM", 0.0, {"status": "oom_error", "error": "GPU out of memory"}
    except Exception as e:
        print(f"Inference Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return "Error", 0.0, {"error": str(e)}
    finally:
        if grad_cam:
            try:
                grad_cam.remove_hooks()
                model.zero_grad()  # Cleanup
            except Exception as e:
                print(f"Error cleaning up Grad-CAM: {e}", flush=True)
    
    # Validate probability
    if not (0.0 <= prob <= 1.0):
        print(f"Warning: Invalid probability {prob}, clamping to [0,1]", flush=True)
        prob = max(0.0, min(1.0, prob))
    
    # Classification with threshold
    threshold = CONFIDENCE_THRESHOLD_JAUNDICE_EYE
    label = "Jaundice" if prob > threshold else "Normal"
    
    if prob > threshold:
        conf = prob  # Confidence in Jaundice
    else:
        conf = 1.0 - prob  # Confidence in Normal
    
    # Ensure confidence is in valid range
    conf = max(0.0, min(1.0, conf))
    
    # --- NERD MODE STATS ---
    # sclera_crop is BGR. Calculate Mean HSV/RGB
    if sclera_crop is not None and sclera_crop.size > 0:
        try:
            # Mean RGB (BGR -> RGB)
            mean_bgr = np.mean(sclera_crop, axis=(0,1))
            mean_rgb = list(mean_bgr[::-1])  # BGR to RGB
            
            # Mean HSV
            hsv_img = cv2.cvtColor(sclera_crop, cv2.COLOR_BGR2HSV)
            mean_hsv = list(np.mean(hsv_img, axis=(0,1)))
            
            debug_info["color_stats"] = {
                "mean_rgb": [int(max(0, min(255, x))) for x in mean_rgb],
                "mean_hsv": [int(max(0, min(255, x))) for x in mean_hsv]
            }
        except Exception as e:
            print(f"Error calculating color stats: {e}", flush=True)
    
    debug_info["raw_probability"] = float(prob)
    debug_info["threshold_used"] = threshold
    
    return label, conf, debug_info


def predict_jaundice_body(img_bgr, debug=False):
    """Predict jaundice from body image with comprehensive error handling."""
    model = get_body_model()
    if not model: 
        return "Model Missing", 0.0, {"error": "Model not loaded"}
    
    # Validate input
    is_valid, msg = validate_image(img_bgr, "body image")
    if not is_valid:
        return "Invalid Input", 0.0, {"error": msg}
    
    # Ensure 3-channel
    img_bgr = ensure_3channel(img_bgr)
    
    img_t = _preprocess_img(img_bgr, size=IMG_SIZE)
    if img_t is None:
        return "Preprocessing Failed", 0.0, {"error": "Failed to preprocess image"}
    
    debug_info = {}
    grad_cam = None
    
    try:
        context = torch.enable_grad() if debug else torch.no_grad()
        
        if debug:
            # Hook backbone conv_head
            if hasattr(model.backbone, 'conv_head'):
                 target_layer = model.backbone.conv_head
            else:
                 target_layer = list(model.backbone.children())[-1]
            grad_cam = GradCAM(model, target_layer)
        
        with context:
            logits = model(img_t)
            
            # Check for numerical issues
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Warning: NaN/Inf in model output", flush=True)
                return "Error", 0.0, {"error": "Model produced NaN/Inf output"}
            
            prob = torch.sigmoid(logits).item()
            
            if debug and grad_cam:
                score = logits[0]
                model.zero_grad()
                score.backward(retain_graph=False)
                
                heatmap = grad_cam.generate()
                if heatmap is not None:
                     disp_img = cv2.resize(img_bgr, IMG_SIZE)
                     overlay = generate_heatmap_overlay(heatmap, disp_img)
                     debug_info["grad_cam"] = overlay
                     
    except torch.cuda.OutOfMemoryError:
        print("CRITICAL: GPU Out Of Memory in Jaundice Body Inference!", flush=True)
        torch.cuda.empty_cache()
        return "GPU_OOM", 0.0, {"status": "oom_error", "error": "GPU out of memory"}
    except Exception as e:
        print(f"Inference Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return "Error", 0.0, {"error": str(e)}
    finally:
        if grad_cam:
            try:
                grad_cam.remove_hooks()
                model.zero_grad()
            except Exception as e:
                print(f"Error cleaning up Grad-CAM: {e}", flush=True)

    # Validate probability
    if not (0.0 <= prob <= 1.0):
        print(f"Warning: Invalid probability {prob}, clamping", flush=True)
        prob = max(0.0, min(1.0, prob))
    
    # STRICTER THRESHOLD: Model trained on babies, less reliable on adults
    threshold = CONFIDENCE_THRESHOLD_JAUNDICE_BODY
    label = "Jaundice" if prob > threshold else "Normal"
    
    if prob > threshold:
        conf = prob
    else:
        conf = 1.0 - prob
    
    conf = max(0.0, min(1.0, conf))
    
    # --- NERD MODE STATS ---
    try:
        mean_bgr = np.mean(img_bgr, axis=(0,1))
        mean_rgb = list(mean_bgr[::-1])
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mean_hsv = list(np.mean(hsv_img, axis=(0,1)))
        
        debug_info["color_stats"] = {
            "mean_rgb": [int(max(0, min(255, x))) for x in mean_rgb],
            "mean_hsv": [int(max(0, min(255, x))) for x in mean_hsv]
        }
    except Exception as e:
        print(f"Error calculating color stats: {e}", flush=True)
    
    debug_info["raw_probability"] = float(prob)
    debug_info["threshold_used"] = threshold
    
    return label, conf, debug_info


def predict_skin_disease_torch(img_bgr, debug=False):
    """Predict skin disease with comprehensive error handling."""
    model = get_skin_model()
    if not model: 
        return "Model Missing", 0.0, {"error": "Model not loaded"}
    
    # Validate input
    is_valid, msg = validate_image(img_bgr, "skin image")
    if not is_valid:
        return "Invalid Input", 0.0, {"error": msg}
    
    # Ensure 3-channel
    img_bgr = ensure_3channel(img_bgr)
    
    img_t = _preprocess_img(img_bgr, size=IMG_SIZE)
    if img_t is None:
        return "Preprocessing Failed", 0.0, {"error": "Failed to preprocess image"}
    
    debug_info = {}
    grad_cam = None
    
    try:
        context = torch.enable_grad() if debug else torch.no_grad()
        
        if debug:
            if hasattr(model.backbone, 'conv_head'):
                target_layer = model.backbone.conv_head
            else:
                target_layer = list(model.backbone.children())[-1]
                
            grad_cam = GradCAM(model, target_layer)
        
        with context:
            logits = model(img_t)
            
            # Check for numerical issues
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Warning: NaN/Inf in model output", flush=True)
                return "Error", 0.0, {"error": "Model produced NaN/Inf output"}
            
            probs = torch.softmax(logits, dim=1)
            conf_t, idx_t = torch.max(probs, 1)
            
            idx = idx_t.item()
            conf = conf_t.item()
            
            # Validate outputs
            if not (0 <= idx < len(_skin_classes)):
                print(f"Warning: Class index {idx} out of range [0, {len(_skin_classes)})", flush=True)
                idx = 0
                conf = 0.0
            
            if debug and grad_cam:
                # Backward on the winning class
                score = logits[0, idx]
                model.zero_grad()
                score.backward(retain_graph=False)
                
                heatmap = grad_cam.generate()
                if heatmap is not None:
                     disp_img = cv2.resize(img_bgr, IMG_SIZE)
                     overlay = generate_heatmap_overlay(heatmap, disp_img)
                     debug_info["grad_cam"] = overlay
                     
    except torch.cuda.OutOfMemoryError:
        print("CRITICAL: GPU Out Of Memory in Skin Disease Inference!", flush=True)
        torch.cuda.empty_cache()
        return "GPU_OOM", 0.0, {"status": "oom_error", "error": "GPU out of memory"}
    except Exception as e:
        print(f"Inference Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return "Error", 0.0, {"error": str(e)}
    finally:
        if grad_cam:
            try:
                grad_cam.remove_hooks()
                model.zero_grad()
            except Exception as e:
                print(f"Error cleaning up Grad-CAM: {e}", flush=True)
    
    # Validate confidence
    if not (0.0 <= conf <= 1.0):
        print(f"Warning: Invalid confidence {conf}, clamping", flush=True)
        conf = max(0.0, min(1.0, conf))
    
    # Default to "Normal" if confidence < 80% to reduce false positives
    threshold = CONFIDENCE_THRESHOLD_SKIN_DISEASE
    if conf < threshold:
        label = "Normal"
    else:
        label = _skin_classes.get(idx, f"Unknown Class {idx}")
    
    # --- NERD MODE STATS ---
    try:
        topk_conf, topk_idx = torch.topk(probs, min(3, len(_skin_classes)))
        topk_conf = topk_conf[0].cpu().numpy()
        topk_idx = topk_idx[0].cpu().numpy()
        
        top_3 = []
        for i in range(len(topk_idx)):
            class_name = _skin_classes.get(int(topk_idx[i]), f"Class {topk_idx[i]}")
            score = float(topk_conf[i])
            score = max(0.0, min(1.0, score))  # Clamp
            top_3.append({"label": class_name, "confidence": score})
            
        debug_info["top_3"] = top_3
    except Exception as e:
        print(f"Error calculating top-3: {e}", flush=True)
    
    debug_info["raw_probability"] = float(conf)
    debug_info["threshold_used"] = threshold
    debug_info["class_index"] = int(idx)
    
    return label, conf, debug_info
