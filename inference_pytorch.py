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
            print("🔄 Loading Eye Model (Thread-Safe)...", flush=True)
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
            print(f"⚠️ Skin model files missing: {model_path} or {map_path}", flush=True)
            return None
        
        try:
            print("🔄 Loading Skin Model (Thread-Safe)...", flush=True)
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

# --- GRAD-CAM UTILS ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        # print(f"DEBUG: Registering hooks on {target_layer}", flush=True)
        self.hook_a = target_layer.register_forward_hook(self.save_activation)
        self.hook_g = target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        # print(f"DEBUG: Activation captured. Shape: {output.shape}", flush=True)
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are typically tuple (grad_output,)
        # print(f"DEBUG: Gradient captured. Shape: {grad_output[0].shape}", flush=True)
        self.gradients = grad_output[0]
        
    def generate(self, class_idx=None):
        if self.gradients is None or self.activations is None:
            print("DEBUG: Grad-CAM failed - Missing gradients or activations", flush=True)
            return None
            
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Get activations of the last conv layer
        activations = self.activations.detach()
        
        # Weight the channels by corresponding gradients
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the weighted activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU on top
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)
        
        # Normalize
        heatmap /= np.max(heatmap) + 1e-8
        
        return heatmap
        
    def remove_hooks(self):
        self.hook_a.remove()
        self.hook_g.remove()

def generate_heatmap_overlay(heatmap, img_bgr):
    """Overlays heatmap on original image."""
    try:
        h, w = img_bgr.shape[:2]
        heatmap = cv2.resize(heatmap, (w, h))
        
        # Colorize
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay
        superimposed_img = heatmap * 0.4 + img_bgr * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # Encode
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return ""

def predict_jaundice_eye(skin_img, sclera_crop=None, debug=False):
    model = get_eye_model()
    if not model: return "Model Missing", 0.0, {}
    
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
        
    debug_info = {}
    
    # Context Manager for Gradients
    # If debug=True, we need gradients for GradCAM. Else no_grad for speed.
    context = torch.enable_grad() if debug else torch.no_grad()
    
    grad_cam = None
    if debug:
        # Hook the last conv layer of the Skin Backbone (EfficientNet)
        # timm efficientnet usually has 'conv_head' or 'blocks'
        if hasattr(model.skin_backbone, 'conv_head'):
             target_layer = model.skin_backbone.conv_head
        elif hasattr(model.skin_backbone, 'blocks'):
             target_layer = model.skin_backbone.blocks[-1]
        else:
             print("DEBUG: Could not find target layer for Grad-CAM", flush=True)
             target_layer = list(model.skin_backbone.children())[-1]
             
        grad_cam = GradCAM(model, target_layer)
        
    try:
        with context:
            logits = model(skin_t, sclera_t)
            prob = torch.sigmoid(logits).item()
            
            if debug and grad_cam:
                # Backward for Grad-CAM
                score = logits[0] # Single output
                model.zero_grad()
                score.backward(retain_graph=False)
                
                heatmap = grad_cam.generate()
                if heatmap is not None:
                     # Overlay on Skin Image (Original Resize)
                     overlay = generate_heatmap_overlay(heatmap, skin_resized)
                     debug_info["grad_cam"] = overlay
                     
    except torch.cuda.OutOfMemoryError:
        print("❌ CRITICAL: GPU Out Of Memory in Jaundice Eye Inference!", flush=True)
        return "GPU_OOM", 0.0, {"status": "oom_error"}
    except Exception as e:
        print(f"❌ Inference Error: {e}", flush=True)
        return "Error", 0.0, {}
    finally:
        if grad_cam:
            grad_cam.remove_hooks()
            model.zero_grad() # Cleanup
    
    # Standard threshold after retraining with balanced data
    
    # Standard threshold after retraining with balanced data
    threshold = 0.5
    label = "Jaundice" if prob > threshold else "Normal"
    
    if prob > threshold:
        conf = prob  # Confidence in Jaundice
    else:
        conf = 1.0 - prob  # Confidence in Normal
    
    # --- NERD MODE STATS ---
    # debug_info is already initialized
    
    # 1. Color Stats (Sclera)
    # sclera_crop is BGR. Calculate Mean HSV/RGB
    if sclera_crop is not None and sclera_crop.size > 0:
        # Mean RGB (BGR -> RGB)
        mean_bgr = np.mean(sclera_crop, axis=(0,1))
        mean_rgb = list(mean_bgr[::-1]) # BGR to RGB
        
        # Mean HSV
        hsv_img = cv2.cvtColor(sclera_crop, cv2.COLOR_BGR2HSV)
        mean_hsv = list(np.mean(hsv_img, axis=(0,1)))
        
        debug_info["color_stats"] = {
            "mean_rgb": [int(x) for x in mean_rgb],
            "mean_hsv": [int(x) for x in mean_hsv]
        }
        
    # 2. Sclera Mask (Not available here, as mask generation happens in tasks.py/SegFormer)
    # We will attach the mask in tasks.py
    
    return label, conf, debug_info

def predict_jaundice_body(img_bgr, debug=False):
    model = get_body_model()
    if not model: return "Model Missing", 0.0, {}
    
    img_t = _preprocess_img(img_bgr, size=IMG_SIZE)
    
    debug_info = {}
    context = torch.enable_grad() if debug else torch.no_grad()
    grad_cam = None
    
    if debug:
        # Hook backbone conv_head
        if hasattr(model.backbone, 'conv_head'):
             target_layer = model.backbone.conv_head
        else:
             target_layer = list(model.backbone.children())[-1]
        grad_cam = GradCAM(model, target_layer)
        
    try:
         with context:
            logits = model(img_t)
            prob = torch.sigmoid(logits).item()
            
            if debug and grad_cam:
                score = logits[0]
                model.zero_grad()
                score.backward(retain_graph=False)
                
                heatmap = grad_cam.generate()
                if heatmap is not None:
                     # Resize img_bgr to model input size for overlay consistency
                     # or just overlay on resized inputs.
                     # Let's resize original to match heatmap (380x380) for display
                     disp_img = cv2.resize(img_bgr, IMG_SIZE)
                     overlay = generate_heatmap_overlay(heatmap, disp_img)
                     debug_info["grad_cam"] = overlay
                     
    except torch.cuda.OutOfMemoryError:
        print("❌ CRITICAL: GPU Out Of Memory in Jaundice Body Inference!", flush=True)
        return "GPU_OOM", 0.0, {"status": "oom_error"}
    except Exception as e:
        print(f"❌ Inference Error: {e}", flush=True)
        return "Error", 0.0, {}
    finally:
        if grad_cam:
            grad_cam.remove_hooks()
            model.zero_grad()

    # STRICTER THRESHOLD: Model trained on babies, less reliable on adults
    # Require 70% confidence instead of 50% to reduce false positives
    threshold = 0.70
    label = "Jaundice" if prob > threshold else "Normal"
    
    # Return actual probability as confidence
    if prob > threshold:
        conf = prob  # Confidence in Jaundice prediction
    else:
        conf = 1.0 - prob  # Confidence in Normal prediction
    
    # --- NERD MODE STATS ---
    # debug_info already initialized
    
    # Color Stats (Body Crop) - Calculate average color of the input image
    # Note: img_bgr is likely the cropped skin area (or full frame if body segmentation failed)
    mean_bgr = np.mean(img_bgr, axis=(0,1))
    mean_rgb = list(mean_bgr[::-1])
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = list(np.mean(hsv_img, axis=(0,1)))
    
    debug_info["color_stats"] = {
        "mean_rgb": [int(x) for x in mean_rgb],
        "mean_hsv": [int(x) for x in mean_hsv]
    }
    
    return label, conf, debug_info

def predict_skin_disease_torch(img_bgr, debug=False):
    model = get_skin_model()
    if not model: return "Model Missing", 0.0, {}
    
    img_t = _preprocess_img(img_bgr, size=IMG_SIZE)
    
    debug_info = {}
    context = torch.enable_grad() if debug else torch.no_grad()
    grad_cam = None
    
    if debug:
        if hasattr(model.backbone, 'conv_head'):
            target_layer = model.backbone.conv_head
        else:
            target_layer = list(model.backbone.children())[-1]
            
        grad_cam = GradCAM(model, target_layer)
        
    try:
        with context:
            logits = model(img_t)
            probs = torch.softmax(logits, dim=1)
            conf_t, idx_t = torch.max(probs, 1)
            
            idx = idx_t.item()
            conf = conf_t.item()
            
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
                     
    finally:
        if grad_cam:
            grad_cam.remove_hooks()
            model.zero_grad()
        
    # Default to "Normal" if confidence < 80% to reduce false positives
    if conf < 0.80:
        label = "Normal"
    else:
        label = _skin_classes.get(idx, f"Unknown Class {idx}")
    
    # --- NERD MODE STATS ---
    # debug_info already initialized
    
    # Top-3 Probabilities
    topk_conf, topk_idx = torch.topk(probs, min(3, len(_skin_classes)))
    topk_conf = topk_conf[0].cpu().numpy()
    topk_idx = topk_idx[0].cpu().numpy()
    
    top_3 = []
    for i in range(len(topk_idx)):
        class_name = _skin_classes.get(int(topk_idx[i]), f"Class {topk_idx[i]}")
        score = float(topk_conf[i])
        top_3.append({"label": class_name, "confidence": score})
        
    debug_info["top_3"] = top_3
    
    return label, conf, debug_info
