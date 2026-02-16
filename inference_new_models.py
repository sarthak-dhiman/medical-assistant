
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
_cataract_model = None
_oral_cancer_model = None
_teeth_model = None
_teeth_classes = None

# Thread-safe locks
_model_locks = {
    'burns': threading.Lock(),
    'nail': threading.Lock(),
    'cataract': threading.Lock(),
    'oral_cancer': threading.Lock(),
    'teeth': threading.Lock()
}


# --- Model Architectures ---

class BurnsModel(nn.Module):
    """Wrapper for YOLO-based Burns detection model"""
    def __init__(self, yolo_model=None):
        super().__init__()
        self.model = yolo_model
    
    def forward(self, x):
        # YOLO models expect different input format
        # For now, we'll use the model's predict method in inference
        if self.model is not None:
            return self.model(x)
        return None





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


# CataractModel wrapper removed - model is saved as complete timm object
# The training script saves the unwrapped model directly, so we load it as-is





# --- Model Loading Functions (Thread-Safe) ---

def get_burns_model():
    global _burns_model
    if _burns_model:
        return _burns_model
    
    with _model_locks['burns']:
        if _burns_model:
            return _burns_model
        
        # Priority 1: User specified YOLO model (YOLOv7 custom)
        path = BASE_DIR / "saved_models" / "skin_burn_2022_8_21.pt"
        is_yolov7 = True
        
        if not path.exists():
            # Priority 2: Standard/Renamed model (YOLOv8/v5)
            path = BASE_DIR / "saved_models" / "burns_pytorch.pt"
            is_yolov7 = False
            
        if not path.exists():
            return None
        
        try:
            print(f"Loading Burns Model from {path.name}...", flush=True)
            
            # 1. Try Local YOLOv7 Repo (for skin_burn_2022_8_21.pt)
            if is_yolov7 or path.name == "skin_burn_2022_8_21.pt":
                repo_path = BASE_DIR / "external" / "Skin-Burn-Detection-Classification"
                if repo_path.exists():
                    import sys
                    if str(repo_path) not in sys.path:
                        sys.path.insert(0, str(repo_path))
                    
                    try:
                        from models.experimental import attempt_load
                        model = attempt_load(str(path), map_location=get_device())
                        _burns_model = BurnsModel(model)
                        _burns_model.model_type = 'yolov7'
                        print(f"Burns Model Loaded from {path.name} (YOLOv7 Native)", flush=True)
                        return _burns_model
                    except ImportError as e:
                        print(f"Failed to load YOLOv7 native: {e}")
                    except Exception as e:
                        print(f"Error loading YOLOv7: {e}")
            
            # 2. Try Ultralytics (YOLOv8/v5 fallback)
            try:
                from ultralytics import YOLO
                model = YOLO(str(path))
                _burns_model = model
                # Check if it has 'model' attribute to distinguish
                if not hasattr(_burns_model, 'model_type'):
                     _burns_model.model_type = 'ultralytics'
                print(f"Burns Model Loaded from {path.name} (ultralytics)", flush=True)
                return _burns_model
            except ImportError:
                print("Ultralytics not available, loading checkpoint directly...", flush=True)
            except Exception as e:
                 print(f"Ultralytics load failed: {e}")

            # 3. Direct Checkpoint Load (Legacy/Raw)
            checkpoint = torch.load(path, map_location=get_device())
            
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                     yolo_model = checkpoint['model']
                     _burns_model = BurnsModel(yolo_model)
                     _burns_model.model_type = 'yolov7' # Assume raw model dict is yolo
                     print("Burns Model Loaded (from checkpoint 'model' key)", flush=True)
                     return _burns_model
                elif 'ema' in checkpoint:
                     yolo_model = checkpoint['ema']
                     _burns_model = BurnsModel(yolo_model)
                     _burns_model.model_type = 'yolov7'
                     print("Burns Model Loaded (from checkpoint 'ema' key)", flush=True)
                     return _burns_model

            # Direct model object
            elif hasattr(checkpoint, 'eval'):
                 _burns_model = BurnsModel(checkpoint)
                 _burns_model.model_type = 'yolov7'
                 print("Burns Model Loaded (direct model object)", flush=True)
                 return _burns_model
            
            print("Warning: Could not load YOLO model from checkpoint.", flush=True)
            return None
            
        except Exception as e:
            print(f"Failed to load Burns Model: {e}", flush=True)
            import traceback
            traceback.print_exc()
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
        # Check model type
        model_type = getattr(model, 'model_type', 'legacy')
        
        # 1. Ultralytics YOLO (v8/v5)
        if model_type == 'ultralytics' or (hasattr(model, 'model') and not isinstance(model, BurnsModel)):
             # YOLO Inference
            try:
                results = model.predict(img_bgr, verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    conf = float(boxes.conf[0].item())
                    cls = int(boxes.cls[0].item())
                    label = results[0].names[cls]
                    
                    if debug:
                        annotated_img = results[0].plot()
                        _, buffer = cv2.imencode('.jpg', annotated_img)
                        debug_info["overlay"] = base64.b64encode(buffer).decode('utf-8')
                    
                    return label, conf, debug_info
                else:
                    return "Healthy", 0.95, debug_info
            except Exception as e:
                print(f"Ultralytics Inference Error: {e}")

        # 2. Native YOLOv7 (Custom Repo)
        elif model_type == 'yolov7':
            try:
                from utils.general import non_max_suppression, scale_coords
                
                # Preprocess for YOLOv7 (usually needs 640x640, BGR->RGB, CHW)
                # But expects 0-255 or 0-1? attempt_load models usually want 0-1 and normalized?
                # YOLOv7 logic: img /= 255.0.
                
                img_size = 640
                img = cv2.resize(img_bgr, (img_size, img_size))
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(get_device())
                img = img.float() 
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                
                # Inference
                with torch.no_grad():
                    pred = model.model(img)[0]
                
                # NMS
                pred = non_max_suppression(pred, 0.25, 0.45)
                
                # Process detections
                det = pred[0]
                if det is not None and len(det):
                    # Rescale boxes from img_size to original image size
                    # scale_coords(img.shape[2:], det[:, :4], img_bgr.shape).round()
                    
                    # We typically want best confidence
                    # Columns: x1, y1, x2, y2, conf, cls
                    best_conf = 0.0
                    best_cls = -1
                    
                    for *xyxy, conf, cls in det:
                         if conf > best_conf:
                             best_conf = conf.item()
                             best_cls = int(cls.item())
                    
                    # Map class to label
                    # Check model.names if available
                    names = getattr(model.model, 'names', ['Burn'])
                    if hasattr(names, 'keys'): # dict
                         label = names.get(best_cls, 'Burn')
                    else: # list
                         label = names[best_cls] if best_cls < len(names) else 'Burn'
                         
                    # Debug overlay
                    if debug:
                         # Draw boxes (simplified)
                         debug_img = img_bgr.copy()
                         for *xyxy, conf, cls in det:
                             c = int(cls)
                             label_viz = f'{names[c] if isinstance(names, list) else names.get(c, "Burn")} {conf:.2f}'
                             p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                             # Scaling needed? No, we didn't rescale back to original img yet?
                             # Actually we need rescale if we want to draw on original
                             # For simplicity, let's just draw on resized input or skip overlay for now
                             # Or use the img passed to model
                             pass
                         pass

                    return label, best_conf, debug_info
                else:
                    return "Healthy", 0.95, debug_info

            except Exception as e:
                print(f"YOLOv7 Inference Error: {e}")
                import traceback
                traceback.print_exc()

        # 3. Legacy / Fallback
        # ... (Existing legacy code handles here if needed, but assuming yolo logic covers it)
        # If we reached here, inference failed or mode unknown.
        
        return "Error", 0.0, {"error": "Inference failed or unknown model type"}

    except Exception as e:
        print(f"Error in predict_burns: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return "Error", 0.0, {"error": str(e)}



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
        
        # For Grad-CAM, tensor needs requires_grad=True
        if debug:
            img_tensor = img_tensor.requires_grad_(True)
        
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
        topk_conf = topk_conf[0].detach().cpu().numpy()
        topk_idx = topk_idx[0].detach().cpu().numpy()
        
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
        import traceback
        print(f"Error in predict_nail_disease: {e}", flush=True)
        traceback.print_exc()
        return "Error", 0.0, {"error": str(e)}


# --- Cataract Detection ---

CATARACT_CLASSES = ['Cataract', 'Normal']

# Oral Cancer classes (binary)
ORAL_CANCER_CLASSES = ['Oral_Cancer', 'Normal']

# Teeth Disease classes (7 classes)
TEETH_CLASSES = ['Calculus', 'Caries', 'Gingivitis', 'Healthy', 'Hypodontia', 'Mouth Ulcer', 'Tooth Discoloration']

def get_oral_cancer_model():
    global _oral_cancer_model
    if _oral_cancer_model:
        return _oral_cancer_model
    
    with _model_locks['oral_cancer']:
        if _oral_cancer_model:
            return _oral_cancer_model
        
        model_path = BASE_DIR / "saved_models" / "oral_cancer_model.pth"
        if not model_path.exists():
            # Fallback: Recursive search
            print(f"Oral cancer model not found at {model_path}, searching...", flush=True)
            found = list(BASE_DIR.rglob("oral_cancer_model.pth"))
            if found:
                model_path = found[0]
                print(f"Found Oral Cancer model at {model_path}", flush=True)
            else:
                print(f"Oral cancer model file not found in {BASE_DIR}", flush=True)
                return None
        try:
            print("Loading Oral Cancer Model (Thread-Safe)...", flush=True)
            # Load complete model object (saved by updated training script)
            # PyTorch 2.6+ requires weights_only=False for full model objects (pickle)
            try:
                model = torch.load(model_path, map_location=get_device(), weights_only=False)
            except TypeError:
                 # Fallback for older PyTorch versions that don't support weights_only arg
                model = torch.load(model_path, map_location=get_device())
            model.eval()
            _oral_cancer_model = model
            print(f"Oral Cancer Model Loaded - 2 classes (Oral_Cancer/Normal)", flush=True)
            return _oral_cancer_model
        except Exception as e:
            print(f"Failed to load Oral Cancer Model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None


def predict_oral_cancer(img_bgr, debug=False):
    """
    Predict oral cancer from oral cavity image.
    Returns: (label, confidence, debug_info)
    """
    model = get_oral_cancer_model()
    if model is None:
        # Debugging: List available files
        try:
            saved_models_path = BASE_DIR / "saved_models"
            if saved_models_path.exists():
                files = [f.name for f in saved_models_path.glob("*.pth")]
                debug_msg = f"Models in {saved_models_path}: {files}"
            else:
                debug_msg = f"{saved_models_path} does not exist"
        except Exception as e:
            debug_msg = f"List dir failed: {e}"
            
        return "Model Not Loaded", 0.0, {"error": f"Oral cancer model not available. Debug: {debug_msg}"}
    
    debug_info = {}
    try:
        img_tensor, img_resized = preprocess_image(img_bgr, IMG_SIZE_LARGE)
        grad_cam = None
        context = torch.no_grad()
        if debug:
            context = torch.enable_grad()
            # Model is unwrapped timm EfficientNet - access layers directly
            try:
                if hasattr(model, 'conv_head'):
                    target_layer = model.conv_head
                elif hasattr(model, 'blocks'):
                    target_layer = model.blocks[-1]
                else:
                    # Fallback: get last convolutional layer
                    target_layer = list(model.children())[-2]
                grad_cam = GradCAM(model, target_layer)
            except Exception as e:
                logger.warning(f"Grad-CAM setup failed for oral cancer: {e}. Continuing without visualization.")
                grad_cam = None

        with context:
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, dim=1)
            conf = conf.item()
            idx = idx.item()

            if debug and grad_cam:
                score = output[0, idx]
                model.zero_grad()
                score.backward(retain_graph=False)
                heatmap = grad_cam.generate()
                if heatmap is not None:
                    overlay = generate_heatmap_overlay(heatmap, img_resized)
                    debug_info["grad_cam"] = overlay

        label = ORAL_CANCER_CLASSES[idx] if idx < len(ORAL_CANCER_CLASSES) else f"Unknown Class {idx}"

        # Top-2 probabilities
        top2 = []
        probs_np = probs[0].detach().cpu().numpy()
        for i in range(len(ORAL_CANCER_CLASSES)):
            top2.append({"label": ORAL_CANCER_CLASSES[i], "confidence": float(probs_np[i])})

        debug_info["top_3"] = top2

        if debug:
            _, buffer = cv2.imencode('.jpg', img_resized)
            debug_info["debug_image"] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        if grad_cam:
            grad_cam.remove_hooks()
            model.zero_grad()

        return label, conf, debug_info
    except Exception as e:
        import traceback
        print(f"Error in predict_oral_cancer: {e}", flush=True)
        traceback.print_exc()
        return "Error", 0.0, {"error": str(e)}

def get_cataract_model():
    global _cataract_model
    if _cataract_model:
        return _cataract_model
    
    with _model_locks['cataract']:
        if _cataract_model:
            return _cataract_model
        
        model_path = BASE_DIR / "cataract_model.pth"
        
        # Also check saved_models folder
        if not model_path.exists():
            model_path = BASE_DIR / "saved_models" / "cataract_model.pth"
        
        if not model_path.exists():
            # Fallback: Recursive search
            print(f"Cataract model not found, searching...", flush=True)
            found = list(BASE_DIR.rglob("cataract_model.pth"))
            if found:
                model_path = found[0]
                print(f"Found Cataract model at {model_path}", flush=True)
            else:
                print(f"Cataract model file not found in {BASE_DIR}", flush=True)
                return None
        
        try:
            print("Loading Cataract Model (Thread-Safe)...", flush=True)
            # Try to load as complete model object first
            try:
                # PyTorch 2.6+ requires weights_only=False for full model objects
                try:
                    model = torch.load(model_path, map_location=get_device(), weights_only=False)
                except TypeError:
                    model = torch.load(model_path, map_location=get_device())
                
                # Check if it's a state_dict (OrderedDict)
                if isinstance(model, dict):
                    print("Cataract model loaded as state_dict, identifying architecture...", flush=True)
                    keys = list(model.keys())
                    
                    is_torchvision_efficientnet = any('features.8' in k for k in keys)
                    is_mobilenet_v2 = any('classifier.1.weight' in k for k in keys) or any('classifier.1.1.weight' in k for k in keys) # Classifier usually has 2 submodules (Dropout, Linear)
                    
                    if is_torchvision_efficientnet:
                        print("Identified Torchvision EfficientNet-B0 (likely) architecture.", flush=True)
                        from torchvision import models
                        
                        # Fix keys if classifier.1.1 is present (legacy mapping issue)
                        new_state_dict = {}
                        for k, v in model.items():
                            new_k = k
                            if 'classifier.1.1' in k:
                                new_k = k.replace('classifier.1.1', 'classifier.1')
                            new_state_dict[new_k] = v
                        model = new_state_dict

                        # Try EfficientNet-B0 (1280 features)
                        try:
                            model_arch = models.efficientnet_b0(pretrained=False)
                            model_arch.classifier[1] = nn.Linear(1280, 2)
                            model_arch.to(get_device())
                            model_arch.load_state_dict(model)
                            model = model_arch
                            print("Successfully loaded as EfficientNet-B0", flush=True)
                        except Exception as e_b0:
                            print(f"Failed B0 load: {e_b0}. Trying MobileNetV2...", flush=True)
                            raise e_b0

                    elif is_mobilenet_v2: 
                        print("Identified MobileNetV2 architecture.", flush=True)
                        from torchvision import models
                        model_arch = models.mobilenet_v2(pretrained=False)
                        # Replace classifier to match 2 classes
                        model_arch.classifier[1] = nn.Linear(model_arch.last_channel, 2)
                        model_arch.to(get_device())
                        model_arch.load_state_dict(model)
                        model = model_arch

                    else:
                        # Assume EfficientNet-B4 (from current training script)
                        print("Assuming EfficientNet-B4 architecture.", flush=True)
                        import timm
                        model_arch = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
                        model_arch.to(get_device())
                        model_arch.load_state_dict(model)
                        model = model_arch
                
                model.eval()
                _cataract_model = model
                print(f"Cataract Model Loaded - 2 classes (Cataract/Normal)", flush=True)
                return _cataract_model
            except Exception as load_err:
                print(f"Direct load failed ({load_err}), trying fallback...", flush=True)

                return None

        except Exception as e:
            print(f"Failed to load Cataract Model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None


def predict_cataract(img_bgr, debug=False):
    """
    Predict cataract from eye image.
    Returns: (label, confidence, debug_info)
    """
    model = get_cataract_model()
    if model is None:
        # Debugging: List available files
        try:
            saved_models_path = BASE_DIR / "saved_models"
            if saved_models_path.exists():
                files = [f.name for f in saved_models_path.glob("*.pth")]
                debug_msg = f"Models in {saved_models_path}: {files}"
            else:
                debug_msg = f"{saved_models_path} does not exist"
        except Exception as e:
            debug_msg = f"List dir failed: {e}"
            
        return "Model Not Loaded", 0.0, {"error": f"Cataract model not available. Debug: {debug_msg}"}
    
    debug_info = {}
    
    try:
        # Preprocess
        img_tensor, img_resized = preprocess_image(img_bgr, IMG_SIZE_LARGE)
        
        # Grad-CAM setup
        grad_cam = None
        context = torch.no_grad()
        
        if debug:
            context = torch.enable_grad()
            # Model is unwrapped timm EfficientNet - access layers directly
            try:
                if hasattr(model, 'conv_head'):
                    target_layer = model.conv_head
                elif hasattr(model, 'blocks'):
                    target_layer = model.blocks[-1]
                elif hasattr(model, 'features'):
                     target_layer = model.features[-1]
                else:
                    # Fallback: get last convolutional layer
                    target_layer = list(model.children())[-2]
                grad_cam = GradCAM(model, target_layer)
            except Exception as e:
                logger.warning(f"Grad-CAM setup failed for cataract: {e}. Continuing without visualization.")
                grad_cam = None
        
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
        
        label = CATARACT_CLASSES[idx] if idx < len(CATARACT_CLASSES) else f"Unknown Class {idx}"
        
        # Top-2 predictions (only 2 classes)
        top2 = []
        probs_np = probs[0].detach().cpu().numpy()
        for i in range(len(CATARACT_CLASSES)):
            top2.append({"label": CATARACT_CLASSES[i], "confidence": float(probs_np[i])})
        
        debug_info["top_3"] = top2  # Keep key name for frontend compatibility
        
        # Add Debug Image
        if debug:
            _, buffer = cv2.imencode('.jpg', img_resized)
            debug_info["debug_image"] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        
        if grad_cam:
            grad_cam.remove_hooks()
            model.zero_grad()
        
        return label, conf, debug_info
    
    except Exception as e:
        import traceback
        print(f"Error in predict_cataract: {e}", flush=True)
        traceback.print_exc()
        return "Error", 0.0, {"error": str(e)}


# --- Teeth Disease Detection ---

def get_teeth_model():
    global _teeth_model, _teeth_classes
    if _teeth_model:
        return _teeth_model
    
    with _model_locks['teeth']:
        if _teeth_model:
            return _teeth_model
        
        model_path = BASE_DIR / "saved_models" / "teeth_model.pth"
        map_path = BASE_DIR / "saved_models" / "teeth_disease_mapping.json"
        
        if not model_path.exists():
            print(f"Teeth model file not found at {model_path}", flush=True)
            return None
        
        try:
            print("Loading Teeth Disease Model (Thread-Safe)...", flush=True)
            
            # Load complete model object (saved by training script)
            try:
                model = torch.load(model_path, map_location=get_device(), weights_only=False)
            except TypeError:
                model = torch.load(model_path, map_location=get_device())
            model.eval()
            _teeth_model = model
            
            # Load class mapping if available
            if map_path.exists():
                with open(map_path, 'r') as f:
                    raw_map = json.load(f)
                # Invert mapping (class_name -> idx to idx -> class_name)
                _teeth_classes = {v: k for k, v in raw_map.items()}
            else:
                # Fallback to default classes
                _teeth_classes = {i: TEETH_CLASSES[i] for i in range(len(TEETH_CLASSES))}
            
            print(f"Teeth Disease Model Loaded - {len(_teeth_classes)} classes", flush=True)
            return _teeth_model
        except Exception as e:
            print(f"Failed to load Teeth Model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None


def predict_teeth_disease(img_bgr, debug=False):
    """
    Predict teeth disease from oral/teeth image.
    Returns: (label, confidence, debug_info)
    """
    model = get_teeth_model()
    if model is None or _teeth_classes is None:
        return "Model Not Loaded", 0.0, {"error": "Teeth disease model not available"}
    
    debug_info = {}
    
    try:
        img_tensor, img_resized = preprocess_image(img_bgr, IMG_SIZE_LARGE)
        
        # For Grad-CAM, tensor needs requires_grad=True
        if debug:
            img_tensor = img_tensor.requires_grad_(True)
        
        # Setup Grad-CAM if debug mode is on
        context = torch.enable_grad() if debug else torch.no_grad()
        grad_cam = None
        
        if debug:
            # Model is a custom class with backbone attribute
            try:
                if hasattr(model, 'backbone'):
                    if hasattr(model.backbone, 'conv_head'):
                        target_layer = model.backbone.conv_head
                    elif hasattr(model.backbone, 'blocks'):
                        target_layer = model.backbone.blocks[-1]
                    else:
                        target_layer = list(model.backbone.children())[-1]
                else:
                    # Fallback for unwrapped models
                    if hasattr(model, 'conv_head'):
                        target_layer = model.conv_head
                    elif hasattr(model, 'blocks'):
                        target_layer = model.blocks[-1]
                    else:
                        target_layer = list(model.children())[-2]
                grad_cam = GradCAM(model, target_layer)
            except Exception as e:
                print(f"Grad-CAM setup failed for teeth: {e}. Continuing without visualization.", flush=True)
                grad_cam = None

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
            
        label = _teeth_classes.get(idx, f"Unknown Class {idx}")
        
        # Top-3 predictions
        topk_conf, topk_idx = torch.topk(probs, min(3, len(_teeth_classes)))
        topk_conf = topk_conf[0].detach().cpu().numpy()
        topk_idx = topk_idx[0].detach().cpu().numpy()
        
        top3 = []
        for i in range(len(topk_idx)):
            class_name = _teeth_classes.get(int(topk_idx[i]), f"Class {topk_idx[i]}")
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
        import traceback
        print(f"Error in predict_teeth_disease: {e}", flush=True)
        traceback.print_exc()
        return "Error", 0.0, {"error": str(e)}
