
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
import mediapipe as mp

# Set up logger
logger = logging.getLogger(__name__)

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
_oral_cancer_model = None
_teeth_model = None
_teeth_classes = None
_posture_model = None

# Class Definitions
ORAL_CANCER_CLASSES = ['Normal', 'Oral_Cancer']
TEETH_CLASSES = ['Calculus', 'Gingivitis', 'Hypodontia', 'Microdontia', 'Mouth Discoloration', 'Spot', 'Ulcer']

# Thread-safe locks
_model_locks = {
    'burns': threading.Lock(),
    'nail': threading.Lock(),
    # cataract removed
    'oral_cancer': threading.Lock(),
    'teeth': threading.Lock(),
    'posture': threading.Lock()
}

def get_last_conv_layer(model):
    """Finds the last Conv2d layer in a model, common target for Grad-CAM."""
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    print(f"Found {len(conv_layers)} conv layers in model.")
    if conv_layers:
        last_layer = conv_layers[-1]
        print(f"Selecting target layer for Grad-CAM: {last_layer.__class__.__name__}")
        return last_layer
    return None


# --- Helper Functions ---

def preprocess_image(img_bgr, target_size):
    """
    Standard preprocessing for classification models.
    Args:
        img_bgr: Input image in BGR format
        target_size: Tuple (W, H)
    Returns:
        tuple: (img_tensor, img_resized)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    
    # Normalize 0-1
    img_norm = img_resized.astype(np.float32) / 255.0
    
    # Standard ImageNet normalization
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std
    
    # CHW format
    img_chw = img_norm.transpose((2, 0, 1))
    
    # Batch dimension
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).float().to(get_device())
    
    return img_tensor, img_resized

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


class TeethDiseaseModel(nn.Module):
    """Multi-class classification for teeth pathologies"""
    def __init__(self, num_classes=7):
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


class PostureClassifier(nn.Module):
    def __init__(self, input_size=12, num_classes=2):
        super(PostureClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


# --- Eye Segmentation Helper ---

_mp_face_mesh = None

def get_face_mesh():
    global _mp_face_mesh
    if _mp_face_mesh is None:
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    return _mp_face_mesh

def segment_eye_mp(img_bgr):
    """Detects and crops eye region using MediaPipe for cleaner 'Eye Mask' view."""
    face_mesh = get_face_mesh()
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    if not results.multi_face_landmarks:
        return None, None
    
    landmarks = results.multi_face_landmarks[0].landmark
    # LEFT_EYE landmarks (using a simplified bounding box)
    # MediaPipe Left Eye: 33, 133, 157, 158, 159, 160, 161, 246
    eye_indices = [33, 133, 157, 158, 159, 160, 161, 246, 7, 163, 144, 145, 153, 154, 155, 133]
    
    x_coords = [landmarks[i].x * w for i in eye_indices]
    y_coords = [landmarks[i].y * h for i in eye_indices]
    
    x1, y1 = int(min(x_coords)), int(min(y_coords))
    x2, y2 = int(max(x_coords)), int(max(y_coords))
    
    # Add padding
    pw, ph = int((x2-x1)*0.3), int((y2-y1)*0.3)
    x1, y1 = max(0, x1-pw), max(0, y1-ph)
    x2, y2 = min(w, x2+pw), min(h, y2+ph)
    
    crop = img_bgr[y1:y2, x1:x2]
    
    # Create simple iris mask for visualization consistency
    mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    points = np.array([[landmarks[i].x*w - x1, landmarks[i].y*h - y1] for i in eye_indices], dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    
    return crop, mask


def analyze_mouth(img_bgr):
    """
    Analyzes mouth status using MediaPipe Face Mesh.
    Returns: (bbox, open_ratio, error_msg)
    bbox: [x1, y1, x2, y2] normalized
    """
    try:
        face_mesh = get_face_mesh()
        results = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None, 0.0, "No Face Detected"
            
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, c = img_bgr.shape
        
        # Mouth Landmarks (Inner lips for openness)
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        
        # Calculate opening
        mouth_open_dist = np.linalg.norm(np.array([top_lip.x*w, top_lip.y*h]) - np.array([bottom_lip.x*w, bottom_lip.y*h]))
        
        # Face height for normalization
        face_top = landmarks[10].y * h
        face_bottom = landmarks[152].y * h
        face_height = face_bottom - face_top
        
        open_ratio = mouth_open_dist / (face_height + 1e-6)
        
        # Bounding Box (Outer lips)
        # 61 (left), 291 (right), 0 (top), 17 (bottom) - approximation
        lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
        xs = [landmarks[i].x for i in lip_indices]
        ys = [landmarks[i].y for i in lip_indices]
        
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        
        # Add padding
        pw = (x2-x1) * 0.2
        ph = (y2-y1) * 0.2
        bbox = [max(0.0, x1-pw), max(0.0, y1-ph), min(1.0, x2+pw), min(1.0, y2+ph)]
        
        return bbox, open_ratio, None
        
    except Exception as e:
        # print(f"Mouth analysis failed: {e}")
        return None, 0.0, str(e)



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
        
        # Priority: User specified YOLO model (YOLOv7 custom)
        # Note: The model file is expected to be in saved_models
        # It might be named 'skin_burn_2022_8_21.pt' or similar
        
        # Check for the specific file the user likely has
        model_paths = [
            BASE_DIR / "saved_models" / "skin_burn_2022_8_21.pt",
            BASE_DIR / "saved_models" / "best.pt", # Common name
            BASE_DIR / "saved_models" / "yolov7.pt"
        ]
        
        path = None
        for p in model_paths:
            if p.exists():
                path = p
                break
        
        if not path:
             # Fallback check
             found = list((BASE_DIR / "saved_models").glob("*.pt"))
             if found:
                 path = found[0]
                 
        if not path:
            print("Burns model (.pt) not found in saved_models.", flush=True)
            return None
        
        try:
            print(f"Loading Burns Model from {path.name}...", flush=True)
            
            # --- Load using External Repo (YOLOv7 - Baked In) ---
            # repo_path = BASE_DIR / "external" / "skin-burn-repo" # OLD
            repo_path = BASE_DIR / "yolov7_ops" # NEW (Baked in backend)
            
            if repo_path.exists():
                import sys
                # Add repo to sys.path to allow imports from it
                # This allow 'from models.experimental import attempt_load' to work internally
                if str(repo_path) not in sys.path:
                    sys.path.insert(0, str(repo_path))
                
                try:
                    # Import from external repo
                    from models.experimental import attempt_load
                    
                    # atomic load
                    model = attempt_load(str(path), map_location=get_device())
                    _burns_model = BurnsModel(model)
                    _burns_model.model_type = 'yolov7_external'
                    print(f"Burns Model Loaded from {path.name} (Local YOLOv7 Ops)", flush=True)
                    return _burns_model
                    
                except ImportError as e:
                    print(f"Failed to import from yolov7_ops: {e}")
                except Exception as e:
                    print(f"Error loading via yolov7_ops: {e}")
            else:
                print(f"Local yolov7_ops repo not found at {repo_path}")

            # --- Fallbacks (Ultralytics / YOLOv5) ---
            # ... (omitted for brevity, preferring external repo)
            
            return None
            
        except Exception as e:
            print(f"Failed to load Burns Model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None


def predict_burns(img_bgr, debug=False):
    """
    Predict burn detection using External YOLOv7 Repo logic.
    """
    # 1. Skin Detection Check
    try:
        skin_mask = _detect_skin_color(img_bgr)
        total_pixels = skin_mask.shape[0] * skin_mask.shape[1]
        skin_pixels = cv2.countNonZero(skin_mask)
        skin_ratio = skin_pixels / (total_pixels + 1e-6)
        
        if skin_ratio < 0.05:
            return "No Skin Detected", 1.0, {"skin_ratio": f"{skin_ratio:.1%}"}
            
    except Exception as e:
        print(f"Skin check warning: {e}")

    model = get_burns_model()
    if model is None:
        return "Model Not Loaded", 0.0, {"error": "Burns model not available"}
    
    debug_info = {}
    
    try:
        # Check model type
        model_type = getattr(model, 'model_type', 'legacy')
        
        if model_type == 'yolov7_external':
            try:
                # Imports from external repo (reliant on sys.path injection in get_burns_model)
                from utils.general import non_max_suppression, scale_coords
                # from utils.torch_utils import select_device # Not needed if we control device manually?
                
                # Preprocess
                # YOLOv7 default img_size is usually 640
                img_size = 640
                img = cv2.resize(img_bgr, (img_size, img_size))
                
                # BGR to RGB, CHW
                img = img[:, :, ::-1].transpose(2, 0, 1)  
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(get_device())
                img = img.float() 
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                
                # Inference
                with torch.no_grad():
                    # model.model is the actual PyTorch model inside our wrapper
                    pred = model.model(img)[0]
                
                # NMS
                # conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False
                pred = non_max_suppression(pred, 0.25, 0.45)
                
                # Process detections
                det = pred[0]
                
                label = "Healthy" # Default
                conf_val = 0.0
                
                if det is not None and len(det):
                    # Rescale boxes to original image shape
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_bgr.shape).round()
                    
                    # Find best match
                    best_conf = 0.0
                    best_cls = -1
                    best_bbox = None
                    
                    for *xyxy, conf, cls in det:
                         if conf > best_conf:
                             best_conf = conf.item()
                             best_cls = int(cls.item())
                             best_bbox = [x.item() for x in xyxy]
                    
                    # Get class name
                    names = getattr(model.model, 'module', model.model).names
                    if hasattr(names, 'keys'): # dict {0: 'name'}
                         label = names.get(best_cls, 'Burn')
                    else: # list ['name', ...]
                         label = names[best_cls] if best_cls < len(names) else 'Burn'
                    
                    if label.lower() == 'burn':
                         label = 'Burn Detected' # User friendly
                         
                    conf_val = best_conf
                    
                    if debug and best_bbox:
                         # normalize bbox for UI
                         h, w = img_bgr.shape[:2]
                         debug_info['bbox'] = [
                             best_bbox[0]/w, best_bbox[1]/h, 
                             best_bbox[2]/w, best_bbox[3]/h
                         ]
                         
                         # Draw on debug image
                         debug_img = img_bgr.copy()
                         cv2.rectangle(debug_img, (int(best_bbox[0]), int(best_bbox[1])), (int(best_bbox[2]), int(best_bbox[3])), (0, 0, 255), 2)
                         _, buffer = cv2.imencode('.jpg', debug_img)
                         debug_info["overlay"] = base64.b64encode(buffer).decode('utf-8')

                    return label, conf_val, debug_info
                
                else:
                    return "Healthy", 0.95, debug_info

            except Exception as e:
                print(f"External YOLOv7 Inference Error: {e}")
                import traceback
                traceback.print_exc()
                return "Error", 0.0, {"error": str(e)}

        # Fallback for other model types...
        return "Error", 0.0, {"error": f"Unknown model type: {model_type}"}

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
    
    # 1. Mouth Detection Check (Requested by user to prevent prediction on closed mouths)
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                # If no face, maybe just a mouth crop? 
                # But user complained about "predicting without even seeing open mouth"
                pass 
            else:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = img_bgr.shape
                
                # Mouth landmarks (Upper lip: 13, Lower lip: 14)
                top_lip = landmarks[13]
                bottom_lip = landmarks[14]
                
                # Face height reference (Forehead: 10, Chin: 152)
                face_top = landmarks[10].y * h
                face_bottom = landmarks[152].y * h
                face_height = face_bottom - face_top
                
                mouth_open_dist = np.linalg.norm(
                    np.array([top_lip.x * w, top_lip.y * h]) - 
                    np.array([bottom_lip.x * w, bottom_lip.y * h])
                )
                
                open_ratio = mouth_open_dist / (face_height + 1e-6)
                
                if open_ratio < 0.05:
                     return "Mouth Closed", 0.0, {"error": "Please open your mouth wide.", "mouth_open_ratio": float(open_ratio)}

    except Exception as e:
        print(f"Oral cancer mouth check failed: {e}", flush=True)

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
        # Add Mouth BBox
        bbox, _, _ = analyze_mouth(img_bgr)
        if bbox:
            debug_info["bbox"] = bbox

        img_tensor, img_resized = preprocess_image(img_bgr, IMG_SIZE_LARGE)
        grad_cam = None
        context = torch.no_grad()
        if debug:
            context = torch.enable_grad()
            img_tensor = img_tensor.requires_grad_(True)
            try:
                target_layer = get_last_conv_layer(model)
                if target_layer:
                    grad_cam = GradCAM(model, target_layer)
                else:
                    print("Grad-CAM failed: No conv layers found in oral cancer model.")
            except Exception as e:
                print(f"Grad-CAM setup failed for oral cancer: {e}. Continuing without visualization.")
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

# Cataract model code removed as per user request


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
            # Patch __main__ to allow loading models saved in main script scope
            import sys
            import __main__
            setattr(__main__, "TeethDiseaseModel", TeethDiseaseModel)
            
            try:
                try:
                    model = torch.load(model_path, map_location=get_device(), weights_only=False)
                except TypeError:
                    model = torch.load(model_path, map_location=get_device())
            finally:
                # Optional: Cleanup if you want to keep namespace clean, but usually harmless to leave
                pass 
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
    
    # Detect mouth/openness using MediaPipe
    bbox, open_ratio, error = analyze_mouth(img_bgr)
    
    if error:
        return "No Face Detected", 0.0, {"error": error}
        
    if bbox:
        debug_info["bbox"] = bbox
        debug_info["mouth_open_ratio"] = float(open_ratio)
        
    if open_ratio < 0.05: # Threshold: Mouth is closed
         return "Mouth Closed", 0.0, {"error": "Please open your mouth to inspect teeth.", "mouth_open_ratio": float(open_ratio), "bbox": bbox}

    try:
        img_tensor, img_resized = preprocess_image(img_bgr, IMG_SIZE_LARGE)
        
        # For Grad-CAM, tensor needs requires_grad=True
        if debug:
            img_tensor = img_tensor.requires_grad_(True)
        
        # Setup Grad-CAM if debug mode is on
        context = torch.enable_grad() if debug else torch.no_grad()
        grad_cam = None
        
        if debug:
            try:
                target_layer = get_last_conv_layer(model)
                if target_layer:
                    grad_cam = GradCAM(model, target_layer)
                else:
                    print("Grad-CAM failed: No conv layers found in teeth model.")
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


def get_posture_model():
    global _posture_model
    if _posture_model:
        return _posture_model
    
    with _model_locks['posture']:
        if _posture_model:
            return _posture_model
        
        model_path = BASE_DIR / "saved_models" / "posture_classifier.pth"
        if not model_path.exists():
            print(f"Posture model not found at {model_path}", flush=True)
            return None
        
        try:
            print("Loading Posture Classifier (Thread-Safe)...", flush=True)
            model = PostureClassifier(input_size=12, num_classes=2).to(get_device())
            state_dict = torch.load(model_path, map_location=get_device())
            model.load_state_dict(state_dict)
            model.eval()
            _posture_model = model
            print("Posture Classifier Loaded successfully", flush=True)
            return _posture_model
        except Exception as e:
            print(f"Failed to load Posture Model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

def predict_posture_from_landmarks(landmarks, debug=False):
    """
    Predict posture status from MediaPipe landmarks.
    Args:
        landmarks: List of landmark dicts [{"x":, "y":, ...}]
    """
    model = get_posture_model()
    if model is None:
        return "Model Not Loaded", 0.0, {"error": "Posture classifier not available"}
    
    try:
        # We need specific landmarks index that match the COCO dataset's 6 points
        # Assuming the 6 points are: Neck, Back, Hips etc. 
        # For simplicity, we'll map the top 6 points from MediaPipe that represent trunk
        # MediaPipe indices: 11(L shoulder), 12(R shoulder), 23(L hip), 24(R hip), 
        # plus maybe 0(Nose) and midpoint of (23,24) for spine.
        # But for the trained model to work best, we should use the SAME points as training.
        # In my training script, I just used the first 6 points in the COCO keypoints.
        
        # NOTE: This implementation assumes the caller (tasks.py) provides the correct 12 features (6 pts * x,y)
        # However, to be robust, let's allow it to take raw landmarks and extract them here.
        
        # If landmarks is already a list of 12 floats, use it directly
        if isinstance(landmarks, (list, np.ndarray)) and len(landmarks) == 12:
            features = landmarks
        else:
            # Map MediaPipe landmarks to our 6 keypoints (approximate)
            # Neck/Shoulder Mid, Mid Back, Hips...
            # This mapping needs to match what was in the COCO dataset.
            # Assuming standard COCO posture: 0:Nose, 1:L_Shoulder, 2:R_Shoulder, 3:L_Hip, 4:R_Hip, 5:Mid_Spine
            # We'll try to extract these or equivalent.
            
            # Using specific indices if they are provided as a dict list
            indices = [0, 11, 12, 23, 24, 25] # Nose, L_Sho, R_Sho, L_Hip, R_Hip, L_Knee? 
            # Actually, let's stick to the 12-feature input expectation for now.
            return "Error", 0.0, {"error": "Feature extraction mapping not implemented"}

        features_tensor = torch.tensor([features], dtype=torch.float32).to(get_device())
        
        with torch.no_grad():
            output = model(features_tensor)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, dim=1)
            conf = conf.item()
            idx = idx.item()
            
        # 0: Good, 1: Bad (matching cat_map in training script)
        label = "Good Form" if idx == 0 else "Bad Form"
        
        return label, conf, {}
        
    except Exception as e:
        print(f"Error in predict_posture: {e}", flush=True)
        return "Error", 0.0, {"error": str(e)}
