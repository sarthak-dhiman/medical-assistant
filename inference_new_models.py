"""
inference_new_models.py  —  ONNX Runtime Edition
==================================================
Inference for Burns (YOLO), Nail Disease, Oral Cancer,
Teeth Disease, and Posture models.

EfficientNet classifiers (Nails, Oral Cancer, Teeth)  → ONNX Runtime
Posture classifier (linear MLP)                        → ONNX Runtime
"""

import logging
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import json
import threading
import base64
import sys

# Lazy mediapipe import — only needed for face mesh helpers, not for model inference.
# Avoids crashing nail/oral/teeth/posture if mediapipe has import issues.
mp = None

def _get_mediapipe():
    global mp
    if mp is None:
        import mediapipe as _mp
        mp = _mp
    return mp

logger = logging.getLogger(__name__)

# --- Configuration ---
IMG_SIZE_LARGE = (380, 380)
IMG_SIZE_SMALL = (224, 224)

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

ONNX_DIR = BASE_DIR / "saved_models_onnx"

# --- Class Definitions ---
ORAL_CANCER_CLASSES = ['Oral_Cancer', 'Normal']
TEETH_CLASSES = ['Calculus', 'Caries', 'Gingivitis', 'Healthy', 'Hypodontia', 'Mouth Ulcer', 'Tooth Discoloration']

# --- ONNX Session Cache ---
_sess_nail    = None
_nail_classes = {}
_sess_oral    = None
_sess_teeth   = None
_teeth_classes = {}
_sess_posture = None
_burns_model  = None   # kept as PyTorch/YOLO – no ONNX for detection models

_model_locks = {
    'nail':        threading.Lock(),
    'oral_cancer': threading.Lock(),
    'teeth':       threading.Lock(),
    'posture':     threading.Lock(),
    'burns':       threading.Lock(),
}

# --- ONNX Session Factory ---

# Standardization params (ImageNet)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Clinical Calibration Temperature
TEMPERATURE = 1.5

def _make_session(onnx_path: Path) -> ort.InferenceSession | None:
    try:
        import torch
        _cuda_ok = torch.cuda.is_available()
    except Exception:
        _cuda_ok = False
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if _cuda_ok else ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
        logger.info(f"ONNX session for {onnx_path.name}: {sess.get_providers()[0]}")
        return sess
    except Exception as e:
        logger.error(f"Failed to create ONNX session for {onnx_path}: {e}")
        return None

def apply_color_constancy(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply Gray World Color Constancy algorithm.
    Removes color tint by forcing the average color of the image to neutral gray.
    """
    try:
        # Convert to float to avoid overflow
        img_float = img_bgr.astype(np.float32)
        
        # Calculate per-channel average
        avg_b = np.mean(img_float[:, :, 0])
        avg_g = np.mean(img_float[:, :, 1])
        avg_r = np.mean(img_float[:, :, 2])
        
        # Calculate scaling factors
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        if avg_b == 0 or avg_g == 0 or avg_r == 0:
            return img_bgr
            
        kp = avg_gray / avg_b
        kg = avg_gray / avg_g
        kr = avg_gray / avg_r
        
        # Scale channels
        img_float[:, :, 0] *= kp
        img_float[:, :, 1] *= kg
        img_float[:, :, 2] *= kr
        
        # Clip and convert back to uint8
        return np.clip(img_float, 0, 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"Color constancy error: {e}")
        return img_bgr

def preprocess_to_tensor(img_bgr: np.ndarray, size: tuple, calibrate: bool = False) -> np.ndarray | None:
    """BGR image → float32 NCHW numpy tensor ready for ONNX inference."""
    try:
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
            
        # Optional Calibration (Phase 2 feature)
        if calibrate:
            img_bgr = apply_color_constancy(img_bgr)
            
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, size), cv2.COLOR_BGR2RGB)
        img_f   = img_rgb.astype(np.float32) / 255.0
        img_f   = (np.clip(img_f, 0.0, 1.0) - _MEAN) / _STD
        img_f   = np.nan_to_num(img_f, nan=0.0)
        return img_f.transpose(2, 0, 1)[np.newaxis]   # (1, C, H, W)
    except Exception as e:
        logger.error(f"Preprocess error: {e}")
        return None

def _softmax(logits: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    """Softmax with clinical temperature scaling."""
    scaled = logits / temperature
    e = np.exp(scaled - scaled.max())
    return e / e.sum()

def _run_classification(sess, tensor) -> tuple[np.ndarray, int, float]:
    """Helper: run session, return (probs, idx, conf)."""
    input_name = sess.get_inputs()[0].name
    logits     = sess.run(None, {input_name: tensor})[0][0]
    probs      = _softmax(logits)
    idx        = int(np.argmax(probs))
    conf       = float(np.clip(probs[idx], 0.0, 1.0))
    return probs, idx, conf

# ─────────────────────────────────────────────────────────────
# NAIL DISEASE
# ─────────────────────────────────────────────────────────────

def get_nail_session() -> ort.InferenceSession | None:
    global _sess_nail, _nail_classes
    if _sess_nail: return _sess_nail
    with _model_locks['nail']:
        if _sess_nail: return _sess_nail
        onnx_path = ONNX_DIR / "nail_disease.onnx"
        map_path  = BASE_DIR / "saved_models" / "nail_disease_mapping.json"
        if not onnx_path.exists():
            logger.error(f"Nail ONNX model missing: {onnx_path}"); return None
        if map_path.exists():
            with open(map_path) as f:
                raw = json.load(f)
            _nail_classes = {int(k): v for k, v in raw.items()}
        _sess_nail = _make_session(onnx_path)
        logger.info(f"Nail ONNX loaded — {len(_nail_classes)} classes.")
    return _sess_nail

def predict_nail_disease(img_bgr: np.ndarray, debug=False, calibrate=False):
    sess = get_nail_session()
    if not sess or not _nail_classes:
        return "Model Not Loaded", 0.0, {"error": "Nail ONNX model unavailable"}
    tensor = preprocess_to_tensor(img_bgr, IMG_SIZE_LARGE, calibrate=calibrate)
    if tensor is None:
        return "Preprocessing Failed", 0.0, {"error": "preprocess failed"}
    try:
        probs, idx, conf = _run_classification(sess, tensor)
    except Exception as e:
        return "Error", 0.0, {"error": str(e)}
    label = _nail_classes.get(idx, f"Unknown {idx}")
    top3_idx = np.argsort(probs)[::-1][:3]
    debug_info = {
        "top_3": [{"label": _nail_classes.get(int(i), f"Class {i}"),
                   "confidence": float(np.clip(probs[i], 0.0, 1.0))} for i in top3_idx]
    }
    if debug:
        img_resized = cv2.resize(img_bgr, IMG_SIZE_LARGE)
        _, buf = cv2.imencode('.jpg', img_resized)
        debug_info["debug_image"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    return label, conf, debug_info

# ─────────────────────────────────────────────────────────────
# ORAL CANCER
# ─────────────────────────────────────────────────────────────

def get_oral_session() -> ort.InferenceSession | None:
    global _sess_oral
    if _sess_oral: return _sess_oral
    with _model_locks['oral_cancer']:
        if _sess_oral: return _sess_oral
        path = ONNX_DIR / "oral_cancer.onnx"
        if not path.exists():
            logger.error(f"Oral Cancer ONNX model missing: {path}"); return None
        _sess_oral = _make_session(path)
        logger.info("Oral Cancer ONNX loaded.")
    return _sess_oral

def predict_oral_cancer(img_bgr: np.ndarray, debug=False, calibrate=False, preprocessed=False):
    sess = get_oral_session()
    if not sess:
        return "Model Not Loaded", 0.0, {"error": "Oral Cancer ONNX unavailable"}
    debug_info = {}
    # Mouth BBox for Nerd Mode (MediaPipe)
    # Skip face detection when image is already a pre-cropped mouth ROI
    if not preprocessed:
        bbox, _, _ = analyze_mouth(img_bgr)
        if bbox:
            debug_info["bbox"] = bbox
    tensor = preprocess_to_tensor(img_bgr, IMG_SIZE_SMALL, calibrate=calibrate)  # 224x224
    if tensor is None:
        return "Preprocessing Failed", 0.0, {"error": "preprocess failed"}
    try:
        probs, idx, conf = _run_classification(sess, tensor)
    except Exception as e:
        return "Error", 0.0, {"error": str(e)}
    label = ORAL_CANCER_CLASSES[idx] if idx < len(ORAL_CANCER_CLASSES) else f"Unknown {idx}"
    debug_info["top_3"] = [
        {"label": ORAL_CANCER_CLASSES[i], "confidence": float(np.clip(probs[i], 0.0, 1.0))}
        for i in range(len(ORAL_CANCER_CLASSES))
    ]
    if debug:
        img_resized = cv2.resize(img_bgr, IMG_SIZE_SMALL)
        _, buf = cv2.imencode('.jpg', img_resized)
        debug_info["debug_image"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    return label, conf, debug_info

# ─────────────────────────────────────────────────────────────
# TEETH DISEASE
# ─────────────────────────────────────────────────────────────

def get_teeth_session() -> ort.InferenceSession | None:
    global _sess_teeth, _teeth_classes
    if _sess_teeth: return _sess_teeth
    with _model_locks['teeth']:
        if _sess_teeth: return _sess_teeth
        onnx_path = ONNX_DIR / "teeth_disease.onnx"
        map_path  = BASE_DIR / "saved_models" / "teeth_disease_mapping.json"
        if not onnx_path.exists():
            logger.error(f"Teeth ONNX model missing: {onnx_path}"); return None
        if map_path.exists():
            with open(map_path) as f:
                raw = json.load(f)
            # mapping is class_name → idx, invert it
            _teeth_classes = {v: k for k, v in raw.items()} if isinstance(list(raw.values())[0], int) else \
                             {int(k): v for k, v in raw.items()}
        else:
            _teeth_classes = {i: TEETH_CLASSES[i] for i in range(len(TEETH_CLASSES))}
        _sess_teeth = _make_session(onnx_path)
        logger.info(f"Teeth ONNX loaded — {len(_teeth_classes)} classes.")
    return _sess_teeth

def predict_teeth_disease(img_bgr: np.ndarray, debug=False, calibrate=False, preprocessed=False):
    sess = get_teeth_session()
    if not sess:
        return "Model Not Loaded", 0.0, {"error": "Teeth Disease ONNX unavailable"}
    debug_info = {}
    # Mouth check — skip when image is already a pre-cropped mouth ROI
    if not preprocessed:
        bbox, open_ratio, error = analyze_mouth(img_bgr)
        if error:
            return "No Face Detected", 0.0, {"error": error}
        if bbox:
            debug_info["bbox"] = bbox
            debug_info["mouth_open_ratio"] = float(open_ratio)
        if open_ratio < 0.05:
            return "Mouth Closed", 0.0, {
                "error": "Please open your mouth to inspect teeth.",
                "mouth_open_ratio": float(open_ratio), "bbox": bbox
            }
    tensor = preprocess_to_tensor(img_bgr, IMG_SIZE_SMALL, calibrate=calibrate)  # 224x224
    if tensor is None:
        return "Preprocessing Failed", 0.0, {"error": "preprocess failed"}
    try:
        probs, idx, conf = _run_classification(sess, tensor)
    except Exception as e:
        return "Error", 0.0, {"error": str(e)}
    label = _teeth_classes.get(idx, f"Unknown {idx}")
    top3_idx = np.argsort(probs)[::-1][:3]
    debug_info["top_3"] = [
        {"label": _teeth_classes.get(int(i), f"Class {i}"),
         "confidence": float(np.clip(probs[i], 0.0, 1.0))} for i in top3_idx
    ]
    if debug:
        img_resized = cv2.resize(img_bgr, IMG_SIZE_SMALL)
        _, buf = cv2.imencode('.jpg', img_resized)
        debug_info["debug_image"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    return label, conf, debug_info

# ─────────────────────────────────────────────────────────────
# POSTURE CLASSIFIER (ONNX)
# ─────────────────────────────────────────────────────────────

def get_posture_session() -> ort.InferenceSession | None:
    global _sess_posture
    if _sess_posture: return _sess_posture
    with _model_locks['posture']:
        if _sess_posture: return _sess_posture
        path = ONNX_DIR / "posture.onnx"
        if not path.exists():
            logger.error(f"Posture ONNX model missing: {path}"); return None
        _sess_posture = _make_session(path)
        logger.info("Posture ONNX loaded.")
    return _sess_posture

def predict_posture_from_landmarks(landmarks, debug=False):
    """ONNX inference — Posture from 12-keypoint (24-float) feature vector."""
    FEATURE_SIZE = 24
    sess = get_posture_session()
    if not sess:
        return "Model Not Loaded", 0.0, {"error": "Posture ONNX unavailable"}
    try:
        # Flatten nested landmark formats
        if isinstance(landmarks, np.ndarray):
            flat = landmarks.flatten().tolist()
        elif isinstance(landmarks, list) and len(landmarks) > 0:
            if isinstance(landmarks[0], (list, tuple, np.ndarray)):
                flat = [float(v) for pt in landmarks for v in (pt[:2] if len(pt) >= 2 else [0.0, 0.0])]
            elif isinstance(landmarks[0], dict):
                flat = [c for pt in landmarks for c in [float(pt.get('x', 0.0)), float(pt.get('y', 0.0))]]
            else:
                flat = [float(v) for v in landmarks]
        else:
            flat = [float(v) for v in landmarks]

        # Pad / truncate
        flat = (flat + [0.0] * FEATURE_SIZE)[:FEATURE_SIZE]

        tensor = np.array([flat], dtype=np.float32)   # (1, 24)
        input_name = sess.get_inputs()[0].name
        logits = sess.run(None, {input_name: tensor})[0][0]
        probs  = _softmax(logits)
        idx    = int(np.argmax(probs))
        conf   = float(np.clip(probs[idx], 0.0, 1.0))

        label = "Good Form" if idx == 0 else "Bad Form"
        return label, conf, {}
    except Exception as e:
        logger.error(f"Posture inference error: {e}")
        return "Error", 0.0, {"error": str(e)}

# ─────────────────────────────────────────────────────────────
# BURNS DETECTION (YOLO — stays as PyTorch)
# ─────────────────────────────────────────────────────────────

class BurnsModel:
    """Thin wrapper to hold a loaded YOLO model + its type tag."""
    def __init__(self, yolo_model=None):
        self.model = yolo_model
        self.model_type = 'unknown'

def _detect_skin_color(img_bgr: np.ndarray) -> np.ndarray:
    """Quick HSV skin-color mask."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower, upper)
    lower2 = np.array([170, 20, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(mask1, mask2)

def get_burns_model():
    global _burns_model
    if _burns_model: return _burns_model
    with _model_locks['burns']:
        if _burns_model: return _burns_model
        model_paths = [
            BASE_DIR / "saved_models" / "skin_burn_2022_8_21.pt",
            BASE_DIR / "saved_models" / "best.pt",
        ]
        path = next((p for p in model_paths if p.exists()), None)
        if not path:
            found = list((BASE_DIR / "saved_models").glob("*.pt"))
            path = found[0] if found else None
        if not path:
            logger.warning("Burns model (.pt) not found."); return None
        repo_path = BASE_DIR / "yolov7_ops"
        if repo_path.exists():
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
            try:
                import torch
                from models.experimental import attempt_load
                model = attempt_load(str(path), map_location='cpu')
                _burns_model = BurnsModel(model)
                _burns_model.model_type = 'yolov7_external'
                logger.info(f"Burns YOLO model loaded from {path.name}")
            except Exception as e:
                logger.error(f"Burns model load failed: {e}")
        return _burns_model


def predict_burns(img_bgr: np.ndarray, debug=False, calibrate=False):
    """Burns detection using YOLOv7."""
    # Optional Calibration
    if calibrate:
        img_bgr = apply_color_constancy(img_bgr)

    # Check for hands/body parts for context if needed...
    try:
        skin_mask = _detect_skin_color(img_bgr)
        total = img_bgr.shape[0] * img_bgr.shape[1]
        if cv2.countNonZero(skin_mask) / (total + 1e-6) < 0.05:
            return "No Skin Detected", 1.0, {}
    except Exception:
        pass

    model = get_burns_model()
    if model is None:
        return "Model Not Loaded", 0.0, {"error": "Burns model unavailable"}

    debug_info = {}
    try:
        import torch
        from utils.general import non_max_suppression, scale_coords

        img_size = 640
        img = cv2.resize(img_bgr, (img_size, img_size))[:, :, ::-1].transpose(2, 0, 1)
        img = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
        if img.ndim == 3: img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model.model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)
        det = pred[0]

        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_bgr.shape).round()
            best_conf, best_cls, best_bbox = 0.0, -1, None
            for *xyxy, conf, cls in det:
                if conf.item() > best_conf:
                    best_conf, best_cls, best_bbox = conf.item(), int(cls.item()), [x.item() for x in xyxy]
            names = getattr(model.model, 'module', model.model).names
            label = (names.get(best_cls, 'Burn') if hasattr(names, 'keys') else
                     (names[best_cls] if best_cls < len(names) else 'Burn'))
            if label.lower() == 'burn': label = 'Burn Detected'
            if debug and best_bbox:
                h, w = img_bgr.shape[:2]
                debug_info['bbox'] = [best_bbox[0]/w, best_bbox[1]/h, best_bbox[2]/w, best_bbox[3]/h]
                dbg = img_bgr.copy()
                cv2.rectangle(dbg, (int(best_bbox[0]), int(best_bbox[1])),
                              (int(best_bbox[2]), int(best_bbox[3])), (0, 0, 255), 2)
                _, buf = cv2.imencode('.jpg', dbg)
                debug_info["overlay"] = base64.b64encode(buf).decode()
            return label, best_conf, debug_info
        return "Healthy", 0.95, debug_info
    except Exception as e:
        logger.error(f"Burns inference error: {e}")
        return "Error", 0.0, {"error": str(e)}

# ─────────────────────────────────────────────────────────────
# MEDIAPIPE HELPERS (used by Oral Cancer + Teeth)
# ─────────────────────────────────────────────────────────────

_mp_face_mesh = None

def get_face_mesh():
    global _mp_face_mesh
    if _mp_face_mesh is None:
        _mp = _get_mediapipe()
        _mp_face_mesh = _mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5
        )
    return _mp_face_mesh

def segment_eye_mp(img_bgr: np.ndarray):
    face_mesh = get_face_mesh()
    h, w = img_bgr.shape[:2]
    results = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks: return None, None
    lm = results.multi_face_landmarks[0].landmark
    eye_indices = [33, 133, 157, 158, 159, 160, 161, 246, 7, 163, 144, 145, 153, 154, 155, 133]
    xs = [lm[i].x * w for i in eye_indices]
    ys = [lm[i].y * h for i in eye_indices]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    pw, ph = int((x2-x1)*0.3), int((y2-y1)*0.3)
    x1, y1, x2, y2 = max(0,x1-pw), max(0,y1-ph), min(w,x2+pw), min(h,y2+ph)
    crop = img_bgr[y1:y2, x1:x2]
    mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    pts = np.array([[lm[i].x*w - x1, lm[i].y*h - y1] for i in eye_indices], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return crop, mask

def analyze_mouth(img_bgr: np.ndarray):
    try:
        face_mesh = get_face_mesh()
        results = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None, 0.0, "No Face Detected"
        lm = results.multi_face_landmarks[0].landmark
        h, w, _ = img_bgr.shape
        top_lip, bot_lip = lm[13], lm[14]
        face_h = (lm[152].y - lm[10].y) * h
        mouth_d = np.linalg.norm(
            np.array([top_lip.x*w, top_lip.y*h]) - np.array([bot_lip.x*w, bot_lip.y*h])
        )
        ratio = mouth_d / (face_h + 1e-6)
        lip_indices = [61,146,91,181,84,17,314,405,321,375,291,185,40,39,37,0,267,269,270,409]
        xs = [lm[i].x for i in lip_indices]; ys = [lm[i].y for i in lip_indices]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        pw, ph = (x2-x1)*0.2, (y2-y1)*0.2
        bbox = [max(0.0,x1-pw), max(0.0,y1-ph), min(1.0,x2+pw), min(1.0,y2+ph)]
        return bbox, ratio, None
    except Exception as e:
        return None, 0.0, str(e)
