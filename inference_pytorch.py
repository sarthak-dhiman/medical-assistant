"""
inference_pytorch.py  —  ONNX Runtime Edition
===============================================
Inference for Jaundice (Eye + Body) and Skin Disease models.
Uses onnxruntime instead of PyTorch for faster, lighter inference.
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import json
import threading
import base64
import logging

logger = logging.getLogger(__name__)

# --- Configuration ---
IMG_SIZE  = (380, 380)
SCLERA_SIZE = (64, 64)
MIN_IMAGE_SIZE = 10
MAX_IMAGE_SIZE = 4096
CONFIDENCE_THRESHOLD_JAUNDICE_EYE  = 0.5
CONFIDENCE_THRESHOLD_JAUNDICE_BODY = 0.70
CONFIDENCE_THRESHOLD_SKIN_DISEASE  = 0.80

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

ONNX_DIR = BASE_DIR / "saved_models_onnx"

# --- Session Cache (one session per model) ---
_sess_eye   = None
_sess_body  = None
_sess_skin  = None
_skin_classes: dict = {}
_model_locks = {
    'eye':  threading.Lock(),
    'body': threading.Lock(),
    'skin': threading.Lock(),
}

# --- ONNX Session Factory ---

def _make_session(onnx_path: Path) -> ort.InferenceSession:
    """Create an ONNX InferenceSession, preferring CUDA then CPU."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
        logger.info(f"ONNX session for {onnx_path.name} using: {sess.get_providers()[0]}")
        return sess
    except Exception as e:
        logger.error(f"Failed to create ONNX session for {onnx_path}: {e}")
        return None

def get_eye_session() -> ort.InferenceSession | None:
    global _sess_eye
    if _sess_eye: return _sess_eye
    with _model_locks['eye']:
        if _sess_eye: return _sess_eye
        path = ONNX_DIR / "jaundice_eye.onnx"
        if not path.exists():
            logger.error(f"ONNX model not found: {path}")
            return None
        _sess_eye = _make_session(path)
        logger.info("Jaundice Eye ONNX model loaded.")
    return _sess_eye

def get_body_session() -> ort.InferenceSession | None:
    global _sess_body
    if _sess_body: return _sess_body
    with _model_locks['body']:
        if _sess_body: return _sess_body
        path = ONNX_DIR / "jaundice_body.onnx"
        if not path.exists():
            logger.error(f"ONNX model not found: {path}")
            return None
        _sess_body = _make_session(path)
        logger.info("Jaundice Body ONNX model loaded.")
    return _sess_body

def get_skin_session() -> ort.InferenceSession | None:
    global _sess_skin, _skin_classes
    if _sess_skin: return _sess_skin
    with _model_locks['skin']:
        if _sess_skin: return _sess_skin
        onnx_path = ONNX_DIR / "skin_disease.onnx"
        map_path  = BASE_DIR / "saved_models" / "skin_disease_mapping.json"
        if not onnx_path.exists() or not map_path.exists():
            logger.error(f"Skin ONNX model or mapping missing.")
            return None
        with open(map_path) as f:
            raw = json.load(f)
        first_val = list(raw.values())[0]
        if isinstance(first_val, int):
            _skin_classes = {v: k for k, v in raw.items()}
        else:
            _skin_classes = {int(k): v for k, v in raw.items()}
        _sess_skin = _make_session(onnx_path)
        logger.info(f"Skin Disease ONNX model loaded — {len(_skin_classes)} classes.")
    return _sess_skin

# --- Preprocessing ---

def validate_image(img_bgr, context="image") -> tuple[bool, str]:
    if img_bgr is None: return False, f"{context} is None"
    if not isinstance(img_bgr, np.ndarray): return False, f"{context} is not numpy array"
    if img_bgr.size == 0: return False, f"{context} is empty"
    h, w = img_bgr.shape[:2]
    if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE: return False, f"{context} too small ({w}x{h})"
    if h > MAX_IMAGE_SIZE or w > MAX_IMAGE_SIZE: return False, f"{context} too large ({w}x{h})"
    return True, "Valid"

def ensure_3channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4: return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_to_tensor(img_bgr: np.ndarray, size: tuple) -> np.ndarray | None:
    """BGR image → float32 NCHW numpy tensor ready for ONNX input."""
    try:
        img_bgr = ensure_3channel(img_bgr)
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, size), cv2.COLOR_BGR2RGB)
        img_f   = img_rgb.astype(np.float32) / 255.0
        img_f   = (np.clip(img_f, 0.0, 1.0) - _MEAN) / _STD
        img_f   = np.nan_to_num(img_f, nan=0.0, posinf=0.0, neginf=0.0)
        return img_f.transpose(2, 0, 1)[np.newaxis]  # NCHW
    except Exception as e:
        logger.error(f"Preprocess failed: {e}")
        return None

# --- Prediction Functions ---

def predict_jaundice_eye(skin_img: np.ndarray, sclera_crop=None, debug=False):
    """ONNX inference — Jaundice Eye (dual-input: skin 380x380 + sclera 64x64)."""
    sess = get_eye_session()
    if not sess:
        return "Model Missing", 0.0, {"error": "ONNX model not loaded"}

    is_valid, msg = validate_image(skin_img, "skin image")
    if not is_valid:
        return "Invalid Input", 0.0, {"error": msg}

    skin_tensor = preprocess_to_tensor(skin_img, IMG_SIZE)
    if skin_tensor is None:
        return "Preprocessing Failed", 0.0, {"error": "skin preprocess failed"}

    # Sclera fallback
    if sclera_crop is None or (isinstance(sclera_crop, np.ndarray) and sclera_crop.size == 0):
        sclera_crop = np.zeros((*SCLERA_SIZE, 3), dtype=np.uint8)
    else:
        sclera_crop = ensure_3channel(sclera_crop)

    sclera_tensor = preprocess_to_tensor(sclera_crop, SCLERA_SIZE)
    if sclera_tensor is None:
        sclera_tensor = np.zeros((1, 3, *SCLERA_SIZE), dtype=np.float32)

    try:
        # Multi-input model: skin_input and sclera_input
        input_names = [inp.name for inp in sess.get_inputs()]
        feeds = {input_names[0]: skin_tensor, input_names[1]: sclera_tensor}
        logit = sess.run(None, feeds)[0][0, 0]
        prob  = float(1.0 / (1.0 + np.exp(-logit)))  # sigmoid
    except Exception as e:
        logger.error(f"Jaundice Eye inference failed: {e}")
        return "Error", 0.0, {"error": str(e)}

    threshold = CONFIDENCE_THRESHOLD_JAUNDICE_EYE
    label = "Jaundice" if prob > threshold else "Normal"
    conf  = prob if prob > threshold else 1.0 - prob
    conf  = float(np.clip(conf, 0.0, 1.0))

    debug_info: dict = {"raw_probability": prob, "threshold_used": threshold}

    # Sclera color stats for Nerd Mode
    try:
        mean_bgr = np.mean(sclera_crop, axis=(0, 1))
        mean_hsv = np.mean(cv2.cvtColor(sclera_crop, cv2.COLOR_BGR2HSV), axis=(0, 1))
        debug_info["color_stats"] = {
            "mean_rgb": [int(np.clip(x, 0, 255)) for x in mean_bgr[::-1]],
            "mean_hsv": [int(np.clip(x, 0, 255)) for x in mean_hsv],
        }
    except Exception:
        pass

    return label, conf, debug_info


def predict_jaundice_body(img_bgr: np.ndarray, debug=False):
    """ONNX inference — Jaundice Body."""
    sess = get_body_session()
    if not sess:
        return "Model Missing", 0.0, {"error": "ONNX model not loaded"}

    is_valid, msg = validate_image(img_bgr, "body image")
    if not is_valid:
        return "Invalid Input", 0.0, {"error": msg}

    tensor = preprocess_to_tensor(img_bgr, IMG_SIZE)
    if tensor is None:
        return "Preprocessing Failed", 0.0, {"error": "preprocess failed"}

    try:
        input_name = sess.get_inputs()[0].name
        logit = sess.run(None, {input_name: tensor})[0][0, 0]
        prob  = float(1.0 / (1.0 + np.exp(-logit)))
    except Exception as e:
        logger.error(f"Jaundice Body inference failed: {e}")
        return "Error", 0.0, {"error": str(e)}

    threshold = CONFIDENCE_THRESHOLD_JAUNDICE_BODY
    label = "Jaundice" if prob > threshold else "Normal"
    conf  = prob if prob > threshold else 1.0 - prob
    conf  = float(np.clip(conf, 0.0, 1.0))

    debug_info: dict = {"raw_probability": prob, "threshold_used": threshold}

    try:
        mean_bgr = np.mean(img_bgr, axis=(0, 1))
        mean_hsv = np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), axis=(0, 1))
        debug_info["color_stats"] = {
            "mean_rgb": [int(np.clip(x, 0, 255)) for x in mean_bgr[::-1]],
            "mean_hsv": [int(np.clip(x, 0, 255)) for x in mean_hsv],
        }
    except Exception:
        pass

    return label, conf, debug_info


def predict_skin_disease_torch(img_bgr: np.ndarray, debug=False):
    """ONNX inference — Skin Disease (38-class)."""
    sess = get_skin_session()
    if not sess:
        return "Model Missing", 0.0, {"error": "ONNX model not loaded"}

    is_valid, msg = validate_image(img_bgr, "skin image")
    if not is_valid:
        return "Invalid Input", 0.0, {"error": msg}

    tensor = preprocess_to_tensor(img_bgr, IMG_SIZE)
    if tensor is None:
        return "Preprocessing Failed", 0.0, {"error": "preprocess failed"}

    try:
        input_name = sess.get_inputs()[0].name
        logits = sess.run(None, {input_name: tensor})[0][0]  # shape: (num_classes,)
        # Softmax
        exp_l  = np.exp(logits - logits.max())
        probs  = exp_l / exp_l.sum()
        idx    = int(np.argmax(probs))
        conf   = float(probs[idx])
    except Exception as e:
        logger.error(f"Skin Disease inference failed: {e}")
        return "Error", 0.0, {"error": str(e)}

    conf  = float(np.clip(conf, 0.0, 1.0))
    threshold = CONFIDENCE_THRESHOLD_SKIN_DISEASE

    if conf < threshold:
        label = "Normal"
    else:
        label = _skin_classes.get(idx, f"Unknown Class {idx}")

    debug_info: dict = {
        "raw_probability": conf,
        "threshold_used": threshold,
        "class_index": idx,
    }

    # Top-3 for Nerd Mode
    try:
        top3_idx  = np.argsort(probs)[::-1][:3]
        debug_info["top_3"] = [
            {"label": _skin_classes.get(int(i), f"Class {i}"), "confidence": float(np.clip(probs[i], 0.0, 1.0))}
            for i in top3_idx
        ]
    except Exception:
        pass

    return label, conf, debug_info
