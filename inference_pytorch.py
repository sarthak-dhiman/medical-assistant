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
IMG_SIZE = (380, 380)  # ONNX models expect 380x380
SCLERA_SIZE = (64, 64)   # ONNX model expects 64x64 for sclera input
MIN_IMAGE_SIZE = 20     # px (single int, used in scalar comparisons)
MAX_IMAGE_SIZE = 3000   # px

CONFIDENCE_THRESHOLD_JAUNDICE_BODY = 0.7
CONFIDENCE_THRESHOLD_JAUNDICE_EYE = 0.7
CONFIDENCE_THRESHOLD_SKIN_DISEASE = 0.8

# Clinical Calibration Temperature
TEMPERATURE = 1.5

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
    """Create an ONNX InferenceSession with smart CUDA memory management."""
    # Check CUDA availability and memory
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # Check GPU memory before attempting CUDA
            torch.cuda.empty_cache()  # Clear cache first
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            # Require at least 1GB free memory for CUDA execution
            cuda_available = free_memory > 1024 * 1024 * 1024
            if not cuda_available:
                logger.warning(f"Insufficient GPU memory ({free_memory/1024**3:.2f}GB free), falling back to CPU")
    except Exception as e:
        logger.warning(f"CUDA check failed: {e}, using CPU")
        cuda_available = False
    
    # Configure providers based on memory availability
    if cuda_available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # CUDA-specific options for memory efficiency
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1  # Reduce threads for CUDA
        # Enable memory optimization
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:
        providers = ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4  # More threads for CPU
    
    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
        logger.info(f"ONNX session for {onnx_path.name} using: {sess.get_providers()[0]}")
        return sess
    except Exception as e:
        if cuda_available:
            logger.warning(f"CUDA failed for {onnx_path.name}: {e}, retrying with CPU")
            # Fallback to CPU if CUDA fails
            providers = ["CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            try:
                sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
                logger.info(f"ONNX session for {onnx_path.name} fallback using: {sess.get_providers()[0]}")
                return sess
            except Exception as e2:
                logger.error(f"CPU fallback also failed for {onnx_path.name}: {e2}")
                return None
        else:
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
        if _sess_eye:
            logger.info("Jaundice Eye ONNX model loaded.")
        else:
            logger.error("Jaundice Eye ONNX session creation failed.")
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
        if _sess_body:
            logger.info("Jaundice Body ONNX model loaded.")
        else:
            logger.error("Jaundice Body ONNX session creation failed.")
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
        if _sess_skin:
            logger.info(f"Skin Disease ONNX model loaded — {len(_skin_classes)} classes.")
        else:
            logger.error("Skin Disease ONNX session creation failed.")
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

def apply_color_constancy(img_bgr: np.ndarray) -> np.ndarray:
    """Gray World Color Constancy."""
    try:
        img_float = img_bgr.astype(np.float32)
        avg_b = np.mean(img_float[:, :, 0])
        avg_g = np.mean(img_float[:, :, 1])
        avg_r = np.mean(img_float[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        if avg_b == 0 or avg_g == 0 or avg_r == 0: return img_bgr
        img_float[:, :, 0] *= (avg_gray / avg_b)
        img_float[:, :, 1] *= (avg_gray / avg_g)
        img_float[:, :, 2] *= (avg_gray / avg_r)
        return np.clip(img_float, 0, 255).astype(np.uint8)
    except Exception: return img_bgr

def _preprocess(img_bgr, size, calibrate=False):
    """BGR numpy -> float32 NCHW tensor."""
    try:
        img_bgr = ensure_3channel(img_bgr)
        if calibrate:
            img_bgr = apply_color_constancy(img_bgr)
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, size), cv2.COLOR_BGR2RGB)
        img_f   = img_rgb.astype(np.float32) / 255.0
        # Normalization (ImageNet)
        img_f   = (img_f - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return img_f.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    except Exception as e:
        logger.error(f"Preprocess failed: {e}")
        return None

# --- Prediction Functions ---

def predict_jaundice_eye(skin_img: np.ndarray, sclera_crop=None, debug=False, calibrate=False):
    sess = get_eye_session()
    if not sess: return "Model Not Loaded", 0.0, {"error": "Eye ONNX unavailable"}

    is_valid, msg = validate_image(skin_img, "skin image")
    if not is_valid:
        return "Invalid Input", 0.0, {"error": msg}

    # Clear GPU cache before inference
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Preprocess with optional calibration
    skin_tensor = _preprocess(skin_img, IMG_SIZE, calibrate=calibrate)
    if skin_tensor is None:
        return "Preprocessing Failed", 0.0, {"error": "skin preprocess failed"}

    # Sclera fallback
    if sclera_crop is None or (isinstance(sclera_crop, np.ndarray) and sclera_crop.size == 0):
        sclera_crop = np.zeros((*SCLERA_SIZE, 3), dtype=np.uint8)
    else:
        sclera_crop = ensure_3channel(sclera_crop)

    sclera_tensor = _preprocess(sclera_crop, SCLERA_SIZE, calibrate=calibrate)
    if sclera_tensor is None:
        sclera_tensor = np.zeros((1, 3, *SCLERA_SIZE), dtype=np.float32)

    try:
        # Multi-input model: skin_input and sclera_input
        input_names = [inp.name for inp in sess.get_inputs()]
        feeds = {input_names[0]: skin_tensor, input_names[1]: sclera_tensor}
        logit = sess.run(None, feeds)[0][0, 0]
        # Temperature Scaling for Clinical Calibration
        prob  = float(1.0 / (1.0 + np.exp(-(logit / TEMPERATURE))))  # sigmoid
    except Exception as e:
        logger.error(f"Jaundice Eye inference failed: {e}")
        return "Error", 0.0, {"error": str(e)}
    finally:
        # Clear GPU cache after inference
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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


def predict_jaundice_body(img_bgr: np.ndarray, debug=False, calibrate=False):
    """ONNX inference — Jaundice Body."""
    sess = get_body_session()
    if not sess: return "Model Not Loaded", 0.0, {"error": "Body ONNX unavailable"}

    is_valid, msg = validate_image(img_bgr, "body image")
    if not is_valid:
        return "Invalid Input", 0.0, {"error": msg}

    # Clear GPU cache before inference
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    tensor = _preprocess(img_bgr, IMG_SIZE, calibrate=calibrate)
    if tensor is None:
        return "Preprocessing Failed", 0.0, {"error": "preprocess failed"}

    try:
        input_name = sess.get_inputs()[0].name
        print(f"[DEBUG Jaundice Body] img_bgr shape: {img_bgr.shape}, tensor shape: {tensor.shape}", flush=True)
        logit = sess.run(None, {input_name: tensor})[0][0, 0]
        # Temperature Scaling for Clinical Calibration
        prob  = float(1.0 / (1.0 + np.exp(-(logit / TEMPERATURE))))
    except Exception as e:
        logger.error(f"Jaundice Body inference failed: {e}")
        return "Error", 0.0, {"error": str(e)}
    finally:
        # Clear GPU cache after inference
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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


def predict_skin_disease_torch(img_bgr: np.ndarray, debug=False, calibrate=False):
    """ONNX inference — Skin Disease (38-class)."""
    sess = get_skin_session()
    if not sess:
        return "Model Missing", 0.0, {"error": "ONNX model not loaded"}

    is_valid, msg = validate_image(img_bgr, "skin image")
    if not is_valid:
        return "Invalid Input", 0.0, {"error": msg}

    # Clear GPU cache before inference
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    tensor = _preprocess(img_bgr, IMG_SIZE, calibrate=calibrate)
    if tensor is None:
        return "Preprocessing Failed", 0.0, {"error": "preprocess failed"}

    try:
        input_name = sess.get_inputs()[0].name
        logits = sess.run(None, {input_name: tensor})[0][0]  # shape: (num_classes,)
        
        # Temperature Scaling for Softmax
        scaled_logits = logits / TEMPERATURE
        exp_l  = np.exp(scaled_logits - scaled_logits.max())
        probs  = exp_l / exp_l.sum()
        idx    = int(np.argmax(probs))
        conf   = float(probs[idx])
    except Exception as e:
        logger.error(f"Skin Disease inference failed: {e}")
        return "Error", 0.0, {"error": str(e)}
    finally:
        # Clear GPU cache after inference
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
