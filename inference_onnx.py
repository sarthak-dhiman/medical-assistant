import cv2
import numpy as np
import onnxruntime as ort
import os
import sys
import json
import time
from pathlib import Path

# --- Configuration ---
IMG_SIZE = (380, 380)
SCLERA_SIZE = (64, 64)

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

# --- Global Sessions ---
_eye_sess = None
_body_sess = None
_skin_sess = None
_burns_sess = None
_nail_sess = None
_skin_classes = {}
_nail_classes = {}
USE_CUDA = os.getenv("ONNX_USE_CUDA", "0") == "1"

def get_ort_session(model_name):
    path = BASE_DIR / "saved_models" / "onnx" / f"{model_name}.onnx"
    if not path.exists():
        print(f"ONNX Model missing: {path}")
        return None
    
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if USE_CUDA else ['CPUExecutionProvider']
        sess = ort.InferenceSession(str(path), providers=providers)
        print(f"Loaded {model_name} (Providers: {sess.get_providers()})")
        return sess
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

def get_body_session():
    global _body_sess
    if _body_sess: return _body_sess
    _body_sess = get_ort_session("jaundice_body")
    return _body_sess

def get_eye_session():
    global _eye_sess
    if _eye_sess: return _eye_sess
    _eye_sess = get_ort_session("jaundice_eye")
    return _eye_sess

def get_skin_session():
    global _skin_sess, _skin_classes
    if _skin_sess: return _skin_sess
    
    _skin_sess = get_ort_session("skin_disease")
    
    # Load Mapping from original JSON
    map_path = BASE_DIR / "saved_models" / "skin_disease_mapping.json"
    if map_path.exists():
        with open(map_path, 'r') as f:
            raw_map = json.load(f)
            # Logic from inference_pytorch.py
            first_val = list(raw_map.values())[0]
            if isinstance(first_val, int):
                _skin_classes = {v: k for k, v in raw_map.items()}
            else:
                _skin_classes = {int(k): v for k, v in raw_map.items()}
    
    return _skin_sess

def get_burns_session():
    global _burns_sess
    if _burns_sess: return _burns_sess
    _burns_sess = get_ort_session("burns")
    return _burns_sess

def get_nail_session():
    global _nail_sess, _nail_classes
    if _nail_sess: return _nail_sess
    
    _nail_sess = get_ort_session("nail_disease")
    if _nail_sess:
        _load_nail_classes()
    return _nail_sess

def _load_nail_classes():
    global _nail_classes
    if _nail_classes:
        return
    map_path = BASE_DIR / "saved_models" / "nail_disease_mapping.json"
    if map_path.exists():
        with open(map_path, 'r') as f:
            raw_map = json.load(f)
            try:
                _nail_classes = {int(k): v for k, v in raw_map.items()}
            except ValueError:
                _nail_classes = {int(v): k for k, v in raw_map.items()}

# --- Preprocessing ---
def preprocess_efficientnet(img_bgr, size=(380, 380)):
    # Resize
    img_resized = cv2.resize(img_bgr, size)
    # RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # Normalize (ImageNet)
    img_float = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std
    # CHW
    img_chw = np.transpose(img_norm, (2, 0, 1))
    # Batch
    return np.expand_dims(img_chw, axis=0)

def preprocess_simple(img_bgr, size):
    img_resized = cv2.resize(img_bgr, size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_float, (2, 0, 1))
    return np.expand_dims(img_chw, axis=0)

# --- Inference ---

def predict_jaundice_body_onnx(img_bgr):
    session = get_body_session()
    if not session: return "Model Missing", 0.0

    input_name = session.get_inputs()[0].name
    img_in = preprocess_efficientnet(img_bgr, size=IMG_SIZE)
    
    outputs = session.run(None, {input_name: img_in})
    logits = outputs[0]
    
    # Sigmoid
    prob = 1 / (1 + np.exp(-logits)).item()
    
    label = "Jaundice" if prob > 0.5 else "Normal"
    conf = prob if prob > 0.5 else 1.0 - prob
    return label, conf

def predict_jaundice_eye_onnx(skin_img, sclera_crop=None):
    session = get_eye_session()
    if not session: return "Model Missing", 0.0

    # Inputs: skin_input, sclera_input (Verified in export script)
    # Preprocessing: MATCH TRAINING (Simple / 255.0)
    skin_in = preprocess_simple(skin_img, size=IMG_SIZE)
    
    if sclera_crop is None or sclera_crop.size == 0:
        sclera_crop = np.zeros((*SCLERA_SIZE, 3), dtype=np.uint8)
    sclera_in = preprocess_simple(sclera_crop, size=SCLERA_SIZE)
    
    inputs = {
        'skin_input': skin_in, 
        'sclera_input': sclera_in
    }
    
    outputs = session.run(None, inputs)
    logits = outputs[0]
    prob = 1 / (1 + np.exp(-logits)).item()
    
    label = "Jaundice" if prob > 0.5 else "Normal"
    conf = prob if prob > 0.5 else 1.0 - prob
    return label, conf

def predict_skin_disease_onnx(img_bgr):
    session = get_skin_session()
    if not session: return "Model Missing", 0.0

    input_name = session.get_inputs()[0].name
    img_in = preprocess_efficientnet(img_bgr, size=IMG_SIZE)
    
    outputs = session.run(None, {input_name: img_in})
    logits = outputs[0]
    
    # Softmax
    exp = np.exp(logits - np.max(logits)) # Stability
    probs = exp / exp.sum(axis=1)
    
    idx = np.argmax(probs)
    conf = probs[0, idx]
    
    if conf < 0.80:
        label = "Normal"
    else:
        label = _skin_classes.get(idx, f"Class {idx}")
    
    return label, conf

def predict_burns_onnx(img_bgr):
    session = get_burns_session()
    if not session: return "Model Missing", 0.0

    input_name = session.get_inputs()[0].name
    img_in = preprocess_efficientnet(img_bgr, size=IMG_SIZE)
    outputs = session.run(None, {input_name: img_in})
    logits = outputs[0]
    prob = 1 / (1 + np.exp(-logits)).item()
    if prob > 0.5:
        return "Burns Detected", prob
    return "Healthy/Normal", 1.0 - prob

def predict_nail_disease_onnx(img_bgr):
    session = get_nail_session()
    if not session: return "Model Missing", 0.0

    input_name = session.get_inputs()[0].name
    img_in = preprocess_efficientnet(img_bgr, size=IMG_SIZE)
    outputs = session.run(None, {input_name: img_in})
    logits = outputs[0]

    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum(axis=1)
    idx = int(np.argmax(probs))
    conf = float(probs[0, idx])
    label = _nail_classes.get(idx, f"Class {idx}") if _nail_classes else f"Class {idx}"
    return label, conf


def _benchmark(iterations=10):
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    tests = [
        ("JAUNDICE_BODY", predict_jaundice_body_onnx),
        ("JAUNDICE_EYE", lambda img: predict_jaundice_eye_onnx(img, img)),
        ("SKIN_DISEASE", predict_skin_disease_onnx),
        ("BURNS", predict_burns_onnx),
        ("NAIL_DISEASE", predict_nail_disease_onnx)
    ]
    for name, fn in tests:
        start = time.perf_counter()
        for _ in range(iterations):
            fn(dummy)
        avg_ms = (time.perf_counter() - start) / iterations * 1000
        print(f"{name}: {avg_ms:.2f} ms per inference")


if __name__ == "__main__":
    _benchmark()
