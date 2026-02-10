import onnxruntime as ort
import numpy as np
import cv2
import sys
import os
import json
import base64

# constants
IMG_SIZE = (380, 380)
SCLERA_SIZE = (64, 64)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

sessions = {}

def get_model_path(name):
    # PyInstaller puts files in sys._MEIPASS
    base_dir = getattr(sys, '_MEIPASS', os.getcwd())
    
    # Check bundled path (models/)
    path = os.path.join(base_dir, 'models', name)
    if os.path.exists(path): return path
    
    # Check dev path (saved_models/onnx/)
    # Assuming run from project root 'd:/Disease Prediction'
    dev_path = os.path.join(base_dir, 'saved_models', 'onnx', name)
    if os.path.exists(dev_path): return dev_path
    
    print(f"Warning: Model {name} not found in {path} or {dev_path}")
    return path

def get_session(name):
    if name in sessions: return sessions[name]
    path = get_model_path(name)
    try:
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        sessions[name] = sess
        return sess
    except Exception as e:
        print(f"Failed to load ONNX model {name}: {e}")
        return None

def preprocess(img_bgr, size):
    img = cv2.resize(img_bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0).astype(np.float32)

def predict_body(frame, debug_info):
    sess = get_session('jaundice_body.onnx')
    if not sess: return {"error": "Model Missing"}
    
    inp = preprocess(frame, IMG_SIZE)
    # Output name 'output' from export script
    logits = sess.run(['output'], {'input': inp})[0]
    
    prob = 1.0 / (1.0 + np.exp(-logits)) # Sigmoid
    prob = float(prob.item())
    
    threshold = 0.70
    label = "Jaundice" if prob > threshold else "Normal"
    conf = prob if prob > threshold else 1.0 - prob
    
    return {"label": label, "confidence": conf, "debug_info": debug_info}

def predict_eye(frame, debug_info):
    sess = get_session('jaundice_eye.onnx')
    if not sess: return {"error": "Model Missing"}
    
    # TODO: Proper Eye Segmentation (requires SegFormer ONNX or MediaPipe Python)
    # For now, simplistic approach: Center Crop as Sclera (Fallback)
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    # Assume frame IS the eye crop passed from frontend via webcam
    # But wait, frontend passes full frame usually?
    # In web app, frontend passes full frame, backend crops.
    # Here, let's assume frame is what we want to process.
    
    # Actually, Jaundice Eye model requires Skin Context + Sclera Crop.
    # In Desktop App, implementing full pipeline (Face Mesh -> Eye Crop) without MediaPipe/SegFormer is hard.
    # Simplification: Use Center Crop as Sclera, Full Frame as Skin Context.
    
    sclera_crop = cv2.resize(frame, SCLERA_SIZE) # Naive
    
    skin_inp = preprocess(frame, IMG_SIZE)
    sclera_inp = preprocess(sclera_crop, SCLERA_SIZE)
    
    # Inputs: skin_input, sclera_input
    logits = sess.run(['output'], {'skin_input': skin_inp, 'sclera_input': sclera_inp})[0]
    
    prob = 1.0 / (1.0 + np.exp(-logits))
    prob = float(prob.item())
    
    threshold = 0.5
    label = "Jaundice" if prob > threshold else "Normal"
    conf = prob if prob > threshold else 1.0 - prob
    
    return {"label": label, "confidence": conf, "debug_info": debug_info}

SKIN_CLASSES = {}

def load_skin_mapping():
    global SKIN_CLASSES
    if SKIN_CLASSES: return
    
    base_dir = getattr(sys, '_MEIPASS', os.getcwd())
    # Try bundled path first (models/...)
    bundled_path = os.path.join(base_dir, 'models', 'skin_disease_mapping.json')
    # Try dev path (saved_models/...)
    dev_path = os.path.join(base_dir, 'saved_models', 'skin_disease_mapping.json')
    
    path = bundled_path if os.path.exists(bundled_path) else dev_path
    
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                SKIN_CLASSES = json.load(f)
        except Exception as e:
            print(f"Failed to load mapping: {e}")

def predict_skin(frame, debug_info):
    sess = get_session('skin_disease.onnx')
    if not sess: return {"error": "Model Missing"}
    
    inp = preprocess(frame, IMG_SIZE)
    logits = sess.run(['output'], {'input': inp})[0]
    
    # Softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    probs = probs[0]
    
    idx = np.argmax(probs)
    conf = float(probs[idx])
    
    # Load mapping
    load_skin_mapping()
    # JSON keys result in strings usually
    label = SKIN_CLASSES.get(str(idx), f"Class {idx}") 
    
    # Top 3
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [{"label": SKIN_CLASSES.get(str(i), f"Class {i}"), "confidence": float(probs[i])} for i in top3_idx]
    debug_info["top_3"] = top3
    
    return {"label": label, "confidence": conf, "debug_info": debug_info}

def predict_image(image_data_b64, mode, debug=False):
    # Decode
    try:
        if "," in image_data_b64:
             _, image_data_b64 = image_data_b64.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(image_data_b64), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return {"status": "error", "error": "Decode Failed"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

    debug_info = {}
    if debug:
        # Simple Color Stats
        mean_bgr = np.mean(frame, axis=(0,1))
        mean_rgb = list(mean_bgr[::-1])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_hsv = list(np.mean(hsv, axis=(0,1)))
        
        debug_info["color_stats"] = {
            "mean_rgb": [int(x) for x in mean_rgb],
            "mean_hsv": [int(x) for x in mean_hsv]
        }
        # Masks: Not available in ONNX-only mode yet
        
    try:
        if mode == "JAUNDICE_BODY":
             return predict_body(frame, debug_info)
        elif mode == "JAUNDICE_EYE":
             return predict_eye(frame, debug_info)
        elif mode == "SKIN_DISEASE":
             return predict_skin(frame, debug_info)
        return {"status": "error", "error": f"Unknown Mode: {mode}"}
    except Exception as e:
        return {"status": "error", "error": f"Inference Failed: {e}"}
