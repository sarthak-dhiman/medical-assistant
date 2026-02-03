import cv2
import json
import numpy as np
import os
import sys
from pathlib import Path

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- GPU Configuration ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU Detected: {len(gpus)} device(s). Inference (Skin) will use GPU.")
        except RuntimeError as e:
            print(f"GPU config error: {e}")
    else:
        print("⚠️ No GPU detected. Inference (Skin) will run on CPU.")
except Exception as e:
    print(f"Error checking GPU: {e}")

from tensorflow import keras

# Workaround for models that include 'quantization_config' in layer configs.
try:
    from tensorflow.keras.layers import Dense as DenseLayer
except ImportError:
    from keras.layers import Dense as DenseLayer

class DenseWrapper(DenseLayer):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

# --- CONFIGURATION ---
MODEL_PATH = Path("saved_models/_skin_model.keras")
MAPPING_PATH = Path("saved_models/class_indices.json")

# --- LOAD RESOURCES (LAZY) ---
_model = None
_idx_to_class = {}

def get_model_and_mapping():
    global _model, _idx_to_class
    if _model is not None:
        return _model, _idx_to_class

    MODEL_PATH = Path("saved_models/_skin_model.keras")
    MAPPING_PATH = Path("saved_models/new_class_indices.json")

    # Fallback paths
    if getattr(sys, 'frozen', False):
        BASE_DIR = Path(sys.executable).parent
    else:
        BASE_DIR = Path(__file__).parent

    # Try explicit path from Base Dir
    MODEL_PATH = BASE_DIR / "saved_models" / "_skin_model.keras"
    MAPPING_PATH = BASE_DIR / "saved_models" / "class_indices.json"

    print(f"⏳ Loading model from {MODEL_PATH}...")
    try:
        if not MODEL_PATH.exists():
            print(f"⚠️ Error: Model file not found at {MODEL_PATH}")
            return None, {}

        # Load with custom objects
        _model = keras.models.load_model(MODEL_PATH, compile=False, custom_objects={"Dense": DenseWrapper})
        print("✅ Model loaded.")

        if MAPPING_PATH.exists():
            print(f"⏳ Loading class mapping from {MAPPING_PATH}...")
            with open(MAPPING_PATH, 'r') as f:
                class_indices = json.load(f)
            _idx_to_class = {v: k for k, v in class_indices.items()}
            print(f"✅ Loaded {len(_idx_to_class)} classes.")
        else:
            print("⚠️ Mapping file not found. Using numeric labels.")
            _idx_to_class = {}
            
    except Exception as e:
        print(f"⚠️ Warning: Could not load model or mapping. Ensure you have trained the model first.\nError: {e}")
        _model = None
        _idx_to_class = {}

    return _model, _idx_to_class

def predict_skin_disease(img):
    """
    Predicts skin disease from an image.
    Returns: (predicted_class_name, confidence_score)
    """
    model, idx_to_class = get_model_and_mapping()
    
    if model is None:
        return "Model Not Loaded", 0.0

    # 1. Resize to (224, 224)
    img_resized = cv2.resize(img, (224, 224))
    
    # 2. Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 3. Rescale (1/255) - Must match training!
    # img_scaled = img_rgb.astype(np.float32) / 255.0 # DISABLED
    img_scaled = img_rgb
    
    # 4. Expand dims
    img_batch = np.expand_dims(img_scaled, axis=0)
    
    # 5. Predict
    preds = model.predict(img_batch, verbose=0)[0]
    
    # 6. Get Top Prediction
    top_idx = np.argmax(preds)
    confidence = preds[top_idx]
    label = idx_to_class.get(top_idx, "Unknown")
    
    return label, confidence

if __name__ == "__main__":
    test_path = "test.jpg"
    if Path(test_path).exists():
        img = cv2.imread(test_path)
        label, conf = predict_skin_disease(img)
        print(f"Prediction: {label} ({conf*100:.2f}%)")
    else:
        print("Usage: python inference_skin.py")
        print("Ensure 'saved_models/skin_disease_model.keras' exists.")
