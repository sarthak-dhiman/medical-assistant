import cv2
import json
import numpy as np
import os
import sys
from pathlib import Path

# CRITICAL: Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # FORCE CPU FOR TF/KERAS

import tensorflow as tf

# --- GPU Configuration (will see no GPUs) ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"⚠️ WARNING: Skin model TF detected {len(gpus)} GPU(s) despite CUDA_VISIBLE_DEVICES=-1")
    else:
        print("✅ Skin Disease Model correctly using CPU only")
except Exception as e:
    print(f"Skin TF GPU check: {e}")

from tensorflow import keras

# Workaround for models that include 'quantization_config' in layer configs.
try:
    from tensorflow.keras.layers import Dense as DenseLayer
except ImportError:
    from keras.layers import Dense as DenseLayer

class DenseWrapper(DenseLayer):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

from tensorflow.keras.applications.efficientnet import preprocess_input

# --- CONFIGURATION ---
MODEL_PATH_DEFAULT = "saved_models/skin_disease_model.keras"
MAPPING_PATH_DEFAULT = "saved_models/new_class_indices.json"

# --- LOAD RESOURCES (LAZY) ---
_model = None
_idx_to_class = {}

def get_model_and_mapping():
    global _model, _idx_to_class
    if _model is not None:
        return _model, _idx_to_class

    # Determine base directory
    if getattr(sys, 'frozen', False):
        BASE_DIR = Path(sys.executable).parent
    else:
        BASE_DIR = Path(__file__).parent

    print(f"DEBUG: Skin Base Dir: {BASE_DIR}")
    print(f"DEBUG: Current CWD: {os.getcwd()}")

    # Prioritize '_skin_model.keras' as requested by the user
    path_candidates = [
        BASE_DIR / "saved_models" / "_skin_model.keras",
        BASE_DIR / "saved_models" / "skin_disease_model.keras"
    ]
    
    mapping_candidates = [
        BASE_DIR / "saved_models" / "new_class_indices.json",
        BASE_DIR / "saved_models" / "class_indices.json"
    ]

    selected_model_path = None
    for p in path_candidates:
        print(f"DEBUG: Checking path: {p.absolute()}")
        if p.exists():
            selected_model_path = p
            print(f"DEBUG: Found model at {p.absolute()}")
            break
            
    selected_mapping_path = None
    for p in mapping_candidates:
        if p.exists():
            selected_mapping_path = p
            break

    if not selected_model_path:
        err_msg = f"⚠️ Error: No skin model file found in {BASE_DIR / 'saved_models'}"
        print(err_msg)
        return None, {}

    print(f"⏳ Loading skin model from {selected_model_path}...")
    try:
        # Load with custom objects
        _model = keras.models.load_model(selected_model_path, compile=False, custom_objects={"Dense": DenseWrapper})
        print("✅ Skin Model loaded successfully.")

        if selected_mapping_path:
            print(f"⏳ Loading class mapping from {selected_mapping_path}...")
            with open(selected_mapping_path, 'r') as f:
                class_indices = json.load(f)
            # Handle float/int keys if they were saved as strings in JSON
            _idx_to_class = {int(v): k for k, v in class_indices.items()}
            print(f"✅ Loaded {len(_idx_to_class)} skin classes.")
        else:
            print("⚠️ Mapping file not found for Skin Model.")
            _idx_to_class = {}
            
    except Exception as e:
        print(f"⚠️ ERROR: Could not load Skin model instance.\nError: {e}")
        import traceback
        traceback.print_exc()
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
        return "Model Not Loaded (Internal Error)", 0.0

    # 1. Resize to (224, 224) as per Colab
    img_resized = cv2.resize(img, (224, 224))
    
    # 2. Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 3. EfficientNet Preprocessing (Critical for GPU models)
    img_preprocessed = preprocess_input(img_rgb)
    
    # 4. Expand dims
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    # 5. Predict
    preds = model.predict(img_batch, verbose=0)[0]
    
    # 6. Get Top Prediction
    top_idx = np.argmax(preds)
    confidence = float(preds[top_idx])
    label = idx_to_class.get(top_idx, f"Unknown ({top_idx})")
    
    # Map internal training name to UI friendly name
    if label == "Unknown_Normal":
        label = "Healthy"
    
    return label, confidence

if __name__ == "__main__":
    test_path = "test.jpg"
    if Path(test_path).exists():
        img = cv2.imread(test_path)
        label, conf = predict_skin_disease(img)
        print(f"Prediction: {label} ({conf*100:.2f}%)")
    else:
        print("Usage: python inference_skin.py")
