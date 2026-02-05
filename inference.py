import cv2
import numpy as np
import os
import sys
from pathlib import Path

# CRITICAL: Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduce TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # FORCE CPU FOR TF/KERAS

import tensorflow as tf

# --- GPU Configuration (will see no GPUs due to CUDA_VISIBLE_DEVICES=-1) ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"⚠️ WARNING: TensorFlow detected {len(gpus)} GPU(s) despite CUDA_VISIBLE_DEVICES=-1")
    else:
        print("✅ TensorFlow correctly using CPU only (GPU reserved for PyTorch)")
except Exception as e:
    print(f"TF GPU check: {e}")

keras = tf.keras
# --- Configuration ---
if getattr(sys, 'frozen', False):
    # PyInstaller: Use executable's directory
    BASE_DIR = Path(sys.executable).parent
else:
    # Dev: Use script's directory
    BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "saved_models" / "jaundice_model.keras"

_model = None

def get_model():
    global _model
    if _model is not None:
        return _model

    # Monkey-patching Helper / Wrapper
    def get_custom_objects():
        custom_objects = {}
        try:
            import keras.layers
            import tensorflow.keras.layers as tf_layers
            bases = [getattr(keras.layers, 'Dense', None), getattr(tf_layers, 'Dense', None)]
            bases = [b for b in bases if b is not None]
            if bases:
                OriginalDense = bases[0]
                class DenseWrapper(OriginalDense):
                    def __init__(self, *args, quantization_config=None, **kwargs):
                        super().__init__(*args, **kwargs)
                custom_objects['Dense'] = DenseWrapper
                custom_objects['keras.layers.Dense'] = DenseWrapper
        except Exception as e:
            pass
        return custom_objects

    print(f"Loading model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None

    try:
        custom_objs = get_custom_objects()
        _model = keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objs)
        print("Model loaded successfully.")
        return _model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# --- Preprocessing Function ---
def white_balance_preprocessing(img_rgb):
    """
    Applies Gray World assumption to normalize lighting using LAB color space.
    Input: RGB Image (H, W, 3)
    Output: RGB Image (H, W, 3)
    """
    # 1. Convert RGB to LAB
    result = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # 2. Calculate Averages of A and B channels
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    
    # 3. Adjust A and B channels based on Luminance (L channel)
    # logic: shift color center towards 128 (gray) based on brightness
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    
    # 4. Convert back to RGB
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result

def predict_frame(img_bgr):
    """
    Takes an image (numpy array from cv2 in BGR format), preprocesses it, and returns prediction.
    """
    model = get_model()
    if model is None:
        return "Error (Model Failed)", 0.0

    if img_bgr is None:
        return "Error", 0.0

    # 1. Resize to (224, 224) as required by EfficientNetB0
    img_resized = cv2.resize(img_bgr, (224, 224))
    
    # 2. Convert BGR (OpenCV default) to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 3. Apply White Balance (Optional)
    # img_wb = white_balance_preprocessing(img_rgb)
    img_wb = img_rgb
    
    # 4. EfficientNet Preprocessing
    # Assuming the original model expects standard inputs (maybe 0-255 or normalized)
    # We'll stick to basic RGB as it was before my changes
    
    # 5. Expand dimensions to match batch shape (1, 224, 224, 3)
    img_batch = np.expand_dims(img_wb, axis=0)
    
    # 6. Predict
    prediction = model.predict(img_batch, verbose=0)[0]
    
    # Handle Prediction Output
    if len(prediction) == 1:
        # Binary Classification
        score = float(prediction[0])
        if score < 0.5:
            label = "Jaundice"
            confidence = 1 - score
        else:
            label = "Normal"
            confidence = score
    else:
        # Categorical
        idx = np.argmax(prediction)
        confidence = float(prediction[idx])
        label = "Jaundice" if idx == 0 else "Normal"
        
    return label, confidence

if __name__ == "__main__":
    # --- Test Usage ---
    test_image_path = BASE_DIR / "Test files" / "test_1.jpg"
    if test_image_path.exists():
        print(f"Testing on: {test_image_path}")
        img = cv2.imread(str(test_image_path))
        label, confidence = predict_frame(img)
        print(f"Prediction: {label} ({confidence*100:.2f}%)")