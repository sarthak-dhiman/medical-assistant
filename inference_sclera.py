import cv2
import numpy as np
import os
import sys
from pathlib import Path

# --- LAZY LOADING SETUP ---
_model = None

# Monkey-patching Helper for Keras Version Compatibility
def get_custom_objects():
    custom_objects = {}
    try:
        import keras.layers
        import tensorflow.keras.layers as tf_layers
        
        targets = [keras.layers, tf_layers]
        for target in targets:
            if hasattr(target, 'Dense'):
                OriginalDense = target.Dense
                class DenseWrapper(OriginalDense):
                    def __init__(self, *args, quantization_config=None, **kwargs):
                        super().__init__(*args, **kwargs)
                custom_objects['Dense'] = DenseWrapper
    except Exception:
        pass
    return custom_objects

def get_sclera_model():
    global _model
    if _model is not None:
        return _model

    # Determine Base Directory
    if getattr(sys, 'frozen', False):
        BASE_DIR = Path(sys.executable).parent
    else:
        BASE_DIR = Path(__file__).parent

    MODEL_PATH = BASE_DIR / "saved_models" / "jaundice_with_sclera.keras"

    print(f"Loading Sclera Model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return None

    try:
        import keras
        # Import necessary layers for deserialization if custom objects alone aren't enough
        # But 'load_model' is usually smart. We add the DenseWrapper just in case.
        custom_objs = get_custom_objects()
        
        # Load the model
        _model = keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objs)
        print("Sclera Model loaded successfully.")
        return _model
    except Exception as e:
        print(f"Error loading Sclera Model: {e}")
        return None

def predict_sclera_jaundice(skin_img, sclera_img):
    """
    Predicts Jaundice using the dual-input model.
    Inputs:
        skin_img: BGR image (Face/Skin crop)
        sclera_img: BGR image (Eye/Sclera crop)
    Returns:
        (label, confidence)
    """
    model = get_sclera_model()
    if model is None:
        return "Model Missing", 0.0

    if skin_img is None or sclera_img is None:
        return "Invalid Input", 0.0

    # --- Preprocessing ---
    # 1. Skin Branch: (224, 224, 3)
    skin_resized = cv2.resize(skin_img, (224, 224))
    skin_rgb = cv2.cvtColor(skin_resized, cv2.COLOR_BGR2RGB)
    
    # 2. Sclera Branch: (64, 64, 3) - Matches make_model in train script
    sclera_resized = cv2.resize(sclera_img, (64, 64))
    sclera_rgb = cv2.cvtColor(sclera_resized, cv2.COLOR_BGR2RGB)
    
    # 3. Normalization (1/255.0) as per load_image in train script
    skin_norm = skin_rgb.astype('float32') / 255.0
    sclera_norm = sclera_rgb.astype('float32') / 255.0
    
    # 4. Batch expansion
    skin_batch = np.expand_dims(skin_norm, axis=0)     # (1, 224, 224, 3)
    sclera_batch = np.expand_dims(sclera_norm, axis=0) # (1, 64, 64, 3)
    
    # 5. Predict
    # Model inputs map key names usually: {'skin_input': ..., 'sclera_input': ...}
    # Or list: [skin_input, sclera_input]
    # We'll try dictionary first as it's safer if named, or list if that fails.
    
    try:
        prediction = model.predict(
            {'skin_input': skin_batch, 'sclera_input': sclera_batch}, 
            verbose=0
        )[0]
    except Exception:
        # Fallback to list order (Skin first, Sclera second)
        prediction = model.predict([skin_batch, sclera_batch], verbose=0)[0]

    score = float(prediction[0])
    
    # Assuming Binary Sigmoid (0=Jaundice, 1=Normal based on generator logic)
    # logic in train script: labels.append(1 if ... 'normal' ... else 0)
    # So 0 = Jaundice, 1 = Normal
    
    if score < 0.5:
        return "Jaundice", (1 - score) # High confidence if close to 0
    else:
        return "Normal", score # High confidence if close to 1
