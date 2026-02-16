import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from inference_new_models import get_cataract_model, predict_cataract
import cv2
import numpy as np
import threading

def verify_cataract():
    print("--- Verifying Cataract Model Fix ---")
    try:
        # 1. Test Load
        print("1. Calling get_cataract_model()...")
        model = get_cataract_model()
        
        if model is None:
            print("FAILURE: Model returned None")
            return
            
        print(f"SUCCESS: Model loaded. Type: {type(model)}")
        print(f"Architecture check: {model}")
        
        # 2. Test Prediction (Run inference)
        print("\n2. Testing Prediction...")
        dummy_img = np.zeros((380, 380, 3), dtype=np.uint8)
        
        # Simple prediction
        label, conf, debug = predict_cataract(dummy_img, debug=False)
        print(f"Prediction (No Debug): Label={label}, Conf={conf}")
        
        # Debug prediction (Grad-CAM)
        print("\n3. Testing Grad-CAM (Debug Mode)...")
        label, conf, debug = predict_cataract(dummy_img, debug=True)
        print(f"Prediction (Debug): Label={label}, Conf={conf}")
        if 'grad_cam_error' in debug:
            print(f"Grad-CAM Error: {debug['grad_cam_error']}")
        elif 'grad_cam' in debug:
            print("Grad-CAM generated successfully (base64 string present)")
        else:
            print("Grad-CAM not generated (no error, but key missing?)")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_cataract()
