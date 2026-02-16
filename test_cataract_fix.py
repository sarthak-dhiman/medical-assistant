import torch
import sys
import os

# Add current directory to path so we can import inference_new_models
sys.path.append(os.getcwd())

from inference_new_models import get_cataract_model, predict_cataract
import cv2
import numpy as np

def test_cataract_loading():
    print("--- Testing Cataract Model Loading ---")
    try:
        model = get_cataract_model()
        if model:
            print("SUCCESS: Cataract model loaded successfully.")
            print(f"Model type: {type(model)}")
        else:
            print("FAILURE: Cataract model returned None.")
            return
            
        # Create a dummy image to test prediction
        print("\n--- Testing Prediction ---")
        dummy_img = np.zeros((380, 380, 3), dtype=np.uint8)
        label, conf, debug = predict_cataract(dummy_img, debug=True)
        print(f"Prediction result: Label={label}, Conf={conf}")
        print("Note: Prediction on black image might be random, just checking for crashes.")
        
    except Exception as e:
        print(f"CRITICAL ERROR during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cataract_loading()
