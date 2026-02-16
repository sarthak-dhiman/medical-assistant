import cv2
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from inference_new_models import predict_burns

def test_skin_detection():
    print("--- Testing Burns Model Skin Detection ---")
    
    # 1. Create Non-Skin Image (Blue Square)
    img_non_skin = np.zeros((300, 300, 3), dtype=np.uint8)
    img_non_skin[:] = (255, 0, 0) # Blue
    
    print("\nTesting Non-Skin Image (Blue)...")
    label, conf, debug = predict_burns(img_non_skin, debug=True)
    print(f"Result: {label} ({conf})")
    print(f"Debug Info: {debug}")
    
    if label == "No Skin Detected":
        print("PASS: Non-skin correctly rejected.")
    else:
        print("FAIL: Non-skin was processed.")
        
    # 2. Create Skin Image (Pink/Beige)
    # HSV for skin approx: H=[0,20], S=[48,255], V=[50,255]
    # BGR for skin approx: (180, 200, 240) -> Light Pinkish
    img_skin = np.zeros((300, 300, 3), dtype=np.uint8)
    img_skin[:] = (180, 200, 240) 
    
    print("\nTesting Skin Image (Synthetic)...")
    label, conf, debug = predict_burns(img_skin, debug=True)
    print(f"Result: {label} ({conf})")
    # If model not loaded, it might return "Model Not Loaded" but skin check passes first?
    # Actually skin check is BEFORE model check in my code.
    
    if label == "No Skin Detected":
        print("FAIL: Skin image was rejected.")
    else:
        print("PASS: Skin image proceeded to model inference (or model load check).")

if __name__ == "__main__":
    test_skin_detection()
