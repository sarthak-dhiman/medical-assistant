import torch
import cv2
import numpy as np
import sys
import os
import base64

# Add current directory to path
sys.path.append(os.getcwd())

from inference_new_models import get_oral_cancer_model, predict_oral_cancer, get_cataract_model, predict_cataract

def save_base64_image(b64_str, filename):
    if not b64_str:
        print(f"Warning: No base64 data for {filename}")
        return
    
    try:
        img_data = base64.b64decode(b64_str)
        with open(filename, 'wb') as f:
            f.write(img_data)
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Failed to save {filename}: {e}")

def test_nerd_modes():
    print("--- Testing Nerd Modes (Grad-CAM) ---")
    
    # Create a dummy image (or load one if available)
    # Using specific colors to simulate features might work better, but random/black is a start for crash testing
    # Better: Try to find a real sample image
    
    oral_cancer_img_path = r"D:\Disease Prediction\Dataset\oral_cancer_classification\test\Cancer\sample.jpg" 
    cataract_img_path = r"D:\Disease Prediction\Dataset\cataract\test\Cataract\sample.jpg"
    
    # Fallback to dummy if files don't exist (using a simple gradient to check for activation)
    dummy_img = np.zeros((380, 380, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (190, 190), 50, (255, 255, 255), -1) # White circle in center
    
    # 1. Oral Cancer
    print("\n1. Testing Oral Cancer Nerd Mode...")
    if os.path.exists(oral_cancer_img_path):
        img_oral = cv2.imread(oral_cancer_img_path)
    else:
        print("Using dummy image for Oral Cancer")
        img_oral = dummy_img.copy()
        
    try:
        label, conf, debug_info = predict_oral_cancer(img_oral, debug=True)
        print(f"Prediction: {label} ({conf:.2f})")
        
        if 'grad_cam_error' in debug_info:
            print(f"Grad-CAM Error: {debug_info['grad_cam_error']}")
        elif 'grad_cam' in debug_info:
            print("Grad-CAM data present.")
            save_base64_image(debug_info['grad_cam'], "debug_oral_gradcam.jpg")
        else:
            print("No Grad-CAM key in debug_info.")
            
    except Exception as e:
        print(f"Oral Cancer Prediction Crash: {e}")
        import traceback
        traceback.print_exc()

    # 2. Cataract
    print("\n2. Testing Cataract Nerd Mode...")
    if os.path.exists(cataract_img_path):
        img_cat = cv2.imread(cataract_img_path)
    else:
        print("Using dummy image for Cataract")
        img_cat = dummy_img.copy()
        
    try:
        label, conf, debug_info = predict_cataract(img_cat, debug=True)
        print(f"Prediction: {label} ({conf:.2f})")
        
        if 'grad_cam_error' in debug_info:
            print(f"Grad-CAM Error: {debug_info['grad_cam_error']}")
        elif 'grad_cam' in debug_info:
            print("Grad-CAM data present.")
            save_base64_image(debug_info['grad_cam'], "debug_cataract_gradcam.jpg")
        else:
            print("No Grad-CAM key in debug_info.")
            
    except Exception as e:
        print(f"Cataract Prediction Crash: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    # Redirect stdout and stderr to a file
    with open("debug_nerd_modes_output.txt", "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = f
        sys.stderr = f
        
        try:
            test_nerd_modes()
        except Exception as e:
            print(f"Script crash: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            print("Output written to debug_nerd_modes_output.txt")
