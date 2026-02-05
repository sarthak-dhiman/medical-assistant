import sys
import os
import cv2
import numpy as np
import base64
from pathlib import Path
from .celery_app import celery_app
from .config import PROJECT_ROOT

# Ensure project root is in path to import original scripts
sys.path.append(str(PROJECT_ROOT))

# Lazy Global Models
seg_model = None
jaundice_eye_model = None
jaundice_skin_model = None
skin_disease_model = None
init_errors = [] # Persistent global errors

def load_models():
    """Loads models once per worker process."""
    global seg_model, jaundice_eye_model, jaundice_skin_model, skin_disease_model, init_errors
    
    # Do not clear global errors if they exist, but only try loading missing models
    temp_errors = []
    
    if seg_model is None:
        try:
            from segformer_utils import SegFormerWrapper
            print("Loading SegFormer...")
            seg_model = SegFormerWrapper()
        except Exception as e:
            err_msg = f"Failed to load SegFormer: {e}"
            print(err_msg)
            temp_errors.append(err_msg)
            
    if jaundice_eye_model is None:
        try:
            from inference_pytorch import predict_jaundice, get_model
            print("Warming Jaundice Eye Model (PyTorch)...")
            # Load the raw model to verify it exists and is on GPU
            raw_model = get_model()
            if raw_model is None:
                raise Exception("PyTorch Model File Missing")
            
            # Store the WRAPPER function for prediction
            jaundice_eye_model = predict_jaundice 
            
            # Warm up with dummy data
            print("Warming up PyTorch model...")
            jaundice_eye_model(np.zeros((380,380,3), dtype=np.uint8), np.zeros((64,64,3), dtype=np.uint8))
            
            print("✅ Jaundice Eye Model Ready (PyTorch on GPU)", flush=True) 
        except Exception as e:
            err_msg = f"Failed to load Jaundice Eye Logic: {e}"
            print(err_msg)
            errors.append(err_msg)

    if jaundice_skin_model is None:
        try:
            from inference import predict_frame
            print("Warming Jaundice Skin Model (Keras)...")
            # Test load
            try:
                test_result = predict_frame(np.zeros((224, 224, 3), dtype=np.uint8))
                jaundice_skin_model = predict_frame
                print(f"✅ Jaundice Skin Model Ready: {test_result}")
            except Exception as load_err:
                print(f"⚠️ Jaundice Skin Model has architecture mismatch: {load_err}")
                def jaundice_model_error(img):
                    return "Model Architecture Mismatch (Needs Retraining)", 0.0
                jaundice_skin_model = jaundice_model_error
        except Exception as e:
            err_msg = f"Failed to load Jaundice Skin Logic: {e}"
            print(err_msg)
            temp_errors.append(err_msg)

    if skin_disease_model is None:
        try:
            from inference_skin import predict_skin_disease
            print("Warming Skin Disease Model (Keras)...")
            # Test load - if this fails, catch it gracefully
            try:
                test_result = predict_skin_disease(np.zeros((224, 224, 3), dtype=np.uint8))
                skin_disease_model = predict_skin_disease
                print(f"✅ Skin Disease Model Ready: {test_result}")
            except Exception as load_err:
                print(f"⚠️ Skin Disease Model has architecture mismatch: {load_err}")
                print("   This model needs to be retrained with RGB input (3 channels)")
                # Set to a dummy function that returns error
                def skin_model_error(img):
                    return "Model Architecture Mismatch (Needs Retraining)", 0.0
                skin_disease_model = skin_model_error
        except Exception as e:
            err_msg = f"Failed to load Skin Disease Logic: {e}"
            print(err_msg)
            temp_errors.append(err_msg)
            
    # Append new errors to global persistent list
    for err in temp_errors:
        if err not in init_errors:
            init_errors.append(err)

    return temp_errors

from celery.signals import worker_process_init

@worker_process_init.connect
def init_models(**kwargs):
    """
    EAGER LOADING:
    Load all models immediately when the worker process starts.
    This prevents the 'First Request Timeout' issue.
    """
    print("🚀 WORKER INIT: Eagerly loading AI Models...", flush=True)
    errors = load_models()
    if errors:
        print(f"❌ WORKER INIT FAILED: {errors}", flush=True)
    else:
        print("✅ WORKER INIT SUCCESS: All models ready!", flush=True)

@celery_app.task(bind=True)
def check_model_health(self):
    """
    Simple probe to check if models are loaded in the worker.
    Returns: dict of model statuses
    """
    global seg_model, jaundice_eye_model, jaundice_skin_model, skin_disease_model
    status = {
        "seg_model": seg_model is not None,
        "jaundice_eye_model": jaundice_eye_model is not None,
        "jaundice_skin_model": jaundice_skin_model is not None,
        "skin_disease_model": skin_disease_model is not None,
        "config": {
           "device": os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"),
           "omp_threads": os.environ.get("OMP_NUM_THREADS", "Not Set")
        }
    }
    
    # Check if any errors occurred during load
    # (Re-running load_models returns accumulated errors list)
    # This is safe because load_models checks for None before re-loading.
    init_errors = load_models()
    if init_errors:
       status["errors"] = init_errors

    # Return True only if ALL critical models are loaded
    status["ready"] = all([status[k] for k in ["seg_model", "jaundice_eye_model", "jaundice_skin_model", "skin_disease_model"]])
    return status

@celery_app.task(bind=True)
def predict_task(self, image_data_b64, mode):
    # FAST FAIL: Check if models loaded correctly during init
    if mode == "JAUNDICE_EYE" and jaundice_eye_model is None:
         return {"status": "error", "error": "Jaundice Eye Model Not Ready (Check Server Logs)"}
    if mode == "JAUNDICE_BODY" and (seg_model is None or jaundice_skin_model is None):
         return {"status": "error", "error": "Jaundice Body Model Not Ready (Check Server Logs)"}
    if mode == "SKIN_DISEASE" and (seg_model is None or skin_disease_model is None):
         return {"status": "error", "error": "Skin Disease Model Not Ready (Check Server Logs)"}
    
    # Decode Image
    try:
        if "," in image_data_b64:
            image_data_b64 = image_data_b64.split(",")[1]
        
        nparr = np.frombuffer(base64.b64decode(image_data_b64), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"status": "error", "error": "Invalid Image"}
            
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Logic from webcam_app.py (Simplified for API)
    result = {"status": "success", "mode": mode}
    
    h, w = frame.shape[:2]
    
    # 1. Segmentation (Needed for Jaundice)
    if mode in ["JAUNDICE_BODY", "JAUNDICE_EYE", "SKIN_DISEASE"]:
        # Optimization: Downscale for SegFormer
        INFERENCE_WIDTH = 480
        scale = INFERENCE_WIDTH / w
        if scale < 1.0:
            small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        else:
            small_frame = frame
            
        mask_small = seg_model.predict(small_frame)
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Skin Mask
        skin_mask = seg_model.get_skin_mask(mask)
        
        # --- ENCODE MASK FOR FRONTEND ---
        
        if mode == "JAUNDICE_BODY":
             # Detect Jaundice on Body Skin
             if cv2.countNonZero(skin_mask) > 100:
                  # Crop bounding box of skin
                  y, x = np.where(skin_mask > 0)
                  y0, y1, x0, x1 = y.min(), y.max(), x.min(), x.max()
                  
                  # Crop from ORIGINAL frame (not masked_roi) to preserve BGR channels
                  crop = frame[y0:y1, x0:x1]
                  
                  # Ensure it's 3-channel BGR
                  if len(crop.shape) == 2:
                      crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                  elif crop.shape[2] != 3:
                      crop = crop[:, :, :3]
                  
                  # Skin Jaundice stays on TensorFlow
                  label, conf = jaundice_skin_model(crop)
                  result.update({
                      "label": label, 
                      "confidence": float(conf), 
                      "bbox": [x0/w, y0/h, x1/w, y1/h] # Normalized
                  })
             else:
                  result.update({"label": "No Skin", "confidence": 0.0})

        elif mode == "JAUNDICE_EYE":
            # 1. Extract Skin Crop (Context for Model)
            skin_crop = frame # Fallback
            if cv2.countNonZero(skin_mask) > 100:
                 y, x = np.where(skin_mask > 0)
                 y0, y1, x0, x1 = y.min(), y.max(), x.min(), x.max()
                 masked_skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
                 skin_crop = masked_skin[y0:y1, x0:x1]

            eyes = seg_model.get_eye_rois(mask, frame)
            found_eyes = []
            
            any_jaundice = False
            
            for cropped_eye, name, (x1, y1, x2, y2) in eyes:
                # Apply Iris Masking to remove colored iris (which confuses the model)
                cropped_eye_masked, _ = seg_model.apply_iris_mask(cropped_eye)
                
                # Jaundice Eye detection uses PyTorch
                # Ensure skin_crop is valid
                if skin_crop is None or skin_crop.size == 0:
                    skin_crop = frame 

                # Jaundice Eye detection uses PyTorch Wrapper
                label, conf = jaundice_eye_model(skin_crop, cropped_eye_masked)
                
                if label == "Jaundice":
                    any_jaundice = True
                    
                eye_result = {
                    "name": name, 
                    "label": label, 
                    "confidence": float(conf),
                    "bbox": [x1/w, y1/h, x2/w, y2/h], # Normalized
                    "method": "hough" # Default
                }
                
                found_eyes.append(eye_result)
            
            result["eyes"] = found_eyes
            
            # Aggregate Result for UI
            if not found_eyes:
                result["label"] = "No Eyes Detected"
                result["confidence"] = 0.0
            elif any_jaundice:
                result["label"] = "Jaundice"
                result["confidence"] = max([e["confidence"] for e in found_eyes if e["label"] == "Jaundice"])
            else:
                result["label"] = "Normal"
                result["confidence"] = max([e["confidence"] for e in found_eyes])
            
        elif mode == "SKIN_DISEASE":
             # Detect on Skin
             if cv2.countNonZero(skin_mask) > 100:
                  y, x = np.where(skin_mask > 0)
                  y0, y1, x0, x1 = y.min(), y.max(), x.min(), x.max()
                  
                  # Crop from ORIGINAL frame to preserve BGR channels
                  crop = frame[y0:y1, x0:x1]
                  
                  # Ensure it's 3-channel BGR
                  if len(crop.shape) == 2:
                      crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                  elif crop.shape[2] != 3:
                      crop = crop[:, :, :3]
                  
                  label, conf = skin_disease_model(crop)
                  result.update({
                      "label": label, 
                      "confidence": float(conf), 
                      "bbox": [x0/w, y0/h, x1/w, y1/h] # Normalized
                  })
             else:
                  result.update({"label": "No Skin", "confidence": 0.0})

    return result
