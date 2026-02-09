import sys
import os
import cv2
import numpy as np
import base64
import logging
from pathlib import Path
from .celery_app import celery_app
from .config import settings

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# Ensure project root is in path to import original scripts
# (Using settings.PROJECT_ROOT if available, else relative)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Lazy Global Models
seg_model = None
jaundice_eye_model = None
jaundice_skin_model = None
skin_disease_model = None
init_errors = [] # Persistent global errors

def load_models():
    """Loads models once per worker process (Unified PyTorch Engine)."""
    global seg_model, jaundice_eye_model, jaundice_skin_model, skin_disease_model, init_errors
    
    temp_errors = []
    
    # 0. SegFormer (Segmentation)
    if seg_model is None:
        try:
            from segformer_utils import SegFormerWrapper
            logger.info("Loading SegFormer...")
            seg_model = SegFormerWrapper()
        except Exception as e:
            err_msg = f"Failed to load SegFormer: {e}"
            logger.error(err_msg)
            temp_errors.append(err_msg)
            
    # 1. Jaundice Eye Model (PyTorch)
    if jaundice_eye_model is None:
        try:
            from inference_pytorch import predict_jaundice_eye
            logger.info("Warming Jaundice Eye Model (PyTorch)...")
            # Warm up with actual inference to initialize CUDA kernels
            dummy_skin = np.zeros((380,380,3), dtype=np.uint8)
            dummy_sclera = np.zeros((64,64,3), dtype=np.uint8)
            for _ in range(3):  # Run 3 times to ensure JIT compilation
                predict_jaundice_eye(dummy_skin, dummy_sclera)
            jaundice_eye_model = predict_jaundice_eye
            logger.info("✅ Jaundice Eye Model Ready") 
        except Exception as e:
            err_msg = f"Failed to load Jaundice Eye: {e}"
            logger.error(err_msg)
            temp_errors.append(err_msg)
            
    # 2. Jaundice Body Model (PyTorch) - Replaces Keras
    if jaundice_skin_model is None:
        try:
            from inference_pytorch import predict_jaundice_body
            logger.info("Warming Jaundice Body Model (PyTorch)...")
            dummy_img = np.zeros((380, 380, 3), dtype=np.uint8)
            for _ in range(3):  # Run 3 times to ensure JIT compilation
                predict_jaundice_body(dummy_img)
            jaundice_skin_model = predict_jaundice_body
            logger.info("✅ Jaundice Body Model Ready")
        except Exception as e:
            err_msg = f"Failed to load Jaundice Body: {e}"
            logger.error(err_msg)
            temp_errors.append(err_msg)

    # 3. Skin Disease Model (PyTorch) - Replaces Keras
    if skin_disease_model is None:
        try:
            from inference_pytorch import predict_skin_disease_torch
            logger.info("Warming Skin Disease Model (PyTorch)...")
            dummy_img = np.zeros((380, 380, 3), dtype=np.uint8)
            for _ in range(3):  # Run 3 times to ensure JIT compilation
                predict_skin_disease_torch(dummy_img)
            skin_disease_model = predict_skin_disease_torch
            logger.info("✅ Skin Disease Model Ready")
        except Exception as e:
            err_msg = f"Failed to load Skin Disease: {e}"
            logger.error(err_msg)
            temp_errors.append(err_msg)
            
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
    logger.info("🚀 WORKER INIT: Eagerly loading AI Models...")
    errors = load_models()
    if errors:
        logger.error(f"❌ WORKER INIT FAILED: {errors}")
    else:
        logger.info("✅ WORKER INIT SUCCESS: All models ready!")

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
    init_errors = load_models()
    if init_errors:
       status["errors"] = init_errors

    # Return True only if ALL critical models are loaded
    status["ready"] = all([status[k] for k in ["seg_model", "jaundice_eye_model", "jaundice_skin_model", "skin_disease_model"]])
    return status

@celery_app.task(bind=True)
def predict_task(self, image_data_b64, mode):
    # FAST FAIL: Check if models loaded correctly during init
    if mode == "JAUNDICE_EYE":
        if jaundice_eye_model is None:
             return {"status": "error", "error": "Jaundice Eye Model Not Ready"}
        if not seg_model or not seg_model.is_ready:
             return {"status": "error", "error": "SegFormer Not Ready (Check Logs)"}

    if mode == "JAUNDICE_BODY":
        if jaundice_skin_model is None:
             return {"status": "error", "error": "Jaundice Body Model Not Ready"}
        if not seg_model or not seg_model.is_ready:
             return {"status": "error", "error": "SegFormer Not Ready (Check Logs)"}

    if mode == "SKIN_DISEASE":
        if skin_disease_model is None:
             return {"status": "error", "error": "Skin Disease Model Not Ready"}
        if not seg_model or not seg_model.is_ready:
             return {"status": "error", "error": "SegFormer Not Ready (Check Logs)"}
    
    # Decode Image
    try:
        if "," in image_data_b64:
            header, image_data_b64 = image_data_b64.split(",", 1)
            # Basic header validation
            if "image/" not in header:
                 logger.warning(f"Suspicious image header: {header}")
        
        # Safe decoding
        image_bytes = base64.b64decode(image_data_b64, validate=True)
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode image from bytes")

    except Exception as e:
        logger.error(f"Image Decoding Failed: {e}")
        return {"status": "error", "error": "Invalid Image Data"}

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
        
        # Skin Mask (Semantic)
        skin_mask = seg_model.get_skin_mask(mask)
        
        # Fallback: If Semantic Segmentation (Face/Neck) found nothing, 
        # try Color-Based Skin Detection (for Arms/Legs/Hands)
        if cv2.countNonZero(skin_mask) == 0:
            skin_mask = seg_model.get_skin_mask_color(frame)
        
        # --- ENCODE MASK FOR FRONTEND ---
        
        if mode == "JAUNDICE_BODY":
             # Detect Jaundice on Body Skin
             
             # 1. Get Person Mask (MediaPipe)
             try:
                 person_mask = seg_model.get_body_segmentation(frame)
             except Exception:
                 person_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
                 
             # 2. Combine with Skin Color Mask
             # Skin on Person = Person Mask AND Skin Mask
             combined_mask = cv2.bitwise_and(skin_mask, person_mask)
             
             if cv2.countNonZero(combined_mask) > 100:
                  # Crop bounding box of skin
                  y, x = np.where(combined_mask > 0)
                  y0, y1, x0, x1 = y.min(), y.max(), x.min(), x.max()
                  
                  # Crop from ORIGINAL frame
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
                  result.update({"label": "No Skin on Body", "confidence": 0.0})

        elif mode == "JAUNDICE_EYE":
            # 1. Extract Skin Crop (Context for Model)
            skin_crop = frame # Fallback
            if cv2.countNonZero(skin_mask) > 100:
                 y, x = np.where(skin_mask > 0)
                 y0, y1, x0, x1 = y.min(), y.max(), x.min(), x.max()
                 masked_skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
                 skin_crop = masked_skin[y0:y1, x0:x1]

            # 2. Extract Eyes (Priority: MediaPipe -> Fallback: SegFormer)
            used_mediapipe = True
            eyes = seg_model.get_eyes_mediapipe(frame)
            
            if not eyes:
                 used_mediapipe = False
                 # Fallback
                 # print("⚠️ MediaPipe failed. Using SegFormer fallback.", flush=True)
                 raw_eyes = seg_model.get_eye_rois(mask, frame)
                 for cropped_eye, name, bbox in raw_eyes:
                      masked, _ = seg_model.apply_iris_mask(cropped_eye)
                      eyes.append((masked, name, bbox))

            found_eyes = []
            
            any_jaundice = False
            
            for cropped_eye_masked, name, (x1, y1, x2, y2) in eyes:
                # Note: MediaPipe returns ALREADY masked eyes (Sclera only).
                # SegFormer fallback also returns masked eyes.
                
                # ERROR: cv2.imencode requires BGR, assume cropped_eye is BGR
                # Encode for Nerd Mode (Debug View - VISUALIZE WHAT MODEL SEES)
                try:
                    _, buffer = cv2.imencode('.jpg', cropped_eye_masked)
                    eye_b64 = base64.b64encode(buffer).decode('utf-8')
                    debug_img_str = f"data:image/jpeg;base64,{eye_b64}"
                except Exception:
                    debug_img_str = ""

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
                    "method": "mediapipe" if used_mediapipe else "hough",
                    "debug_image": debug_img_str # Store per eye
                }
                
                found_eyes.append(eye_result)
            
            result["eyes"] = found_eyes
            
            # Aggregate Result for UI
            if not found_eyes:
                result["label"] = "No Eyes Detected"
                result["confidence"] = 0.0
            elif any_jaundice:
                result["label"] = "Jaundice"
                # Prioritize showing the Jaundice eye in debug view
                jaundice_eyes = [e for e in found_eyes if e["label"] == "Jaundice"]
                result["confidence"] = max([e["confidence"] for e in jaundice_eyes])
                result["debug_image"] = jaundice_eyes[0]["debug_image"]
            else:
                result["label"] = "Normal"
                result["confidence"] = max([e["confidence"] for e in found_eyes])
                # Show first eye
                result["debug_image"] = found_eyes[0]["debug_image"]
            
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
