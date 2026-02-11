import sys
import os
import cv2
import numpy as np
import base64
import logging
from pathlib import Path
from .celery_app import celery_app
from .config import settings

import datetime

# --- IST Logging Configuration ---
def ist_converter(*args):
    # IST = UTC + 5:30
    return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5, minutes=30)).timetuple()

logging.Formatter.converter = ist_converter

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# Ensure project root is in path to import original scripts
# (Using settings.PROJECT_ROOT if available, else relative)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import inference service layer
from .inference_service import inference_service

# Lazy Global Models (Segmentation only - inference delegated to service)
seg_model = None
init_errors = [] # Persistent global errors

def encode_mask_b64(mask):
    """Encodes a single-channel mask (uint8) to base64 PNG."""
    try:
        if mask is None: return None
        # Ensure it's visualizable (0 or 255)
        visual_mask = mask.copy()
        if visual_mask.max() <= 1:
             visual_mask = visual_mask * 255
             
        success, buffer = cv2.imencode('.png', visual_mask)
        if not success: return None
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None

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
            
            # DEBUG: Write error to file
            try:
                with open("/app/saved_models/init_error.txt", "w") as f:
                    f.write(f"SegFormer Import Error: {e}\n")
                    import traceback
                    f.write(traceback.format_exc())
            except Exception as file_err:
                logger.error(f"Failed to write debug log: {file_err}")
            
    # Models are now loaded lazily via inference_service (thread-safe)
    # We only need to warm up SegFormer here
    
    # Note: Jaundice and Skin models are loaded on-demand by inference_service
    # with thread-safe locks, so we don't need to pre-load them here
            
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
        logger.info("✅ SegFormer Ready. Warming up Classification Models...")
        try:
            # Import getters locally to avoid top-level side effects (just in case)
            from inference_new_models import get_burns_model, get_nail_model
            from inference_pytorch import get_eye_model, get_body_model, get_skin_model
            
            # Trigger loading
            get_eye_model()
            get_body_model()
            get_skin_model()
            get_burns_model()
            get_nail_model()
            logger.info("✅ WORKER INIT SUCCESS: All models warmed up!")
        except Exception as e:
            logger.error(f"⚠️ Model Warmup Partial Fail: {e}")

@celery_app.task(bind=True)
def check_model_health(self):
    """
    Simple probe to check if models are loaded in the worker.
    Returns: dict of model statuses
    """
    global seg_model
    status = {
        "seg_model": seg_model is not None,
        "inference_service": "ready",  # Service layer is always available
        "config": {
           "device": os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"),
           "omp_threads": os.environ.get("OMP_NUM_THREADS", "Not Set")
        },
        "errors": init_errors,
        "ready": seg_model is not None and len(init_errors) == 0
    }
    return status

@celery_app.task(bind=True)
def predict_task(self, image_data_b64, mode, debug=False):
    # FAST FAIL: Check if SegFormer loaded correctly during init
    # Note: Inference models (jaundice/skin) are loaded lazily by inference_service
    if not seg_model or not seg_model.is_ready:
        return {"status": "error", "error": "SegFormer Not Ready (Check Logs)"}
    
    # Decode Image
    try:
        if "," in image_data_b64:
            header, image_data_b64 = image_data_b64.split(",", 1)
            # Basic header validation
            if "image/" not in header:
                 logger.warning(f"Suspicious image header: {header}")
        
        # Safe decoding with proper error handling
        try:
            image_bytes = base64.b64decode(image_data_b64, validate=True)
        except Exception as b64_err:
            logger.error(f"Base64 validation failed: {b64_err}")
            return {"status": "error", "error": "Invalid base64 encoding"}
            
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode image from bytes")

    except ValueError as ve:
        logger.error(f"Image Decoding Failed: {ve}")
        return {"status": "error", "error": str(ve)}
    except Exception as e:
        logger.error(f"Unexpected error during image processing: {e}")
        return {"status": "error", "error": "Invalid Image Data"}

    # Logic from webcam_app.py (Simplified for API)
    result = {"status": "success", "mode": mode}
    
    h, w = frame.shape[:2]
    
    # 1. Segmentation (Needed for Jaundice, Skin, Burns)
    if mode in ["JAUNDICE_BODY", "JAUNDICE_EYE", "SKIN_DISEASE", "BURNS"]:
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
                  
                  # Use inference service for prediction
                  label, conf, debug_info = inference_service.predict_jaundice_body(crop, debug=debug)
                  
                  # Add Masks to debug_info
                  if "masks" not in debug_info: debug_info["masks"] = {}
                  debug_info["masks"]["skin_mask"] = encode_mask_b64(skin_mask)
                  debug_info["masks"]["person_mask"] = encode_mask_b64(person_mask)
                  
                  result.update({
                      "label": label, 
                      "confidence": float(conf), 
                      "bbox": [x0/w, y0/h, x1/w, y1/h], # Normalized
                      "debug_info": debug_info
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

                # Use inference service for prediction
                label, conf, debug_info = inference_service.predict_jaundice_eye(skin_crop, cropped_eye_masked, debug=debug)
                
                # Add Eye Mask to debug_info
                # (Note: cropped_eye_masked is already the Sclera Mask applied to image)
                # Ideally we want the actual Sclera Mask used.
                # But for now, let's just send the Global Skin Mask as context
                if "masks" not in debug_info: debug_info["masks"] = {}
                debug_info["masks"]["skin_mask"] = encode_mask_b64(skin_mask)
                
                
                if label == "Jaundice":
                    any_jaundice = True
                    
                eye_result = {
                    "name": name, 
                    "label": label, 
                    "confidence": float(conf),
                    "bbox": [x1/w, y1/h, x2/w, y2/h], # Normalized
                    "method": "mediapipe" if used_mediapipe else "hough",
                    "debug_image": debug_img_str, # Store per eye
                    "debug_info": debug_info
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
                  
                  # Use inference service for prediction
                  label, conf, debug_info = inference_service.predict_skin_disease(crop, debug=debug)
                  
                  if "masks" not in debug_info: debug_info["masks"] = {}
                  debug_info["masks"]["skin_mask"] = encode_mask_b64(skin_mask)
                  
                  result.update({
                      "label": label, 
                      "confidence": float(conf), 
                      "bbox": [x0/w, y0/h, x1/w, y1/h], # Normalized
                      "debug_info": debug_info
                  })
             else:
                  debug_info = {}
                  if "masks" not in debug_info: debug_info["masks"] = {}
                  debug_info["masks"]["skin_mask"] = encode_mask_b64(skin_mask)
                  result.update({
                      "label": "No Skin", 
                      "confidence": 0.0,
                      "debug_info": debug_info
                  })
        
        elif mode == "BURNS":
            # Burns detection on skin
            if cv2.countNonZero(skin_mask) > 100:
                y, x = np.where(skin_mask > 0)
                y0, y1, x0, x1 = y.min(), y.max(), x.min(), x.max()
                crop = frame[y0:y1, x0:x1]
                
                if len(crop.shape) == 2:
                    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                elif crop.shape[2] != 3:
                    crop = crop[:, :, :3]
                
                logger.info("🔥 CALLING BURNS INFERENCE SERVICE...")
                label, conf, debug_info = inference_service.predict_burns(crop, debug=debug)
                logger.info(f"🔥 BURNS INFERENCE RETURNED: {label}")
                
                if "masks" not in debug_info: debug_info["masks"] = {}
                debug_info["masks"]["skin_mask"] = encode_mask_b64(skin_mask)
                
                result.update({
                    "label": label,
                    "confidence": float(conf),
                    "bbox": [x0/w, y0/h, x1/w, y1/h],
                    "debug_info": debug_info
                })
            else:
                debug_info = {}
                if "masks" not in debug_info: debug_info["masks"] = {}
                debug_info["masks"]["skin_mask"] = encode_mask_b64(skin_mask)
                result.update({
                    "label": "No Skin",
                    "confidence": 0.0,
                    "debug_info": debug_info
                })
    

    
    elif mode == "NAIL_DISEASE":
        # Nail disease detection
        label, conf, debug_info = inference_service.predict_nail_disease(frame, debug=debug)
        result.update({
            "label": label,
            "confidence": float(conf),
            "debug_info": debug_info
        })
    


    logger.info(f"Task Finished. Result Keys: {list(result.keys())}")
    logger.info(f"Debug Mode: {debug}")
    return result
