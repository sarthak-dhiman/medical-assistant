import sys
import os
import cv2
import numpy as np
import base64
import logging
from pathlib import Path
from .celery_app import celery_app
from .config import settings
import mediapipe as mp

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

# --- Hand Detection Helper ---
# Lazy initialization of MediaPipe Hands
mp_hands = None
hands_detector = None

def get_hand_detector():
    global mp_hands, hands_detector
    if hands_detector is None:
        try:
            mp_hands = mp.solutions.hands
            hands_detector = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5
            )
        except Exception as e:
            logger.error(f"Failed to init MediaPipe Hands: {e}")
            return None
    return hands_detector

def    detect_hand_and_crop(image):
    """
    Detects hand in the image and returns a crop of the hand region.
    Falls back to center crop if no hand detected.
    Returns: (crop_img, debug_info_dict, bbox_list)
    """
    try:
        debug_info = {}
        detector = get_hand_detector()
        
        if detector:
            # Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detector.process(img_rgb)
            
            if results.multi_hand_landmarks:
                # Get bounding box
                h, w, _ = image.shape
                landmarks = results.multi_hand_landmarks[0].landmark
                
                x_min, x_max = w, 0
                y_min, y_max = h, 0
                
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)
                
                # Add padding (20%)
                pad_x = int((x_max - x_min) * 0.2)
                pad_y = int((y_max - y_min) * 0.2)
                
                x_min = max(0, x_min - pad_x)
                x_max = min(w, x_max + pad_x)
                y_min = max(0, y_min - pad_y)
                y_max = min(h, y_max + pad_y)
                
                # Crop
                crop = image[y_min:y_max, x_min:x_max]
                
                # Draw debug box on original image copy
                debug_img = image.copy()
                cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Encode debug image
                success, buffer = cv2.imencode('.jpg', debug_img)
                if success:
                    debug_info["debug_image"] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                
                debug_info["hand_detected"] = True
                
                # Return normalized bbox
                bbox = [x_min/w, y_min/h, x_max/w, y_max/h]
                return crop, debug_info, bbox
                
        # Fallback: Center Crop (50%)
        h, w, _ = image.shape
        cy, cx = h // 2, w // 2
        crop_h, crop_w = h // 2, w // 2
        
        y1 = max(0, cy - crop_h // 2)
        y2 = min(h, cy + crop_h // 2)
        x1 = max(0, cx - crop_w // 2)
        x2 = min(w, cx + crop_w // 2)
        
        crop = image[y1:y2, x1:x2]
        
        # Debug for fallback
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        success, buffer = cv2.imencode('.jpg', debug_img)
        if success:
            debug_info["debug_image"] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
            
        debug_info["hand_detected"] = False
        debug_info["fallback"] = "Center Crop"
        
        # Return normalized bbox (None for fallback to avoid noise)
        bbox = None # [x1/w, y1/h, x2/w, y2/h] 
        return crop, debug_info, bbox
        
    except Exception as e:
        logger.error(f"Hand detection failed: {e}")
        # Return full image and full bbox as fail-safe
        return image, {"error": str(e)}, [0.0, 0.0, 1.0, 1.0]

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

@celery_app.task(bind=True, max_retries=3)
def predict_task(self, image_data_b64, mode, debug=False):
    """
    Celery task for prediction with comprehensive error handling and retry logic.
    """
    try:
        # Validate inputs
        if not image_data_b64 or not isinstance(image_data_b64, str):
            return {"status": "error", "error": "Invalid or missing image data"}
        
        if not mode or mode not in ["JAUNDICE_BODY", "JAUNDICE_EYE", "SKIN_DISEASE"]:
            return {"status": "error", "error": f"Invalid mode: {mode}"}
        
        # Decode Image with comprehensive error handling
        try:
            if "," in image_data_b64:
                header, image_data_b64 = image_data_b64.split(",", 1)
                # Validate header contains image/
                if "image/" not in header:
                    logger.warning(f"Suspicious image header: {header[:50]}")
            
            # Validate base64 data size (rough estimate)
            if len(image_data_b64) > 100 * 1024 * 1024:  # 100MB limit
                return {"status": "error", "error": "Image data too large (max 100MB)"}
            
            # Safe decoding with validation
            try:
                image_bytes = base64.b64decode(image_data_b64, validate=True)
            except Exception as e:
                logger.error(f"Base64 decoding failed: {e}")
                return {"status": "error", "error": "Invalid image encoding"}
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return {"status": "error", "error": "Failed to decode image - invalid format"}
            
            # Validate image dimensions
            if len(frame.shape) < 2:
                return {"status": "error", "error": "Invalid image dimensions"}
            
            h, w = frame.shape[:2]
            if h < 10 or w < 10:
                return {"status": "error", "error": f"Image too small: {w}x{h}"}
            if h > 8192 or w > 8192:
                return {"status": "error", "error": f"Image too large: {w}x{h}"}

        except Exception as e:
            logger.error(f"Image decoding error: {e}")
            return {"status": "error", "error": "Invalid image data"}

        # Initialize result
        result = {"status": "success", "mode": mode}
        
        # Model availability checks - only SegFormer is managed here
        # Other models are loaded by inference_service on demand
        if mode in ["JAUNDICE_EYE", "JAUNDICE_BODY", "SKIN_DISEASE"]:
            if not seg_model or not seg_model.is_ready:
                return {"status": "error", "error": "SegFormer Not Ready (Check Logs)"}
        
        # Process based on mode
        if mode in ["JAUNDICE_BODY", "JAUNDICE_EYE", "SKIN_DISEASE"]:
            return _process_segmentation_mode(frame, mode, debug)
        else:
            return {"status": "error", "error": f"Unknown mode: {mode}"}
            
    except Exception as e:
        logger.error(f"Unexpected error in predict_task: {e}")
        import traceback
        traceback.print_exc()
        
        # Retry logic for transient errors
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying task ({self.request.retries + 1}/{self.max_retries})")
            raise self.retry(countdown=2 ** self.request.retries)
        
        return {"status": "error", "error": f"Task failed after retries: {str(e)}"}


def _process_segmentation_mode(frame, mode, debug):
    """Process image for segmentation-based modes with comprehensive error handling."""
    result = {"status": "success", "mode": mode}
    
    try:
        h, w = frame.shape[:2]
        
        # 1. Segmentation (Needed for Jaundice, Skin, Burns)
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
            
            # Process based on specific mode
            if mode == "JAUNDICE_BODY":
                return _process_jaundice_body(frame, skin_mask, w, h, debug, result)
            elif mode == "JAUNDICE_EYE":
                return _process_jaundice_eye(frame, skin_mask, w, h, debug, result)
            elif mode == "SKIN_DISEASE":
                return _process_skin_disease(frame, skin_mask, w, h, debug, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in segmentation processing: {e}")
        return {"status": "error", "error": f"Processing failed: {str(e)}"}


def _process_jaundice_body(frame, skin_mask, w, h, debug, result):
    """Process jaundice body detection."""
    try:
        # 1. Get Person Mask (MediaPipe)
        try:
            person_mask = seg_model.get_body_segmentation(frame)
        except Exception:
            person_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
            
        # 2. Combine with Skin Color Mask
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
                "bbox": [x0/w, y0/h, x1/w, y1/h],
                "debug_info": debug_info
            })
        else:
            result.update({"label": "No Skin on Body", "confidence": 0.0})
            
        return result
        
    except Exception as e:
        logger.error(f"Jaundice body processing error: {e}")
        return {"status": "error", "error": f"Jaundice body detection failed: {str(e)}"}


def _process_jaundice_eye(frame, skin_mask, w, h, debug, result):
    """Process jaundice eye detection."""
    try:
        # 1. Extract Skin Crop (Context for Model)
        skin_crop = frame  # Fallback
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
            raw_eyes = seg_model.get_eye_rois(mask, frame)
            for cropped_eye, name, bbox in raw_eyes:
                masked, _ = seg_model.apply_iris_mask(cropped_eye)
                eyes.append((masked, name, bbox))

        found_eyes = []
        any_jaundice = False
        
        for cropped_eye_masked, name, (x1, y1, x2, y2) in eyes:
            # Encode for Nerd Mode
            try:
                _, buffer = cv2.imencode('.jpg', cropped_eye_masked)
                eye_b64 = base64.b64encode(buffer).decode('utf-8')
                debug_img_str = f"data:image/jpeg;base64,{eye_b64}"
            except Exception:
                debug_img_str = ""

            # Ensure skin_crop is valid
            if skin_crop is None or skin_crop.size == 0:
                skin_crop = frame 

            # Use inference service for prediction
            label, conf, debug_info = inference_service.predict_jaundice_eye(skin_crop, cropped_eye_masked, debug=debug)
            
            # Add Eye Mask to debug_info
            if "masks" not in debug_info: debug_info["masks"] = {}
            debug_info["masks"]["skin_mask"] = encode_mask_b64(skin_mask)
            
            if label == "Jaundice":
                any_jaundice = True
                
            eye_result = {
                "name": name, 
                "label": label, 
                "confidence": float(conf),
                "bbox": [x1/w, y1/h, x2/w, y2/h],
                "method": "mediapipe" if used_mediapipe else "segformer",
                "debug_image": debug_img_str,
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
            jaundice_eyes = [e for e in found_eyes if e["label"] == "Jaundice"]
            result["confidence"] = max([e["confidence"] for e in jaundice_eyes])
            result["debug_image"] = jaundice_eyes[0]["debug_image"]
        else:
            result["label"] = "Normal"
            result["confidence"] = max([e["confidence"] for e in found_eyes])
            result["debug_image"] = found_eyes[0]["debug_image"]
            
        return result
        
    except Exception as e:
        logger.error(f"Jaundice eye processing error: {e}")
        return {"status": "error", "error": f"Jaundice eye detection failed: {str(e)}"}


def _process_skin_disease(frame, skin_mask, w, h, debug, result):
    """Process skin disease detection."""
    try:
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
            
        return result
        
    except Exception as e:
        logger.error(f"Skin disease processing error: {e}")
        return {"status": "error", "error": f"Skin disease detection failed: {str(e)}"}


# Note: Additional mode handlers (BURNS, NAIL_DISEASE) can be added here following the same pattern
