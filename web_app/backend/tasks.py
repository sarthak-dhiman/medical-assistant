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
try:
    from mmpose.apis import MMPoseInferencer
except ImportError:
    MMPoseInferencer = None

# --- Performance & Stability ---
# Force MediaPipe to CPU to avoid EGL errors in Docker with NVIDIA GPUS
# This is safe because MediaPipe (Hands/Pose) is very fast on CPU
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

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
    global seg_model, init_errors
    
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
            # Import getters directly from source modules to ensure availability inside containers
            from inference_pytorch import get_eye_model, get_body_model, get_skin_model
            # Import getters directly from source modules to ensure availability inside containers
            from inference_pytorch import get_eye_model, get_body_model, get_skin_model
            from inference_new_models import get_burns_model, get_nail_model, get_oral_cancer_model, get_teeth_model, get_posture_model

            # Trigger loading with timing logs
            import time
            warmup_funcs = [
                ("eye_model", get_eye_model),
                ("body_model", get_body_model),
                ("skin_model", get_skin_model),
                ("burns_model", get_burns_model),
                ("nail_model", get_nail_model),
                # cataract_model removed
                ("oral_cancer_model", get_oral_cancer_model),
                ("teeth_model", get_teeth_model),
                ("posture_model", get_posture_model)
            ]

            total_start = time.time()
            for name, fn in warmup_funcs:
                try:
                    start = time.time()
                    fn()
                    duration = time.time() - start
                    logger.info(f"Warmup: {name} loaded in {duration:.3f}s")
                except Exception as me:
                    logger.warning(f"Warmup failed for {name}: {me}")

            total_dur = time.time() - total_start
            logger.info(f"✅ WORKER INIT SUCCESS: Models warmed up (total {total_dur:.3f}s)")
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
        
        if not mode or mode not in ["JAUNDICE_BODY", "JAUNDICE_EYE", "SKIN_DISEASE", "BURNS", "NAIL_DISEASE", "CATARACT", "ORAL_CANCER", "TEETH", "POSTURE"]:
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
        if mode in ["JAUNDICE_EYE", "JAUNDICE_BODY", "SKIN_DISEASE", "BURNS", "NAIL_DISEASE"]:
            if not seg_model or not seg_model.is_ready:
                return {"status": "error", "error": "SegFormer Not Ready (Check Logs)"}
        
        # Process based on mode
        # CATARACT case removed
        if mode == "ORAL_CANCER":
            # Oral cancer is a direct classification on oral cavity images
            try:
                label, conf, debug_info = inference_service.predict_oral_cancer(frame, debug=debug)
                result = {"status": "success", "mode": mode, "label": label, "confidence": float(conf), "debug_info": debug_info}
                if "bbox" in debug_info:
                    result["bbox"] = debug_info["bbox"]
                recommendations = inference_service.get_recommendations(label)
                if recommendations:
                    result["recommendations"] = recommendations
                return result
            except Exception as e:
                logger.error(f"Oral cancer processing error: {e}")
                return {"status": "error", "error": f"Oral cancer detection failed: {str(e)}"}
        elif mode == "TEETH":
            # Teeth disease is a direct classification on teeth/mouth images
            try:
                label, conf, debug_info = inference_service.predict_teeth_disease(frame, debug=debug)
                result = {"status": "success", "mode": mode, "label": label, "confidence": float(conf), "debug_info": debug_info}
                if "bbox" in debug_info:
                    result["bbox"] = debug_info["bbox"]
                recommendations = inference_service.get_recommendations(label)
                if recommendations:
                    result["recommendations"] = recommendations
                return result
            except Exception as e:
                logger.error(f"Teeth disease processing error: {e}")
                return {"status": "error", "error": f"Teeth disease detection failed: {str(e)}"}
        elif mode == "POSTURE":
            return _process_posture(frame, w, h, debug, result)
        elif mode in ["JAUNDICE_BODY", "JAUNDICE_EYE", "SKIN_DISEASE", "BURNS", "NAIL_DISEASE"]:
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
        if mode in ["JAUNDICE_BODY", "JAUNDICE_EYE", "SKIN_DISEASE", "BURNS", "NAIL_DISEASE"]:
            # Optimization: Downscale for SegFormer
            INFERENCE_WIDTH = 480
            scale = INFERENCE_WIDTH / w
            if scale < 1.0:
                small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            else:
                small_frame = frame
                
            mask_small = seg_model.predict(small_frame)
            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Skin Mask (Semantic) - Face/Neck
            skin_mask_sem = seg_model.get_skin_mask(mask)
            
            # Skin Mask (Color) - Arms/Hands/Legs
            # Always compute this to capture non-face skin
            skin_mask_color = seg_model.get_skin_mask_color(frame)
            
            # Combine: Union of both methods
            # This ensures we get high-quality face mask + color-based body mask
            skin_mask = cv2.bitwise_or(skin_mask_sem, skin_mask_color)
            
            # Process based on specific mode
            if mode == "JAUNDICE_BODY":
                return _process_jaundice_body(frame, skin_mask, w, h, debug, result)
            elif mode == "JAUNDICE_EYE":
                return _process_jaundice_eye(frame, skin_mask, w, h, debug, result)
            elif mode == "SKIN_DISEASE":
                return _process_skin_disease(frame, skin_mask, w, h, debug, result)
            elif mode == "BURNS":
                return _process_burns(frame, skin_mask, w, h, debug, result)
            elif mode == "NAIL_DISEASE":
                return _process_nail_disease(frame, skin_mask, w, h, debug, result)
        
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


def _process_burns(frame, skin_mask, w, h, debug, result):
    """Process burns detection."""
    try:
        skin_pixels = cv2.countNonZero(skin_mask)
        if debug:
            print(f"[DEBUG] Burns Skin Pixels: {skin_pixels}", flush=True)
            
        if skin_pixels > 100:
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
            label, conf, debug_info = inference_service.predict_burns(crop, debug=debug)
            
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
        logger.error(f"Burns processing error: {e}")
        return {"status": "error", "error": f"Burns detection failed: {str(e)}"}


def _process_nail_disease(frame, skin_mask, w, h, debug, result):
    """Process nail disease detection with hand detection first."""
    try:
        # 1. Detect Hand
        hand_crop, hand_debug, hand_bbox = detect_hand_and_crop(frame)
        
        if not hand_debug.get("hand_detected", False):
            result.update({
                "label": "No Hand Detected",
                "confidence": 0.0,
                "debug_info": hand_debug
            })
            return result

        # 2. Use hand crop for prediction
        label, conf, debug_info = inference_service.predict_nail_disease(hand_crop, debug=debug)
        
        # Merge debug infos
        final_debug = {**hand_debug, **debug_info}
        if "masks" not in final_debug: final_debug["masks"] = {}
        final_debug["masks"]["skin_mask"] = encode_mask_b64(skin_mask)
        
        result.update({
            "label": label, 
            "confidence": float(conf), 
            "bbox": hand_bbox, # Use accurate hand bbox
            "debug_info": final_debug
        })
        return result
        
    except Exception as e:
        logger.error(f"Nail disease processing error: {e}")
        return {"status": "error", "error": f"Nail disease detection failed: {str(e)}"}


def _process_cataract(frame, debug):
    """Process cataract detection from eye image.
    
    Cataract detection doesn't require body segmentation - it works directly
    on close-up eye images (e.g., slit lamp photos or eye close-ups).
    """
    result = {"status": "success", "mode": "CATARACT"}
    
    try:
        # Direct inference on the input frame (expected to be eye image)
        label, conf, debug_info = inference_service.predict_cataract(frame, debug=debug)
        
        result.update({
            "label": label,
            "confidence": float(conf),
            "debug_info": debug_info
        })
        # if debug:
        #     result["bbox"] = [0.0, 0.0, 1.0, 1.0] # Removed redundant full-frame box
        
        # Add recommendations
        recommendations = inference_service.get_recommendations(label)
        if recommendations:
            result["recommendations"] = recommendations
            
        return result
        
    except Exception as e:
        logger.error(f"Cataract processing error: {e}")
        return {"status": "error", "error": f"Cataract detection failed: {str(e)}"}


# --- Diagnostic Gateway ---

# Lazy initialization
mmpose_inferencer = None

def _process_posture(frame, w, h, debug, result):
    """Process posture detection using MediaPipe Pose.

    Returns all 33 MediaPipe landmarks for frontend skeleton drawing,
    while using a 12-point subset for the posture classifier.
    """
    try:
        mp_pose = mp.solutions.pose

        # 12-point subset used for the classifier (matches training format)
        # MediaPipe indices: nose, l/r shoulder, l/r elbow, l/r hip, l/r knee, l/r ankle, mouth_left
        POSE_INDICES = [0, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 9]
        FEATURE_SIZE = 24  # 12 * 2

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.4,
        ) as pose:
            mp_result = pose.process(img_rgb)

        debug_info = {}

        if mp_result.pose_landmarks is None:
            result.update({
                "label": "No Posture Detected",
                "confidence": 0.0,
                "debug_info": debug_info,
            })
            return result

        lm = mp_result.pose_landmarks.landmark

        # --- ALL 33 landmarks for frontend drawing ---
        all_landmarks = [
            {
                "x": float(pt.x),
                "y": float(pt.y),
                "z": float(getattr(pt, "z", 0.0)),
                "visibility": float(getattr(pt, "visibility", 1.0)),
            }
            for pt in lm
        ]
        result["landmarks"] = all_landmarks  # full 33-pt array

        # --- 12-point flat vector for classifier ---
        flat_features = []
        for idx in POSE_INDICES:
            pt = lm[idx]
            flat_features.append(float(pt.x))
            flat_features.append(float(pt.y))

        # Safety pad/truncate
        if len(flat_features) < FEATURE_SIZE:
            flat_features += [0.0] * (FEATURE_SIZE - len(flat_features))
        else:
            flat_features = flat_features[:FEATURE_SIZE]

        try:
            import torch
            label, conf, _ = inference_service.predict_posture(flat_features, debug=debug)
            
            # --- Neck Posture Heuristic ---
            # If the model predicts "Good Form", we double-check the neck alignment
            if "Good" in label:
                ls = lm[11]
                rs = lm[12]
                nose = lm[0]
                
                if ls.visibility > 0.3 and rs.visibility > 0.3 and nose.visibility > 0.3:
                    shoulder_y = (ls.y + rs.y) / 2.0
                    shoulder_width = abs(ls.x - rs.x)
                    
                    if shoulder_width > 0:
                        # Normalize vertical neck length by shoulder width to make it scale-invariant
                        neck_ratio = (shoulder_y - nose.y) / shoulder_width
                        
                        # If neck_ratio is too small, head is pitched forward or down heavily
                        # If neck_ratio is too high, head is tilted excessively backward/up
                        # Typical upright ratio is ~0.4 to 0.75 depending on camera angle
                        if neck_ratio < 0.35:
                            label = "Bad Form (Neck Down)"
                            conf = 0.85 # High confidence override
                            debug_info["heuristic_override"] = f"neck_ratio={neck_ratio:.2f}"
                        elif neck_ratio > 0.85:
                            label = "Bad Form (Neck Up)"
                            conf = 0.85
                            debug_info["heuristic_override"] = f"neck_ratio={neck_ratio:.2f}"

        except Exception as pe:
            logger.error(f"Posture classification failed: {pe}")
            label = "Posture Detected (Classification Failed)"
            conf = 0.5

        result.update({
            "label": label,
            "confidence": float(conf),
            "debug_info": debug_info,
        })
        return result

    except Exception as e:
        logger.error(f"Posture processing error: {e}")
        return {"status": "error", "error": f"Posture detection failed: {str(e)}"}





# --- Diagnostic Gateway ---

# Lazy initialization
mp_face_mesh = None
face_mesh_detector = None
yolo_model = None

def get_face_mesh():
    global mp_face_mesh, face_mesh_detector
    if face_mesh_detector is None:
        try:
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh_detector = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        except Exception as e:
            logger.error(f"Failed to init MediaPipe Face Mesh: {e}")
            return None
    return face_mesh_detector

def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        try:
            from ultralytics import YOLO
            # Load standard YOLOv8n model (will download if not present)
            # Using 'yolov8n.pt' which is small and fast
            model_path = "/app/saved_models/yolov8n.pt"
            logger.info(f"Loading YOLOv8n from {model_path}...")
            yolo_model = YOLO(model_path) 
        except Exception as e:
            logger.error(f"Failed to init YOLO: {e}")
            return None
    return yolo_model

@celery_app.task(bind=True, max_retries=3)
def diagnostic_gateway_task(self, image_data_b64, debug=False):
    """
    Automatic routing of image to appropriate model based on content analysis.
    Refined Logic:
    1. Hands -> Nail Disease
    2. Face Detected -> 
       - Close-up Eyes -> Cataract
       - Close-up Mouth -> Oral Cancer
       - Else -> Jaundice Eye or Skin Disease (NOT Jaundice Body)
    3. No Face Detected -> 
       - YOLO Person/Infant Detected -> Jaundice Body
       - Else -> Skin Disease (Macro)
    """
    try:
        # 1. Decode Image
        if not image_data_b64 or not isinstance(image_data_b64, str):
            return {"status": "error", "error": "Invalid image data"}
        
        try:
            if "," in image_data_b64:
                _, image_data_b64 = image_data_b64.split(",", 1)
            image_bytes = base64.b64decode(image_data_b64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: raise ValueError("Decode failed")
        except Exception as e:
            return {"status": "error", "error": "Image decoding failed"}

        h, w = frame.shape[:2]
        
        # 2. ANALYSIS & ROUTING
        
        # A. Hand Detection (High Priority)
        _, hand_debug, _ = detect_hand_and_crop(frame)
        if hand_debug.get("hand_detected", False):
            logger.info("Auto-Routing: Hand Detected -> NAIL_DISEASE")
            return predict_task(image_data_b64, "NAIL_DISEASE", debug)

        # B. Face/Feature Detection (Face Mesh)
        mesh = get_face_mesh()
        route_mode = "SKIN_DISEASE" # Default Fallback (Macro Skin)
        
        face_detected = False
        
        if mesh:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mesh.process(img_rgb)
            
            if results.multi_face_landmarks:
                face_detected = True
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Heuristics for Features
                img_area = w * h
                
                def get_bbox_area(indices, landmarks, w, h):
                    xs = [landmarks[i].x * w for i in indices]
                    ys = [landmarks[i].y * h for i in indices]
                    return (max(xs) - min(xs)) * (max(ys) - min(ys))

                # Left Eye (33, 133, 159, 145), Right Eye (362, 263, 386, 374)
                left_eye_area = get_bbox_area([33, 133, 159, 145], landmarks, w, h)
                right_eye_area = get_bbox_area([362, 263, 386, 374], landmarks, w, h)
                total_eye_area = left_eye_area + right_eye_area
                
                # Lips (61, 291, 0, 17)
                lips_area = get_bbox_area([61, 291, 0, 17], landmarks, w, h)
                
                eye_ratio = total_eye_area / img_area
                lips_ratio = lips_area / img_area
                
                logger.info(f"Auto-Routing Analysis: EyeRatio={eye_ratio:.3f}, LipsRatio={lips_ratio:.3f}")
                
                if lips_ratio > 0.12: 
                    # If mouth is a close-up, check if it's open (Teeth) or closed (Oral Cancer)
                    # Mouth open ratio = Vertical distance / Face height
                    top_lip = landmarks[13]
                    bot_lip = landmarks[14]
                    face_h = landmarks[152].y - landmarks[10].y
                    mouth_d = np.linalg.norm(
                        np.array([top_lip.x * w, top_lip.y * h]) - np.array([bot_lip.x * w, bot_lip.y * h])
                    )
                    open_ratio = mouth_d / (face_h * h + 1e-6)
                    
                    if open_ratio > 0.05:
                        route_mode = "TEETH_DISEASE"
                        logger.info(f"Auto-Routing: Mouth is OPEN (ratio={open_ratio:.3f}) -> TEETH_DISEASE")
                    else:
                        route_mode = "ORAL_CANCER" # Close up mouth, but closed
                        logger.info(f"Auto-Routing: Mouth is CLOSED (ratio={open_ratio:.3f}) -> ORAL_CANCER")
                        
                elif eye_ratio > 0.10: 
                    route_mode = "CATARACT" # Close up eyes
                else:
                    # Face detected but not macro.
                    route_mode = "JAUNDICE_EYE"
        
        # C. Infant/Body Check (Only if NO FACE detected)
        # Jaundice Body is restricted to Babies/Infants
        if not face_detected:
            # Check for Body/Person using YOLO
            yolo = get_yolo_model()
            person_detected = False
            
            if yolo:
                # Run inference
                yolo_results = yolo(frame, verbose=False, classes=[0]) # class 0 is person
                # Check if any person detected
                for r in yolo_results:
                    if len(r.boxes) > 0:
                        person_detected = True
                        break
            
            if person_detected:
                # "Infant" proxy
                route_mode = "JAUNDICE_BODY"
            else:
                # No Face, No Body -> Macro Skin or Object
                route_mode = "SKIN_DISEASE"

        # 3. DISPATCH
        logger.info(f"Auto-Routing Decision: {route_mode} (Face: {face_detected})")
        return predict_task(image_data_b64, route_mode, debug)

    except Exception as e:
        logger.error(f"Gateway failed: {e}")
        return {"status": "error", "error": f"Gateway failed: {str(e)}"}
