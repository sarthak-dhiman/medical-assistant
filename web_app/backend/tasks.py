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
jaundice_model = None
skin_model = None

def load_models():
    """Loads models once per worker process."""
    global seg_model, jaundice_model, skin_model
    
    if seg_model is None:
        try:
            from segformer_utils import SegFormerWrapper
            print("Loading SegFormer...")
            seg_model = SegFormerWrapper()
        except Exception as e:
            print(f"Failed to load SegFormer: {e}")
            
    if jaundice_model is None:
        try:
            from inference import predict_frame
            # predict_frame loads model internally on first call
            # We just need reference to the function or pre-warm it
            print("Warming Jaundice Model...")
            jaundice_model = predict_frame # Function reference
        except Exception as e:
            print(f"Failed to load Jaundice Logic: {e}")

    if skin_model is None:
        try:
            from inference_skin import predict_skin_disease
            print("Warming Skin Model...")
            skin_model = predict_skin_disease # Function reference
        except Exception as e:
            print(f"Failed to load Skin Logic: {e}")
            
@celery_app.task(bind=True)
def predict_task(self, image_data_b64, mode):
    load_models()
    
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
        # We can return the mask as base64 to overlay in frontend
        # Or just return bbox coordinates. Let's return coords + prediction.
        
        if mode == "JAUNDICE_BODY":
             # Detect Jaundice on Body Skin
             masked_roi = cv2.bitwise_and(frame, frame, mask=skin_mask)
             if cv2.countNonZero(skin_mask) > 100:
                 # Crop bounding box of skin
                 y, x = np.where(skin_mask > 0)
                 y0, y1, x0, x1 = y.min(), y.max(), x.min(), x.max()
                 crop = masked_roi[y0:y1, x0:x1]
                 
                 label, conf = jaundice_model(crop)
                 result.update({
                     "label": label, 
                     "confidence": conf, 
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
            
            for crop, name, (x1, y1, x2, y2) in eyes:
                # Apply Iris Masking to remove colored iris (which confuses the model)
                crop_masked, method = seg_model.apply_iris_mask(crop)
                
                from inference_sclera import predict_sclera_jaundice
                
                # Pass accurate skin_crop + eye_crop
                label, conf = predict_sclera_jaundice(skin_crop, crop_masked)
                
                if label == "Jaundice":
                    any_jaundice = True
                    
                eye_result = {
                    "name": name, 
                    "label": label, 
                    "confidence": float(conf),
                    "bbox": [x1/w, y1/h, x2/w, y2/h], # Normalized
                    "method": method
                }
                
                if method == "fallback":
                    eye_result["warning"] = "Blurry image detected. Using fallback mode."
                    
                found_eyes.append(eye_result)
            
            result["eyes"] = found_eyes
            
            # DEBUG: Return the first localized/processed eye for user verification
            if found_eyes:
                # Get the processed crop from the last iteration or first
                # We need to re-encode the last masked crop to show user what the AI saw
                _, buffer = cv2.imencode('.jpg', crop_masked)
                debug_b64 = base64.b64encode(buffer).decode('utf-8')
                result["debug_image"] = f"data:image/jpeg;base64,{debug_b64}"
            
            # Fallback: If no eyes found but we are in explicit Eye Mode (e.g. macro shot), 
            # treat the whole frame as an eye.
            if not found_eyes:
                 # Check if the image resembles an eye or just force it? 
                 # For now, just force it as "Macro Eye"
                 # Use the whole frame as crop
                 name = "Macro/Single Eye"
                 
                 # Apply Iris Masking here too!
                 frame_masked, method = seg_model.apply_iris_mask(frame)
                 
                 from inference_sclera import predict_sclera_jaundice
                 # Use the frame itself as skin fallback (or a corner? No, frame is best guess)
                 label, conf = predict_sclera_jaundice(frame, frame_masked) 
                 
                 if label == "Jaundice":
                     any_jaundice = True

                 eye_result = {
                    "name": name,
                    "label": label,
                    "confidence": float(conf),
                    "bbox": [0.0, 0.0, 1.0, 1.0], # Whole frame
                    "method": method
                 }
                 
                 if method == "fallback":
                     eye_result["warning"] = "Blurry image detected. Using fallback mode."
                 
                 found_eyes.append(eye_result)
                 
                 # DEBUG: Return the masked macro frame
                 _, buffer = cv2.imencode('.jpg', frame_masked)
                 debug_b64 = base64.b64encode(buffer).decode('utf-8')
                 result["debug_image"] = f"data:image/jpeg;base64,{debug_b64}"
            
            # Aggregate Result for UI
            if not found_eyes:
                # Should not reach here due to fallback, but safe to keep
                result["label"] = "No Eyes Detected"
                result["confidence"] = 0.0
            elif any_jaundice:
                result["label"] = "Jaundice"
                # Avg confidence of jaundice eyes? Or Max? Taking Max is safer for alerts.
                result["confidence"] = max([e["confidence"] for e in found_eyes if e["label"] == "Jaundice"])
            else:
                result["label"] = "Normal"
                result["confidence"] = max([e["confidence"] for e in found_eyes])
            
        elif mode == "SKIN_DISEASE":
             # Detect on Skin
             if cv2.countNonZero(skin_mask) > 100:
                 y, x = np.where(skin_mask > 0)
                 y0, y1, x0, x1 = y.min(), y.max(), x.min(), x.max()
                 masked_roi = cv2.bitwise_and(frame, frame, mask=skin_mask)
                 crop = masked_roi[y0:y1, x0:x1]
                 
                 label, conf = skin_model(crop)
                 result.update({
                     "label": label, 
                     "confidence": float(conf), 
                     "bbox": [x0/w, y0/h, x1/w, y1/h] # Normalized
                 })
             else:
                 result.update({"label": "No Skin", "confidence": 0.0})

    return result
