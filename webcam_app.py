import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
# from ultralytics import YOLO # Removed as per SegFormer migration

# --- 1. CONFIGURATION & IMPORTS ---
current_model = "JAUNDICE"  # Start in Jaundice mode

try:
    from inference import predict_frame as predict_jaundice_model
    from inference_skin import predict_skin_disease as predict_skin_model
    from inference_sclera import predict_sclera_jaundice as predict_sclera_model
    from segformer_utils import SegFormerWrapper
    MODELS_LOADED = True
except ImportError as e:
    print(f"Warning: Dependency missing: {e}. Using dummy predictions.")
    MODELS_LOADED = False
    class SegFormerWrapper:
        def __init__(self): pass
        def predict(self, img): return np.zeros(img.shape[:2], dtype=np.uint8)
        def get_eye_rois(self, mask, img): return []
        def get_skin_mask(self, mask): return np.zeros(mask.shape, dtype=np.uint8)
    def predict_jaundice_model(img): return "Jaundice", 0.88
    def predict_skin_model(img): return "Eczema", 0.92
    def predict_sclera_model(skin, sclera): return "Jaundice", 0.85

# --- 2. HELPER FUNCTIONS ---
def get_manual_skin_mask(img):
    """
    Robust skin detection (YCbCr) - Works better for phone screens/photos.
    """
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 130, 70], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(ycbcr, lower, upper)
    
    # Cleaning
    kernel = np.ones((3,3), np.uint8) 
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def calculate_blur(img):
    """
    Calculates the Laplacian variance of an image.
    Lower values indicate more blur.
    """
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def draw_sidebar(frame, width, mode_name, result_text, conf_text, warning_text, fallback_active, is_static_image):
    """
    Draws a sidebar on the right side of the frame.
    """
    h, w = frame.shape[:2]
    # Create sidebar canvas with same height as frame
    sidebar = np.zeros((h, width, 3), dtype=np.uint8)
    sidebar[:] = (30, 30, 30) # Dark Gray background

    # Fonts
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    blue = (255, 200, 100)

    # 1. Mode Header
    cv2.putText(sidebar, "MODE:", (10, 40), font, 0.6, white, 1)
    cv2.putText(sidebar, mode_name, (10, 70), font, 0.7, yellow, 2)

    # 2. Source Indicator
    if is_static_image:
        src_text = "IMAGE FILE"
        src_color = blue
    elif fallback_active:
        src_text = "FALLBACK"
        src_color = yellow
    else:
        src_text = "AI MODEL"
        src_color = green
        
    cv2.putText(sidebar, f"Src: {src_text}", (10, 110), font, 0.5, src_color, 1)

    # 3. Separator
    cv2.line(sidebar, (10, 130), (width-10, 130), (100, 100, 100), 1)

    # 4. Result Section
    cv2.putText(sidebar, "RESULT:", (10, 160), font, 0.6, white, 1)
    
    # Dynamic Color for Result
    res_color = white
    if "Jaundice" in result_text or "Blurry" in result_text: res_color = red
    elif "Normal" in result_text or "Healthy" in result_text: res_color = green
    
    # Text Wrap slightly if needed (simple logic)
    y_pos = 200
    for line in result_text.split('\n'):
        cv2.putText(sidebar, line, (10, y_pos), font, 0.8, res_color, 2)
        y_pos += 30

    # 5. Confidence
    if conf_text:
        cv2.putText(sidebar, conf_text, (10, y_pos + 10), font, 0.6, (200, 200, 200), 1)

    # 6. Warnings
    if warning_text:
        cv2.putText(sidebar, "WARNING:", (10, h - 60), font, 0.5, (0, 0, 255), 1)
        cv2.putText(sidebar, warning_text, (10, h - 30), font, 0.6, (0, 0, 255), 2)
    
    # 7. Instructions
    cv2.putText(sidebar, "[M] Switch Mode", (10, h - 10), font, 0.4, (150, 150, 150), 1)
    cv2.putText(sidebar, "[U] Upload [C] Camera", (10, h - 30 if warning_text else h - 30), font, 0.4, (150, 150, 150), 1)

    # Combine
    combined = np.hstack((frame, sidebar))
    return combined

def main():
    global current_model
    
    # Init Tkinter for file dialog
    root = tk.Tk()
    root.withdraw() # Hide the main window

    # Load SegFormer (Heavy Model - Loads once)
    print("Loading SegFormer (Face Parsing)...")
    seg_model = SegFormerWrapper()

    print("Starting Camera...")
    cap = cv2.VideoCapture(0)
    # cap.set(3, 1280) 
    # cap.set(4, 720) # High res might slow down segmentation

    # Mode Cycle: Jaundice Body -> Jaundice Eye -> Skin Disease
    modes = ["JAUNDICE_BODY", "JAUNDICE_EYE", "SKIN_DISEASE"]
    mode_idx = 0
    
    sidebar_width = 300
    
    uploaded_image = None # State variable for static image
    
    # --- PERFORMANCE OPTIMIZATION ---
    frame_count = 0
    SKIP_FRAMES = 4 # Run AI every Nth frame
    cached_seg_mask = None
    INFERENCE_WIDTH = 480 # Downscale for speed (SegFormer is heavy)

    while True:
        # --- INPUT HANDLING ---
        if uploaded_image is not None:
            frame = uploaded_image.copy()
            # If image is too large, resize it to fit screen somewhat
            max_h = 800
            if frame.shape[0] > max_h:
                scale = max_h / frame.shape[0]
                frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            is_static = True
        else:
            ret, frame = cap.read()
            if not ret: break
            is_static = False

        h, w = frame.shape[:2]
        current_model = modes[mode_idx]

        # Init Dashboard Vars
        sidebar_result = "--"
        sidebar_conf = ""
        sidebar_warning = ""
        
        frame_count += 1
        
        # --- OPTIMIZED SEGMENTATION ---
        # Run heavy model only every SKIP_FRAMES or if we don't have a mask yet
        if cached_seg_mask is None or cached_seg_mask.shape[:2] != (h, w) or (not is_static and frame_count % SKIP_FRAMES == 0):
            # 1. Downscale for Inference
            scale_factor = INFERENCE_WIDTH / w
            if scale_factor < 1.0:
                small_frame = cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)
            else:
                small_frame = frame
                
            # 2. Predict
            small_mask = seg_model.predict(small_frame)
            
            # 3. Upscale Mask back to Original Size
            # Nearest Neighbor is fast and keeps class IDs (0, 1, 2...) intact
            cached_seg_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        seg_mask = cached_seg_mask
        
        # Check Fallback Condition
        ai_skin_mask = seg_model.get_skin_mask(seg_mask)
        ai_skin_count = cv2.countNonZero(ai_skin_mask)
        
        using_fallback = False
        final_skin_mask = ai_skin_mask
        
        # Determine fallback threshold based on image size
        # Default webcam is often 640x480 -> 307200 pixels. 2000 is ~0.6%
        threshold = (h * w) * 0.005 # 0.5% of pixels
        
        if ai_skin_count < threshold:
            using_fallback = True
            box_size = min(h, w) // 2
            cx, cy = w//2, h//2
            x1 = max(0, cx - box_size//2)
            y1 = max(0, cy - box_size//2)
            x2 = min(w, cx + box_size//2)
            y2 = min(h, cy + box_size//2)
            
            fallback_mask = np.zeros((h, w), dtype=np.uint8)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                roi_mask = get_manual_skin_mask(roi)
                fallback_mask[y1:y2, x1:x2] = roi_mask
            final_skin_mask = fallback_mask

        # --- MODE 1: JAUNDICE (BODY/SKIN) ---
        if current_model == "JAUNDICE_BODY":
            if cv2.countNonZero(final_skin_mask) > (threshold/2): # Lower threshold for detection
                masked_roi = cv2.bitwise_and(frame, frame, mask=final_skin_mask)
                y_indices, x_indices = np.where(final_skin_mask > 0)
                
                if len(y_indices) > 0:
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    
                    # Pad
                    pad = 20
                    y_min, x_min = max(0, y_min-pad), max(0, x_min-pad)
                    y_max, x_max = min(h, y_max+pad), min(w, x_max+pad)
                    
                    cropped_skin = masked_roi[y_min:y_max, x_min:x_max]
                    
                    if cropped_skin.size > 0:
                        label, conf = predict_jaundice_model(cropped_skin)
                        
                        # Draw Box on Face
                        color = (0, 0, 255) if "Jaundice" in label else (0, 255, 0)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                        
                        # Update Dashboard
                        sidebar_result = label
                        sidebar_conf = f"Confidence: {conf*100:.1f}%"
            else:
                 sidebar_warning = "No Skin Detected"
                 if using_fallback:
                     cx, cy = w//2, h//2
                     cv2.rectangle(frame, (cx-100, cy-100), (cx+100, cy+100), (255, 255, 0), 2)

        # --- MODE 2: JAUNDICE (EYE/SCLERA) ---
        elif current_model == "JAUNDICE_EYE":
            # This mode requires Eyes AND Skin (Dual Input)
            # 1. Get Eyes
            eyes = seg_model.get_eye_rois(seg_mask, frame)
            
            # 2. Get Skin (we can use the whole face/skin crop)
            if cv2.countNonZero(final_skin_mask) > (threshold/2):
                masked_roi = cv2.bitwise_and(frame, frame, mask=final_skin_mask)
                y_indices, x_indices = np.where(final_skin_mask > 0)
                if len(y_indices) > 0:
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    skin_crop = masked_roi[y_min:y_max, x_min:x_max]
                else:
                    skin_crop = frame # Fallback to full frame
            else:
                skin_crop = frame # Fallback

            if eyes:
                # We might find multiple eyes, but the sidebar handles one main result cleanly.
                # Let's show the result of the "most confident" or just the first/last one.
                # For UI clarity, we'll list them sequentially or pick worst case.
                
                results = []
                for cropped_eye, name, (x1, y1, x2, y2) in eyes:
                    # 1. Blur Check (On original crop)
                    blur_score = calculate_blur(cropped_eye)
                    # If using uploaded image, we might relax the blur check or keep it
                    # Static images usually better quality, but still good to check.
                    if blur_score < 100.0: # Threshold for blur
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow for blur
                        results.append(("Blurry", 0.0))
                        continue

                    # 2. Iris Masking (Fix for Cornea Confusion)
                    # We mask the iris to force the model to look at sclera
                    masked_eye = seg_model.apply_iris_mask(cropped_eye)

                    # Model expects (Skin, Sclera)
                    label, conf = predict_sclera_model(skin_crop, masked_eye)
                    
                    color = (0, 0, 255) if "Jaundice" in label else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    results.append((label, conf))
                
                # Consolidate Results for Sidebar
                if not results:
                    sidebar_warning = "Eyes found but skipped?"
                else:
                    # Priority: Jaundice > Blurry > Normal
                    labels = [r[0] for r in results]
                    if "Jaundice" in labels:
                        sidebar_result = "Jaundice"
                        # Find max confidence for jaundice
                        confs = [r[1] for r in results if r[0] == "Jaundice"]
                        sidebar_conf = f"Max Conf: {max(confs)*100:.1f}%"
                    elif "Blurry" in labels:
                        sidebar_result = "Image Blurry"
                        sidebar_conf = "Hold Camera Still" if not is_static else "Image out of focus"
                        sidebar_warning = "Camera is out of focus" if not is_static else "Image is blurry"
                    else:
                        sidebar_result = "Normal"
                        confs = [r[1] for r in results]
                        sidebar_conf = f"Avg Conf: {sum(confs)/len(confs)*100:.1f}%"

            else:
                # FALLBACK: If uploaded image (Static) and no eyes found, 
                # assume it's a MACRO shot of an eye (Close-up).
                if is_static:
                     # Treat full frame as the eye
                     # 1. Blur Check
                     blur_score = calculate_blur(frame)
                     if blur_score < 100.0:
                         sidebar_result = "Image Blurry"
                         sidebar_warning = "Image is blurry"
                     else:
                         # 2. Iris Masking
                         masked_eye = seg_model.apply_iris_mask(frame)
                         
                         # 3. Predict (Skin = Frame, Sclera = Frame)
                         label, conf = predict_sclera_model(frame, masked_eye)
                         
                         sidebar_result = f"{label} (Macro)"
                         sidebar_conf = f"Conf: {conf*100:.1f}%"
                         
                         # Draw box around whole image to indicate selection
                         cv2.rectangle(frame, (0,0), (w-2, h-2), (0,255,0), 4)
                else:
                    sidebar_warning = "Align Eyes with Camera" if not is_static else "No eyes detected"

        # --- MODE 3: SKIN DISEASE ---
        elif current_model == "SKIN_DISEASE":
            if cv2.countNonZero(final_skin_mask) > (threshold/2):
                masked_roi = cv2.bitwise_and(frame, frame, mask=final_skin_mask)
                y_indices, x_indices = np.where(final_skin_mask > 0)
                if len(y_indices) > 0:
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    pad = 20
                    y_min, x_min = max(0, y_min-pad), max(0, x_min-pad)
                    y_max, x_max = min(h, y_max+pad), min(w, x_max+pad)
                    cropped_skin = masked_roi[y_min:y_max, x_min:x_max]
                    
                    if cropped_skin.size > 0:
                        label, conf = predict_skin_model(cropped_skin)
                        color = (0, 255, 0) if "Healthy" in label else (0, 0, 255)
                        
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                        
                        sidebar_result = label
                        sidebar_conf = f"Conf: {conf*100:.1f}%"
            else:
                sidebar_warning = "Show Skin Area"
                if using_fallback:
                     cx, cy = w//2, h//2
                     cv2.rectangle(frame, (cx-100, cy-100), (cx+100, cy+100), (255, 255, 0), 2)


        # --- DRAW FINAL UI ---
        # Instead of writing directly on 'frame', we compose the sidebar
        final_display = draw_sidebar(frame, sidebar_width, current_model, sidebar_result, sidebar_conf, sidebar_warning, using_fallback, is_static)

        cv2.imshow("Medical Assistant (Unified 3-Model)", final_display)

        keys = cv2.waitKey(1) & 0xFF
        if keys == ord('q'): 
            break
        elif keys == ord('m'): 
            mode_idx = (mode_idx + 1) % len(modes)
            print(f"Switched Mode to: {modes[mode_idx]}")
        elif keys == ord('u'): # Upload Image
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
            )
            if file_path:
                print(f"Loading image: {file_path}")
                img = cv2.imread(file_path)
                if img is not None:
                    uploaded_image = img
                    print("Image loaded successfully.")
                else:
                    print("Failed to load image.")
        elif keys == ord('c'): # Clear Image / Return to Camera
            print("Returning to Camera...")
            uploaded_image = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()