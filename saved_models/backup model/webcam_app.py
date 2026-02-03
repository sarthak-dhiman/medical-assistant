import cv2
import numpy as np
import time
from inference import predict_frame as predict_jaundice
from inference_skin import predict_skin_disease

# --- CONFIGURATION ---
current_model = "JAUNDICE" 
use_enhancement = True # Default ON for better detection

# Load Optimized Haar Cascade
# minNeighbors=5 (Standard), minSize=(200, 200) ensures we ONLY detect faces relatively close (approx < 50cm)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def enhance_frame(img):
    """
    Improves video quality. 
    REMOVED sharpening (caused blur/noise).
    Using mild CLAHE for lighting balance only.
    """
    # 1. CLAHE (Lighting Fix) - Reduced clipLimit for softer look
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 2. Sharpening REMOVED to prevent "blurry/static" look
    
    return enhanced

def get_skin_mask(img):
    """
    Robust skin detection using YCbCr color space.
    """
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Widened Thresholds (Indian/Tan support)
    lower = np.array([0, 130, 70], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    
    mask = cv2.inRange(ycbcr, lower, upper)
    
    # Closing to fill holes
    kernel = np.ones((4,4), np.uint8) 
    mask = cv2.dilate(mask, kernel, iterations=2) 
    mask = cv2.erode(mask, kernel, iterations=1) 
    
    # Keep largest blob only
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [max_contour], -1, 255, -1)
        mask = clean_mask

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

def main():
    global current_model, use_enhancement
    print("🎥 Opening Webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return
    
    mode = "FACE" 
    print("✅ Logic Started (Optimized Haar Mode).")
    print("Controls:")
    print(" 'b' -> Toggle Body Mode")
    print(" 'f' -> Toggle Face Mode")
    print(" 'm' -> Switch Model")
    print(" 'e' -> Toggle Enhancement")
    print(" 'q' -> Quit")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Enhance First! (Helps detection)
        if use_enhancement:
            frame = enhance_frame(frame)

        display_frame = frame.copy()
        final_roi = None
        h, w, _ = frame.shape
        
        # --- MODE LOGIC ---
        if mode == "FACE":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detectMultiScale parameters:
            # scaleFactor=1.1
            # minNeighbors=5
            # minSize=(180, 180): ONLY detect faces close to camera
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(180, 180))
            
            if len(faces) > 0:
                # Find largest face
                max_area = 0
                longest_face = None
                for (x, y, wa, ha) in faces:
                    if wa * ha > max_area:
                        max_area = wa * ha
                        longest_face = (x, y, wa, ha)
                
                if longest_face:
                    x, y, wa, ha = longest_face
                    padding = 50
                    y1, y2 = max(0, y-padding), min(h, y+ha+padding)
                    x1, x2 = max(0, x-padding), min(w, x+wa+padding)
                    
                    final_roi = frame[y1:y2, x1:x2]
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(display_frame, "Face Detected", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Scanning...", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        elif mode == "BODY":
            cx, cy = w//2, h//2
            box_w, box_h = w//2, h//2
            x1, y1 = cx - box_w//2, cy - box_h//2
            x2, y2 = cx + box_w//2, cy + box_h//2
            
            final_roi = frame[y1:y2, x1:x2]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # --- PREDICTION LOGIC ---
        if final_roi is not None and final_roi.size > 0:
            skin_mask = get_skin_mask(final_roi)
            masked_roi = cv2.bitwise_and(final_roi, final_roi, mask=skin_mask)
            
            skin_pixels = cv2.countNonZero(skin_mask)
            total_pixels = final_roi.shape[0] * final_roi.shape[1]
            
            if skin_pixels > total_pixels * 0.05:
                try:
                    if current_model == "JAUNDICE":
                        label, confidence = predict_jaundice(masked_roi)
                        color = (0, 0, 255) if label == "Jaundice" else (0, 255, 0)
                    else:
                        label, confidence = predict_skin_disease(masked_roi)
                        np.random.seed(len(label))
                        color = tuple(np.random.randint(0, 255, 3).tolist())
                    
                    text = f"{label}: {confidence*100:.1f}%"
                    
                    # ROI Visual
                    roi_display = cv2.resize(masked_roi, (150, 150))
                    try:
                        display_frame[h-160:h-10, w-160:w-10] = roi_display
                    except: pass
                    
                    cv2.putText(display_frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                1.5, color, 3, cv2.LINE_AA)
                    cv2.putText(display_frame, f"Model: {current_model}", (50, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                except:
                    pass
            else:
                 cv2.putText(display_frame, "No Skin Detected", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(display_frame, f"MODE: {mode} | MODEL: {current_model} | HD: {'ON' if use_enhancement else 'OFF'}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Smart Skin Analyzer', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('b'): mode = "BODY"
        elif key == ord('f'): mode = "FACE"
        elif key == ord('m'): current_model = "SKIN" if current_model == "JAUNDICE" else "JAUNDICE"
        elif key == ord('e'): use_enhancement = not use_enhancement
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
