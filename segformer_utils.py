import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import sys
import os



class SegFormerWrapper:
    def __init__(self, model_name="jonathandinu/face-parsing"):
        """
        Initializes the Face Parsing model (SegFormer/ConvNeXt-based).
        Default: jonathandinu/face-parsing (19 classes)
        """
        try:
            print(f"Loading Face Parsing Model: {model_name}...")
            
            # 1. Determine Local Path
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.getcwd()
            local_path = os.path.join(base_dir, "saved_models", "segformer")

            # 2. Check if VALID local model exists (must have weights)
            has_local_weights = False
            if os.path.exists(local_path):
                files = os.listdir(local_path)
                # Check for any standard weight file
                weight_files = ['model.safetensors', 'pytorch_model.bin', 'tf_model.h5']
                if any(w in files for w in weight_files):
                    has_local_weights = True
            
            # 3. Load
            if has_local_weights:
                print(f"Loading from local path: {local_path}")
                self.processor = SegFormerImageProcessor.from_pretrained(local_path)
                self.model = AutoModelForSemanticSegmentation.from_pretrained(local_path)
            else:
                # Fallback to HuggingFace
                print(f"Local model missing or incomplete. Downloading {model_name}...")
                self.processor = SegformerImageProcessor.from_pretrained(model_name)
                self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
                
                # Optional: Save it for next time (Create directory if needed)
                if not os.path.exists(local_path):
                    os.makedirs(local_path, exist_ok=True)
                # We don't auto-save here to avoid slowing down, but user can manually copy if needed.
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
            
            # Standard Face Parsing Labels (BiSeNet/CelebAMask-HQ convention)
            # 0: background
            # 1: skin (face)
            # 2: l_brow, 3: r_brow
            # 4: l_eye, 5: r_eye
            # 6: glasses (sometimes)
            # 7: l_ear, 8: r_ear
            # 9: ear_r (duplicate?), 
            # 10: nose
            # 11: mouth
            # 12: u_lip, 13: l_lip
            # 14: neck
            # 15: neck_l (clothes?)
            # 16: cloth
            # 17: hair
            # 18: hat
            self.labels = {
                'background': 0, 'skin': 1, 'l_brow': 2, 'r_brow': 3, 
                'l_eye': 4, 'r_eye': 5, 'eye_g': 6, 'l_ear': 7, 
                'r_ear': 8, 'ear_r': 9, 'nose': 10, 'mouth': 11, 
                'u_lip': 12, 'l_lip': 13, 'neck': 14, 'neck_l': 15, 
                'cloth': 16, 'hair': 17, 'hat': 18
            }
            
        except Exception as e:
            print(f"ERROR Loading SegFormer: {e}")
            self.model = None
    
    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, image):
        """
        Runs inference on a single image (numpy array BGR or RGB).
        Returns class mask (H, W).
        """
        if self.model is None:
            # Return an empty mask if the model failed to load
            if isinstance(image, np.ndarray):
                return np.zeros(image.shape[:2], dtype=np.uint8)
            elif isinstance(image, Image.Image):
                return np.zeros(image.size[::-1], dtype=np.uint8)
            else:
                raise ValueError("Unsupported image type. Must be numpy array or PIL Image.")

        # Convert to PIL (RGB)
        if isinstance(image, np.ndarray):
            # Assume BGR if 3 channels, convert to RGB
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            
        inputs = self.processor(images=image_pil, return_tensors="pt")
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Ensure model is on device (lazy move)
        if self.model.device.type != self.device:
            self.model.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits  # shape (1, num_labels, H/4, W/4)
        
        # Upsample logits to original size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image_pil.size[::-1], # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        
        # Argmax to get labels
        pred_seg = upsampled_logits.argmax(dim=1)[0] # (H, W)
        
        return pred_seg.cpu().numpy().astype(np.uint8)

    def get_eye_rois(self, mask, image_bgr):
        """
        Extracts Left and Right Eye crops from the segmentation mask.
        Returns: list of (crop, label_name) tuples
        """
        rois = []
        h, w = mask.shape
        
        # Labels for eyes
        eye_classes = {'Left Eye': 4, 'Right Eye': 5}
        
        for name, idx in eye_classes.items():
            # Create binary mask for this eye
            eye_mask = (mask == idx).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get max contour (largest eye part)
                c = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(c)
                
                # Add padding (eyes are often too tight in segmentation)
                pad = int(max(cw, ch) * 0.5) 
                x1 = max(0, x - pad)
                y1 = max(0, y - int(pad*0.5))
                x2 = min(w, x + cw + pad)
                y2 = min(h, y + ch + int(pad*0.5))
                
                crop = image_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    rois.append((crop, name, (x1, y1, x2, y2)))
                    
        return rois

    def apply_iris_mask(self, eye_img):
        """
        Detects the iris using Hough Circles (robust to blur) or Fallback Thresholding.
        Returns the image with the iris blacked out.
        """
        if eye_img is None or eye_img.size == 0:
            return eye_img
            
        h, w = eye_img.shape[:2]
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        
        # --- STRATEGY 1: Hough Circle Transform (Best for Blurry/Circular shapes) ---
        # Blur slightly to reduce noise
        blurred_hough = cv2.medianBlur(gray, 5)
        
        # HoughCircles params:
        # dp=1: input resolution
        # minDist=w/2: assume only one iris per crop
        # param1=50: Canny high threshold (lower because blur reduces gradients)
        # param2=30: Accumulator threshold (lower = more circles detected)
        circles = cv2.HoughCircles(
            blurred_hough, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=w/2,
            param1=50, 
            param2=30,
            minRadius=int(min(h,w)*0.10), # Iris is at least 10%
            maxRadius=int(min(h,w)*0.55)  # Iris is at most 55%
        )
        
        mask = np.ones_like(eye_img) * 255
        
        if circles is not None:
             circles = np.uint16(np.around(circles))
             # Take the strongest circle (first one)
             for i in circles[0,:]:
                 cx, cy, r = i[0], i[1], i[2]
                 
                 # Sanity check: Circle center must be somewhat central
                 if abs(cx - w//2) < w*0.4 and abs(cy - h//2) < h*0.4:
                     # Draw black circle on mask
                     # Dilate slightly (10%) to ensure coverage
                     cv2.circle(mask, (cx, cy), int(r*1.1), (0,0,0), -1)
                     
                     # SUCCESS: Used Hough
                     return cv2.bitwise_and(eye_img, mask), "hough"

        # --- STRATEGY 2 (Fallback): Contrast + Contours (Previous Logic) ---
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
        
        # Otsu Thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Safety Check: If OTSU makes everything white (bad lighting)
        white_pixel_ratio = cv2.countNonZero(thresh) / (thresh.shape[0] * thresh.shape[1])
        if white_pixel_ratio > 0.6: 
             _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_c = None
        max_area = 0
        center = (w//2, h//2)
        
        if contours:
            for c in contours:
                area = cv2.contourArea(c)
                # Filter noise
                if area < (h*w)*0.02: continue 
                # Filter massive blobs (likely the whole frame/socket)
                if area > (h*w)*0.55: continue

                # Check circularity (optional but good for iris)
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                # Check distance to center
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist = np.sqrt((cx-center[0])**2 + (cy-center[1])**2)
                    
                    # We want blob near center
                    if dist < w*0.4: 
                        # Prefer more circular + larger
                        score = area * (1 + circularity)
                        if score > max_area:
                            max_area = score
                            best_c = c
            
            if best_c is not None:
                 # Draw the iris contour as BLACK on the mask (Remove it)
                 cv2.drawContours(mask, [best_c], -1, (0, 0, 0), -1)
                 
                 # Draw a circle around it to be sure (dilate the removal)
                 (x,y), radius = cv2.minEnclosingCircle(best_c)
                 center_c = (int(x), int(y))
                 
                 # Constraint Radius (Prevent massive blobs)
                 max_radius = min(h, w) * 0.45
                 radius_c = int(min(radius * 1.1, max_radius)) 
                 
                 cv2.circle(mask, center_c, radius_c, (0, 0, 0), -1)

        # Apply mask
        eye_masked = cv2.bitwise_and(eye_img, mask)
        # Fallback used
        return eye_masked, "fallback"

    def get_skin_mask(self, mask):
        """
        Returns a binary mask for skin areas.
        Includes morphology to fill holes and remove noise (color cards).
        """
        # Skin (1), L_Ear (7), R_Ear (8), Neck (14)
        target_classes = [1, 7, 8, 14] 
        
        skin_mask = np.isin(mask, target_classes).astype(np.uint8) * 255
        
        # 1. Fill Holes (MORPH_CLOSE) - Fixes "black blobs on skin"
        # Closing = Dilation followed by Erosion. It closes small holes.
        kernel_close = np.ones((5,5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 2. Remove Noise (MORPH_OPEN) - Removes small speckles
        # Opening = Erosion followed by Dilation.
        kernel_open = np.ones((3,3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open)

        # 3. Largest Contour Filter - Removes Color Cards/checkered patterns
        # Assumption: The Face/Neck is the largest skin-colored object in the frame.
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sort by area (descending)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Create a new blank mask
            filtered_mask = np.zeros_like(skin_mask)
            
            # Draw ONLY the largest contour
            # (We could keeping top 2 if they are close in size, but usually face is one blob)
            cv2.drawContours(filtered_mask, [contours[0]], -1, 255, -1)
            
            return filtered_mask
            
        return skin_mask
