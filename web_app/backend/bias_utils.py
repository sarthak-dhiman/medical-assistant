import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_ita_category(image_bgr: np.ndarray, mask: np.ndarray = None) -> str:
    """
    Calculates the Individual Typology Angle (ITA) for skin tone categorization 
    using the LAB color space.

    Formula: ITA = arctan((L* - 50) / b*) * (180 / PI)
    Categories:
        > 55 : Very Light
        > 41 to 55 : Light
        > 28 to 41 : Intermediate
        > 10 to 28 : Tan
        > -30 to 10 : Brown
        <= -30 : Dark
    """
    try:
        # Convert BGR to LAB
        lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab_image)
        
        # OpenCV LAB ranges are scaled:
        # L is 0-255 (actual 0-100)
        # a, b are 0-255 (actual -127 to 127)
        L_std = L.astype(np.float32) * (100.0 / 255.0)
        b_std = b.astype(np.float32) - 128.0
        
        if mask is not None and np.any(mask > 0):
            valid_pixels = mask > 0
            L_mean = np.mean(L_std[valid_pixels])
            b_mean = np.mean(b_std[valid_pixels])
        else:
            # Fallback: center crop proxy
            h, w = image_bgr.shape[:2]
            crop_L = L_std[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
            crop_b = b_std[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
            L_mean = np.mean(crop_L)
            b_mean = np.mean(crop_b)
            
        # Prevent division by zero
        if b_mean == 0:
            b_mean = 0.0001
            
        ita = np.arctan((L_mean - 50.0) / b_mean) * (180.0 / np.pi)
        
        if ita > 55:
            return "Very Light"
        elif ita > 41:
            return "Light"
        elif ita > 28:
            return "Intermediate"
        elif ita > 10:
            return "Tan"
        elif ita > -30:
            return "Brown"
        else:
            return "Dark"
            
    except Exception as e:
        logger.error(f"Error calculating ITA: {e}")
        return "Unknown"
