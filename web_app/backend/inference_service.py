"""
Inference Service Layer
Encapsulates all ML inference logic, separating it from Celery task orchestration.
"""
import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from inference_pytorch import (
    predict_jaundice_eye,
    predict_jaundice_body,
    predict_skin_disease_torch
)

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Service layer for ML inference operations.
    Provides a clean interface for prediction tasks.
    """
    
    @staticmethod
    def predict_jaundice_eye(skin_img, sclera_crop=None, debug=False):
        """
        Predict jaundice from eye/sclera image.
        
        Args:
            skin_img: BGR image of face/skin region
            sclera_crop: Optional BGR crop of sclera region
            debug: Enable Grad-CAM and debug stats
            
        Returns:
            tuple: (label, confidence, debug_info)
        """
        try:
            return predict_jaundice_eye(skin_img, sclera_crop, debug)
        except Exception as e:
            logger.error(f"Jaundice Eye inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    @staticmethod
    def predict_jaundice_body(img_bgr, debug=False):
        """
        Predict jaundice from body/skin image.
        
        Args:
            img_bgr: BGR image of body/skin region
            debug: Enable Grad-CAM and debug stats
            
        Returns:
            tuple: (label, confidence, debug_info)
        """
        try:
            return predict_jaundice_body(img_bgr, debug)
        except Exception as e:
            logger.error(f"Jaundice Body inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    @staticmethod
    def predict_skin_disease(img_bgr, debug=False):
        """
        Predict skin disease from image.
        
        Args:
            img_bgr: BGR image of affected skin area
            debug: Enable Grad-CAM and debug stats
            
        Returns:
            tuple: (label, confidence, debug_info)
        """
        try:
            return predict_skin_disease_torch(img_bgr, debug)
        except Exception as e:
            logger.error(f"Skin Disease inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}


# Singleton instance
inference_service = InferenceService()
