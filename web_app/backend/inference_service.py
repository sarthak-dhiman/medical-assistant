"""
Inference Service Layer
Encapsulates all ML inference logic, separating it from Celery task orchestration.
"""
import sys
from pathlib import Path
import logging
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from inference_pytorch import (
    predict_jaundice_eye,
    predict_jaundice_body,
    predict_skin_disease_torch
)
from inference_new_models import (
    predict_burns,
    predict_nail_disease,
    # predict_cataract removed
    predict_oral_cancer,
    predict_teeth_disease,
    predict_posture_from_landmarks
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
    
    @staticmethod
    def predict_burns(img_bgr, debug=False):
        """Predict burn detection"""
        try:
            return predict_burns(img_bgr, debug)
        except Exception as e:
            logger.error(f"Burns inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    @staticmethod
    def predict_posture(landmarks, debug=False):
        """Predict posture from landmarks"""
        try:
            return predict_posture_from_landmarks(landmarks, debug)
        except Exception as e:
            logger.error(f"Posture inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    

    
    @staticmethod
    def predict_nail_disease(img_bgr, debug=False):
        """Predict nail disease"""
        try:
            return predict_nail_disease(img_bgr, debug)
        except Exception as e:
            logger.error(f"Nail Disease inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    # predict_cataract method removed

    @staticmethod
    def predict_oral_cancer(img_bgr, debug=False):
        """Predict oral cancer from oral cavity image"""
        try:
            return predict_oral_cancer(img_bgr, debug)
        except Exception as e:
            logger.error(f"Oral cancer inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    @staticmethod
    def predict_teeth_disease(img_bgr, debug=False):
        """Predict teeth disease from teeth/mouth image"""
        try:
            return predict_teeth_disease(img_bgr, debug)
        except Exception as e:
            logger.error(f"Teeth disease inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    @staticmethod
    def get_recommendations(label):
        """
        Fetch recommendations and explanations from the knowledge base.
        
        Args:
            label: The detected disease label.
            
        Returns:
            dict: Recommendations, causes, and description.
        """
        kb_path = Path(__file__).resolve().parent / "knowledge_base.json"
        
        # Mapping variations to knowledge base keys
        label_mapping = {
            # Burns
            "Burns Detected": "Burns Detected",
            "Healthy Skin": "Healthy Skin",
            
            # Jaundice
            "Jaundice": "Jaundice",
            
            # Skin Diseases
            "Acne": "Acne",
            "Atopic Dermatitis": "Atopic_Dermatitis",
            "Atopic_Dermatitis": "Atopic_Dermatitis",
            "Psoriasis": "Psoriasis",
            "Rosacea": "Rosacea",
            "SkinCancer": "SkinCancer",
            "Tinea Fungal": "Tinea_Fungal",
            "Tinea_Fungal": "Tinea_Fungal",
            "Urticaria Hives": "Urticaria_Hives",
            "Urticaria_Hives": "Urticaria_Hives",
            "Vitiligo": "Vitiligo",
            "Warts": "Warts",
            
            # Nail Diseases
            "Acral Lentiginous Melanoma": "Acral_Lentiginous_Melanoma",
            "Blue Finger": "Blue_Finger",
            "Clubbing": "Clubbing",
            "Healthy Nail": "Healthy Nail",
            "Nail Psoriasis": "Nail_Psoriasis",
            "Onychogryphosis": "Onychogryphosis",
            "Onychomycosis": "Onychomycosis",
            "Pitting": "Pitting",
            
            # Cataract removed
            # Oral Cancer
            "Oral_Cancer": "Oral_Cancer"
        }
        
        # Try direct match -> clean match -> mapped match
        kb_key = label_mapping.get(label)
        if not kb_key:
             clean_label = label.replace(" ", "_")
             kb_key = label_mapping.get(clean_label) or clean_label
        
        try:
            if kb_path.exists():
                with open(kb_path, 'r') as f:
                    kb = json.load(f)
                return kb.get(kb_key)
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            
        return None
    



# Singleton instance
inference_service = InferenceService()
