"""
Inference Service Layer
Encapsulates all ML inference logic, separating it from Celery task orchestration.
"""
import sys
from pathlib import Path
import logging
import json
import time
from prometheus_client import Histogram

# Initialize Prometheus Metric for Latency
MODEL_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Time taken for purely ML inference (excluding Celery overhead)",
    ["model_name"]
)

def track_latency(model_name):
    """Decorator to measure and record execution time of inference models."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                MODEL_LATENCY.labels(model_name=model_name).observe(duration)
        return wrapper
    return decorator

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
    predict_oral_cancer,
    predict_teeth_disease,
    predict_posture_from_landmarks
)

# Import LLM Service
try:
    from llm_service import llm_service
except ImportError:
    llm_service = None


logger = logging.getLogger(__name__)


class InferenceService:
    """
    Service layer for ML inference operations.
    Provides a clean interface for prediction tasks.
    """
    
    @staticmethod
    @track_latency("jaundice_eye")
    def predict_jaundice_eye(skin_img, sclera_crop=None, debug=False, calibrate=False):
        """Predict jaundice from eye/sclera image."""
        try:
            return predict_jaundice_eye(skin_img, sclera_crop, debug, calibrate)
        except Exception as e:
            logger.error(f"Jaundice Eye inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    @staticmethod
    @track_latency("jaundice_body")
    def predict_jaundice_body(img_bgr, debug=False, calibrate=False):
        """Predict jaundice from body/skin image."""
        try:
            return predict_jaundice_body(img_bgr, debug, calibrate)
        except Exception as e:
            logger.error(f"Jaundice Body inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    @staticmethod
    @track_latency("skin_disease")
    def predict_skin_disease(img_bgr, debug=False, calibrate=False):
        """Predict skin disease from image."""
        try:
            return predict_skin_disease_torch(img_bgr, debug, calibrate)
        except Exception as e:
            logger.error(f"Skin Disease inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    @staticmethod
    @track_latency("burns")
    def predict_burns(img_bgr, debug=False, calibrate=False):
        """Predict burn detection"""
        try:
            return predict_burns(img_bgr, debug, calibrate)
        except Exception as e:
            logger.error(f"Burns inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    @staticmethod
    @track_latency("posture")
    def predict_posture(landmarks, debug=False):
        """Predict posture from landmarks"""
        try:
            return predict_posture_from_landmarks(landmarks, debug)
        except Exception as e:
            logger.error(f"Posture inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    @staticmethod
    @track_latency("cataract")
    def predict_cataract(img_bgr, debug=False, calibrate=False):
        """Predict cataract (Not available in current model set)"""
        return "Model Not Available", 0.0, {"error": "Cataract model is currently missing from saved_models"}

    @staticmethod
    @track_latency("nail_disease")
    def predict_nail_disease(img_bgr, debug=False, calibrate=False):
        """Predict nail disease"""
        try:
            return predict_nail_disease(img_bgr, debug, calibrate)
        except Exception as e:
            logger.error(f"Nail Disease inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}

    @staticmethod
    @track_latency("oral_cancer")
    def predict_oral_cancer(img_bgr, debug=False, calibrate=False, preprocessed=False):
        """Predict oral cancer from oral cavity image"""
        try:
            return predict_oral_cancer(img_bgr, debug, calibrate, preprocessed=preprocessed)
        except Exception as e:
            logger.error(f"Oral cancer inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    @staticmethod
    @track_latency("teeth_disease")
    def predict_teeth_disease(img_bgr, debug=False, calibrate=False, preprocessed=False):
        """Predict teeth disease from teeth/mouth image"""
        try:
            return predict_teeth_disease(img_bgr, debug, calibrate, preprocessed=preprocessed)
        except Exception as e:
            logger.error(f"Teeth disease inference failed: {e}")
            return "Error", 0.0, {"error": str(e)}
    
    @staticmethod
    def get_recommendations(label: str, mode: str = None, patient_history: str = None):
        """
        Fetch recommendations and explanations from the knowledge base.
        """
        kb_path = Path(__file__).resolve().parent / "knowledge_base.json"
        
        label_mapping = {
            "Burns Detected": "Burns Detected",
            "Healthy Skin": "Healthy Skin",
            "Jaundice": "Jaundice",
            "Acne": "Acne",
            "Atopic_Dermatitis": "Atopic_Dermatitis",
            "Psoriasis": "Psoriasis",
            "Rosacea": "Rosacea",
            "SkinCancer": "SkinCancer",
            "Tinea_Fungal": "Tinea_Fungal",
            "Urticaria_Hives": "Urticaria_Hives",
            "Vitiligo": "Vitiligo",
            "Warts": "Warts",
            "Acral_Lentiginous_Melanoma": "Acral_Lentiginous_Melanoma",
            "Blue_Finger": "Blue_Finger",
            "Clubbing": "Clubbing",
            "Healthy Nail": "Healthy Nail",
            "Nail_Psoriasis": "Nail_Psoriasis",
            "Onychogryphosis": "Onychogryphosis",
            "Onychomycosis": "Onychomycosis",
            "Pitting": "Pitting",
            "Oral_Cancer": "Oral_Cancer"
        }
        
        # Handle generic labels mapped based on the diagnostic mode
        if label == "Normal":
            if mode == "ORAL_CANCER":
                kb_key = "Normal_Oral"
            elif mode == "CATARACT":
                kb_key = "Normal_Cataract"
            else:
                kb_key = "Normal"
        elif label == "Healthy":
            if mode == "TEETH":
                kb_key = "Healthy_Teeth"
            else:
                kb_key = "Healthy"
        else:
            kb_key = label_mapping.get(label)
            if not kb_key:
                 clean_label = label.replace(" ", "_")
                 kb_key = label_mapping.get(clean_label) or clean_label
        
        # 1. Attempt Dynamic LLM Explanation
        if llm_service and llm_service.is_ready:
            try:
                llm_explanation = llm_service.generate_explanation(label, mode, patient_history)
                if llm_explanation:
                    return llm_explanation
            except Exception as e:
                logger.error(f"LLM explanation failed, falling back to static KB: {e}")

        # 2. Fallback to Static Knowledge Base
        try:
            if kb_path.exists():
                with open(kb_path, 'r') as f:
                    kb = json.load(f)
                return kb.get(kb_key)
        except Exception as e:
            logger.error(f"Failed to load static knowledge base: {e}")
            
        return None

# Singleton instance
inference_service = InferenceService()
