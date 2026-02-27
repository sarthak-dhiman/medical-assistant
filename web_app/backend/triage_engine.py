import logging

logger = logging.getLogger(__name__)

class ClinicalTriageEngine:
    """
    Evaluates predictions and assigns priority levels (EMERGENCY, URGENT, ROUTINE, HEALTHY).
    """

    SEVERITY_MAPPINGS = {
        "EMERGENCY": {
            "labels": ["Acral_Lentiginous_Melanoma", "SkinCancer", "Oral_Cancer"],
            "color": "bg-red-600 text-white border-red-400",
            "message": "Immediate medical attention required. Please seek urgent care or go to an emergency room.",
            "threshold": 0.35 # Flag even low confidence for life-threatening
        },
        "URGENT": {
            "labels": ["Jaundice", "Burns Detected", "Clubbing", "Blue_Finger", "Cataract"],
            "color": "bg-orange-500 text-white border-orange-400",
            "message": "Consult a healthcare provider soon. Do not delay medical assessment.",
            "threshold": 0.50
        },
        "ROUTINE": {
            "labels": [
                "Acne", "Atopic_Dermatitis", "Psoriasis", "Rosacea", 
                "Tinea_Fungal", "Urticaria_Hives", "Vitiligo", "Warts", 
                "Onychomycosis", "Nail_Psoriasis", "Pitting", "Onychogryphosis",
                "Caries", "Gingivitis", "Tooth Discoloration", "Ulcer",
                "Poor Posture (Kyphosis Trigger)", "Poor Posture (Forward Head)"
            ],
            "color": "bg-yellow-500 text-black border-yellow-400",
            "message": "Observe and manage symptoms. Schedule a non-urgent doctor visit if symptoms worsen.",
            "threshold": 0.60
        },
        "HEALTHY": {
            "labels": ["Healthy Nail", "Healthy Skin", "Normal", "Healthy_Teeth", "Healthy", "Good Posture"],
            "color": "bg-green-500 text-white border-green-400",
            "message": "No immediate concerns detected. Maintain regular health checkups.",
            "threshold": 0.65 # Require high confidence to declare healthy
        }
    }

    @classmethod
    def evaluate(cls, label: str, confidence: float) -> dict:
        """
        Evaluates the prediction label and returns triage information based on severity and dynamic confidence.
        """
        # Edge cases and low confidence
        if not label or label in ["No Hand Detected", "No Skin on Body", "No Posture Detected", "Acquiring Target..."]:
            return {
                "level": "UNKNOWN",
                "message": "Diagnosis unclear. Please try scanning again with better lighting and positioning.",
                "color": "bg-gray-600 text-white border-gray-400"
            }
        
        # Ensure exact match or partial match safely
        # Note: label from models might have spaces or underscores, normalize for checking
        normalized_label = label.lower().replace(" ", "_")

        matched_level = None
        matched_info = None

        for level, info in cls.SEVERITY_MAPPINGS.items():
            for map_label in info["labels"]:
                norm_map_label = map_label.lower().replace(" ", "_")
                if norm_map_label in normalized_label or normalized_label in norm_map_label:
                    matched_level = level
                    matched_info = info
                    break
            if matched_level:
                break
        
        if matched_level and matched_info:
            threshold = matched_info.get("threshold", 0.5)
            
            # 1. If confidence meets the severity threshold, assign that exact level.
            if confidence >= threshold:
                return {
                    "level": matched_level,
                    "message": matched_info["message"],
                    "color": matched_info["color"]
                }
            
            # 2. If confidence is below threshold, but it's an EMERGENCY, still flag as URGENT (Downgrade safety net)
            elif matched_level == "EMERGENCY" and confidence >= 0.20:
                 return {
                    "level": "URGENT",
                    "message": f"Possible signs of {label} detected (Low Confidence). A professional evaluation is highly recommended.",
                    "color": cls.SEVERITY_MAPPINGS["URGENT"]["color"]
                }
            
            # 3. For ROUTINE or HEALTHY where confidence is low, fallback to OBSERVE
            return {
                "level": "OBSERVE",
                "message": f"Low confidence prediction ({label}). Monitor the condition closely or retake the scan with better focus.",
                "color": "bg-purple-600 text-white border-purple-400"
            }

        # Fallback if label is not explicitly mapped
        if confidence < 0.5:
             return {
                "level": "OBSERVE",
                "message": "Low confidence prediction. Monitor the condition closely or retake the scan.",
                "color": "bg-purple-600 text-white border-purple-400"
            }
            
        return {
            "level": "ROUTINE",
            "message": "Condition detected. Please review recommendations and consider consulting a doctor.",
            "color": "bg-yellow-500 text-black border-yellow-400"
        }
