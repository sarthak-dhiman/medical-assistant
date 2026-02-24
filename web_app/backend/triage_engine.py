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
            "message": "Immediate medical attention required. Please seek urgent care or go to an emergency room."
        },
        "URGENT": {
            "labels": ["Jaundice", "Burns Detected", "Clubbing", "Blue_Finger", "Cataract"],
            "color": "bg-orange-500 text-white border-orange-400",
            "message": "Consult a healthcare provider soon. Do not delay medical assessment."
        },
        "ROUTINE": {
            "labels": [
                "Acne", "Atopic_Dermatitis", "Psoriasis", "Rosacea", 
                "Tinea_Fungal", "Urticaria_Hives", "Vitiligo", "Warts", 
                "Onychomycosis", "Nail Psoriasis", "Pitting", "Onychogryphosis"
            ],
            "color": "bg-yellow-500 text-black border-yellow-400",
            "message": "Observe and manage symptoms. Schedule a non-urgent doctor visit if symptoms worsen."
        },
        "HEALTHY": {
            "labels": ["Healthy Nail", "Healthy Skin", "Normal"],
            "color": "bg-green-500 text-white border-green-400",
            "message": "No immediate concerns detected. Maintain regular health checkups."
        }
    }

    @classmethod
    def evaluate(cls, label: str, confidence: float) -> dict:
        """
        Evaluates the prediction label and returns triage information.
        """
        # Edge cases and low confidence
        if not label or label in ["No Hand Detected", "No Skin on Body"]:
            return {
                "level": "UNKNOWN",
                "message": "Diagnosis unclear. Please try scanning again with better lighting and positioning.",
                "color": "bg-gray-600 text-white border-gray-400"
            }
        
        # We can implement a confidence threshold logic here, but for triage, 
        # it's safer to alert the user even if confidence is moderate for high-severity issues.
        if confidence < 0.4:
            return {
                "level": "OBSERVE",
                "message": "Low confidence prediction. Monitor the condition closely or retake the scan.",
                "color": "bg-purple-600 text-white border-purple-400"
            }

        for level, info in cls.SEVERITY_MAPPINGS.items():
            if any(val.lower() in label.lower() for val in info["labels"]):
                return {
                    "level": level,
                    "message": info["message"],
                    "color": info["color"]
                }
        
        # Fallback if label is not explicitly mapped
        return {
            "level": "ROUTINE",
            "message": "Condition detected. Please review recommendations and consider consulting a doctor.",
            "color": "bg-yellow-500 text-black border-yellow-400"
        }
