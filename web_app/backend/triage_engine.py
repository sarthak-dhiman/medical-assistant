import logging

logger = logging.getLogger(__name__)

# Triage Definitions
# Structure: {
#   "DISEASE_LABEL": {
#       "severity": "CRITICAL" | "URGENT" | "MONITOR" | "ROUTINE" | "NORMAL",
#       "color": "red" | "orange" | "yellow" | "blue" | "green",
#       "action": "Immediate clinical referral required."
#   }
# }

TRIAGE_RULES = {
    # CRITICAL (Red) - Requires immediate or fast-tracked intervention
    "Melanoma": {"severity": "CRITICAL", "color": "red", "action": "Immediate dermatology referral required. High risk of malignancy."},
    "Basal Cell Carcinoma": {"severity": "CRITICAL", "color": "red", "action": "Urgent dermatology referral for biopsy."},
    "Squamous Cell Carcinoma": {"severity": "CRITICAL", "color": "red", "action": "Urgent dermatology referral for biopsy."},
    "Oral Squamous Cell Carcinoma (OSCC)": {"severity": "CRITICAL", "color": "red", "action": "Immediate oncology/maxillofacial referral required."},
    
    # URGENT (Orange) - Serious condition requiring timely medical assessment
    "Jaundice": {"severity": "URGENT", "color": "orange", "action": "Urgent liver function test and medical assessment required."},
    "Third Degree Burn": {"severity": "URGENT", "color": "orange", "action": "Immediate trauma or burn center evaluation required."},
    "Onychomycosis": {"severity": "URGENT", "color": "orange", "action": "Dermatology/Podiatry referral for anti-fungal treatment plan."},
    "Oral Lichen Planus": {"severity": "URGENT", "color": "orange", "action": "Dental/Specialist evaluation required to monitor for malignant transformation."},
    "Dental Caries": {"severity": "URGENT", "color": "orange", "action": "Urgent dental appointment to prevent abscess or systemic infection."},
    "Periodontitis": {"severity": "URGENT", "color": "orange", "action": "Urgent periodontist evaluation to prevent tooth loss."},
    
    # MONITOR (Yellow) - Needs attention but not immediately life-threatening
    "Second Degree Burn": {"severity": "MONITOR", "color": "yellow", "action": "Monitor closely for infection. Use sterile dressings."},
    "Actinic Keratosis": {"severity": "MONITOR", "color": "yellow", "action": "Pre-cancerous. Schedule routine dermatology screening."},
    "Gingivitis": {"severity": "MONITOR", "color": "yellow", "action": "Improve dental hygiene routine. Schedule dentist cleaning."},
    "Tooth Discoloration": {"severity": "MONITOR", "color": "yellow", "action": "Schedule routine dental assessment. Often cosmetic but can indicate pulp necrosis."},
    "Beau's Lines": {"severity": "MONITOR", "color": "yellow", "action": "Investigate potential recent systemic illness or trauma."},
    "Muehrcke's Lines": {"severity": "MONITOR", "color": "yellow", "action": "Evaluate for systemic condition, particularly hypoalbuminemia."},
    
    # ROUTINE (Blue) - Common, benign, or manageable conditions
    "Acne": {"severity": "ROUTINE", "color": "blue", "action": "Routine symptom management. OTC treatments or standard dermatology consult."},
    "First Degree Burn": {"severity": "ROUTINE", "color": "blue", "action": "Apply cool compress. Routine OTC pain management."},
    "Aphthous Ulcer": {"severity": "ROUTINE", "color": "blue", "action": "Usually resolves spontaneously in 1-2 weeks. Manage pain with OTC gels."},
    "Terry's Nails": {"severity": "ROUTINE", "color": "blue", "action": "Typically benign aging change, but monitor if patient has underlying illness."},
    "Pitting": {"severity": "ROUTINE", "color": "blue", "action": "Commonly associated with psoriasis. Routine dermatology follow-up."},
    "Splinter Hemorrhage": {"severity": "ROUTINE", "color": "blue", "action": "Often trauma-related. Monitor. If multiple, assess for endocarditis."},
    "Leukonychia": {"severity": "ROUTINE", "color": "blue", "action": "Often benign trauma. Monitor for changes."},
    
    # NORMAL (Green) - No disease detected
    "Normal": {"severity": "NORMAL", "color": "green", "action": "No immediate action required. Maintain routine health checks."},
    "Healthy Hand": {"severity": "NORMAL", "color": "green", "action": "No immediate action required."},
    "Unknown Class": {"severity": "MONITOR", "color": "yellow", "action": "Algorithm could not confidently classify. Recommend manual physician review."}
}

# Downgrade logic for uncertain but scary predictions
def evaluate_prediction(mode: str, label: str, confidence: float) -> dict:
    """
    Evaluates ML output against clinical rules to generate a triage action plan.
    """
    try:
        # 1. Look up base rule
        rule = TRIAGE_RULES.get(label)
        
        # 2. Fallback if exact label isn't mapped
        if not rule:
            # Handle dynamic labels (e.g. Unknown Class 4)
            if "Unknown" in label or "Normal" in label:
                if "Normal" in label:
                    rule = TRIAGE_RULES["Normal"]
                else:
                    rule = TRIAGE_RULES["Unknown Class"]
            else:
                logger.warning(f"Triage rule missing for label: {label} in mode {mode}")
                # Default "safe" fallback for unmapped diseases
                rule = {
                    "severity": "MONITOR", 
                    "color": "yellow", 
                    "action": "Unmapped condition detected. Clinical override recommended."
                }
        
        # Create a mutable copy of the rule to allow overrides
        triage_status = dict(rule)
        
        # 3. Confidence Safety Throttling (Downgrade)
        # Prevent panic alarms if the model is returning a CRITICAL label but is uncertain
        if triage_status["severity"] == "CRITICAL" and confidence < 0.85:
            logger.info(f"Triage Engine: Downgrading CRITICAL {label} to URGENT due to low confidence ({confidence:.2f})")
            triage_status["severity"] = "URGENT"
            triage_status["color"] = "orange"
            triage_status["action"] = f"Warning (Low AI Confidence): Suspected {label}, but requires manual physician confirmation before initiating critical protocols."
            
        elif triage_status["severity"] == "URGENT" and confidence < 0.70:
            logger.info(f"Triage Engine: Downgrading URGENT {label} to MONITOR due to low confidence ({confidence:.2f})")
            triage_status["severity"] = "MONITOR"
            triage_status["color"] = "yellow"
            triage_status["action"] = f"Warning (Low AI Confidence): Possible {label}. Monitor closely and schedule non-urgent clinical review."

        return triage_status
        
    except Exception as e:
        logger.error(f"Triage Engine failed evaluating {label}: {e}")
        # Fail-safe response
        return {
            "severity": "MONITOR",
            "color": "yellow",
            "action": "Error during clinical triage evaluation. Please rely on manual medical review."
        }
