"""
LLM Service Layer
Handles dynamic medical explanations and recommendations using Google Gemini.
"""
import os
import json
import logging
import time
from prometheus_client import Histogram

logger = logging.getLogger(__name__)

# Try importing the Gemini SDK; gracefully handle if not installed or configured
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    logger.warning("google-generativeai is not installed. LLM Explanations will be disabled.")

# Metrics
LLM_LATENCY = Histogram(
    "llm_inference_latency_seconds",
    "Time taken for LLM text generation",
)

class LLMService:
    """
    Service for generating dynamic, context-aware explanations using an LLM.
    """
    
    def __init__(self):
        self._is_ready = False
        self._model = None
        
        if not HAS_GEMINI:
            return
            
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment. LLM Explanations disabled.")
            return
            
        is_enabled = os.getenv("ENABLE_LLM_EXPLANATIONS", "False").lower() == "true"
        if not is_enabled:
             logger.info("ENABLE_LLM_EXPLANATIONS is false. LLM Explanations disabled.")
             return

        try:
            genai.configure(api_key=api_key)
            # Use gemini-2.5-flash for speed or gemini-1.5-pro for better reasoning
            self._model = genai.GenerativeModel('gemini-2.5-flash') 
            self._is_ready = True
            logger.info("LLM Service (Gemini) initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")

    @property
    def is_ready(self):
        return self._is_ready

    def generate_explanation(self, label: str, mode: str, patient_history: str = None) -> dict:
        """
        Generates a structured medical explanation for a given condition.
        Returns a dict matching the knowledge_base.json format or None on failure.
        """
        if not self._is_ready:
            return None
            
        start_time = time.time()
        
        history_context = f"\nPatient reported history/symptoms: '{patient_history}'" if patient_history else ""

        prompt = f"""
        You are a helpful, professional AI medical assistant integrated into a clinical diagnostic app.
        The AI vision model has just analyzed an image (context: {mode}) and predicted: "{label}".{history_context}
        
        Provide a concise, patient-friendly explanation. If patient history is provided, gently acknowledge it in the context of the prediction.
        You MUST align exactly with this JSON format. Do not include markdown formatting like ```json or newlines outside the JSON format. Return purely the raw JSON string:
        {{
            "description": "A 1-2 sentence overview of what the condition is.",
            "causes": "A brief sentence on common causes.",
            "care": "1-2 brief bullet points on immediate at-home care or next steps.",
            "recommendations": "A single sentence recommending whether they should see a doctor."
        }}
        """
        
        try:
            # Generate the response
            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Keep it deterministic and factual
                    candidate_count=1,
                    # We could use response_mime_type="application/json" if fully supported, 
                    # but fallback parsing is safer for raw JSON output.
                ),
            )
            
            # Observe latency
            duration = time.time() - start_time
            LLM_LATENCY.observe(duration)
            
            raw_text = response.text.strip()
            
            # Clean up potential markdown blocks if the LLM ignores instructions
            if raw_text.startswith("```json"):
                 raw_text = raw_text.replace("```json", "", 1)
            if raw_text.startswith("```"):
                 raw_text = raw_text.replace("```", "", 1)
            raw_text = raw_text.strip("` \n")
            
            # Parse and validate structure
            data = json.loads(raw_text)
            
            # Ensure required keys exist to prevent frontend crashes
            required_keys = ["description", "causes", "care", "recommendations"]
            for key in required_keys:
                if key not in data:
                    logger.warning(f"LLM response missing required key '{key}'. Raw logic: {raw_text}")
                    return None
                    
            return data
            
        except json.JSONDecodeError as e:
             logger.error(f"Failed to parse LLM JSON output. Error: {e}, Raw: {response.text}")
             return None
        except Exception as e:
             logger.error(f"LLM Generation failed: {e}")
             return None

# Singleton
llm_service = LLMService()
