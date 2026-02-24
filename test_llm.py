import sys
import os
sys.path.append('web_app/backend')
from dotenv import load_dotenv

load_dotenv()
os.environ["ENABLE_LLM_EXPLANATIONS"] = "True"

from llm_service import llm_service

print("Ready:", llm_service.is_ready)
if llm_service.is_ready:
    res = llm_service.generate_explanation("Melanoma", "SKIN_DISEASE")
    print(res)
