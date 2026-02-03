import os
from pathlib import Path

# Base Paths
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT / "saved_models"

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Models
JAUNDICE_MODEL_PATH = MODEL_DIR / "jaundice_model.keras"
SKIN_MODEL_PATH = MODEL_DIR / "_skin_model.keras"
SKIN_MAPPING_PATH = MODEL_DIR / "new_class_indices.json"
SEGFORMER_PATH = MODEL_DIR / "segformer"

# App
API_HOST = "0.0.0.0"
API_PORT = 8000
