import os

class Config:
    # --- Infrastructure ---
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # --- Security ---
    # Default to localhost for dev, allow override via env
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    # If explicitly set to "*" via env, allow all
    ALLOW_ALL_ORIGINS = os.getenv("ALLOW_ALL_ORIGINS", "False").lower() == "true"
    
    # --- Models ---
    # Centralized paths
    MODEL_DIR = os.getenv("MODEL_DIR", "saved_models")
    JAUNDICE_MODEL_PATH = os.path.join(MODEL_DIR, "jaundice_model.pth")
    # ... add others as needed
    
    # --- Logging ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # --- Performance ---
    # Max image size (e.g. 10MB)
    MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
    
settings = Config()
