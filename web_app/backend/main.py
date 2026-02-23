from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import base64
import logging
import sys
from .celery_app import celery_app
from .tasks import predict_task
from .config import settings

import datetime

# --- IST Logging Configuration ---
def ist_converter(*args):
    # IST = UTC + 5:30
    return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5, minutes=30)).timetuple()

logging.Formatter.converter = ist_converter

logging.basicConfig(
    stream=sys.stdout,
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Assistant API")

# --- CORS ---
origins = ["*"] if settings.ALLOW_ALL_ORIGINS else settings.ALLOWED_ORIGINS
logger.info(f"CORS Allowed Origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    image: str # Base64 encoded image
    mode: str  # JAUNDICE_BODY, JAUNDICE_EYE, SKIN_DISEASE
    debug: bool = False # Enable Graduate-CAM and extended stats

    @validator('image')
    def validate_image(cls, v):
        if not v:
            raise ValueError('Image data is empty')
        
        # Check size (rough estimate: len(base64) * 0.75 = bytes w/o headers)
        # Header is usually "data:image/jpeg;base64," (~23 chars)
        if len(v) > (settings.MAX_IMAGE_SIZE_BYTES * 1.37): # approx base64 size limit
             raise ValueError(f'Image size exceeds limit of {settings.MAX_IMAGE_SIZE_BYTES/1024/1024}MB')
        return v

@app.get("/")
def read_root():
    return {"status": "online", "system": "Medical Assistant Backend"}

@app.get("/health")
async def health_check():
    """
    Check if AI Worker is ready and models are loaded.
    """
    from fastapi import Response, status
    from .tasks import check_model_health
    try:
        # Wait up to 12 seconds for worker response (Docker timeout is 15s)
        task = check_model_health.delay()
        result = task.get(timeout=12.0)
        
        if result.get("ready"):
            return {"status": "ready", "models_ready": True, "details": result}
        else:
            # Return 503 Service Unavailable during loading
            return Response(
                content='{"status": "loading", "models_ready": False}',
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                media_type="application/json"
            )
    except Exception as e:
        logger.warning(f"Health check warmup (pending): {e}")
        return Response(
            content=f'{{"status": "not_ready", "models_ready": False, "error": "{str(e)}"}}',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json"
        )

@app.post("/predict")
async def predict_endpoint(request: PredictRequest):
    """
    Enqueue an image for prediction.
    Returns a Task ID.
    """
    # Enqueue task to appropriate queue
    try:
        # Determine queue based on mode
        heavy_modes = ["JAUNDICE_BODY", "JAUNDICE_EYE", "SKIN_DISEASE", "BURNS"]
        queue = "q_heavy_cv" if request.mode in heavy_modes else "q_lightweight"
        
        task = predict_task.apply_async(
            args=[request.image, request.mode, request.debug],
            queue=queue
        )
        logger.info(f"Task enqueued to {queue}: {task.id} (Mode: {request.mode})")
        return {"task_id": task.id, "status": "processing"}
    except Exception as e:
        logger.error(f"Failed to enqueue task: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/predict/auto")
async def predict_auto_endpoint(request: PredictRequest):
    """
    Enqueue an image for automatic diagnostic routing.
    Ignores 'mode' in request (automatically determined).
    """
    from .tasks import diagnostic_gateway_task
    try:
        # Auto-gateway uses face-mesh/analysis, route to heavy queue
        task = diagnostic_gateway_task.apply_async(
            args=[request.image, request.debug],
            queue="q_heavy_cv"
        )
        logger.info(f"Auto-Diagnostic Task enqueued to q_heavy_cv: {task.id}")
        return {"task_id": task.id, "status": "processing", "mode": "AUTO"}
    except Exception as e:
        logger.error(f"Failed to enqueue auto-task: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """
    Poll for result.
    """
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        return {"state": "PENDING", "result": None}
    elif task.state != 'FAILURE':
        return {"state": task.state, "result": task.result}
    else:
        return {"state": "FAILURE", "error": str(task.info)}
