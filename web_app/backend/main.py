from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from .celery_app import celery_app
from .tasks import predict_task

app = FastAPI(title="Medical Assistant API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev, limit in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    image: str # Base64 encoded image
    mode: str  # JAUNDICE_BODY, JAUNDICE_EYE, SKIN_DISEASE

@app.get("/")
def read_root():
    return {"status": "online", "system": "Medical Assistant Backend"}

@app.post("/predict")
async def predict_endpoint(request: PredictRequest):
    """
    Enqueue an image for prediction.
    Returns a Task ID.
    """
    if not request.image:
        raise HTTPException(status_code=400, detail="No image provided")
        
    # Enqueue task
    task = predict_task.delay(request.image, request.mode)
    
    return {"task_id": task.id, "status": "processing"}

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
