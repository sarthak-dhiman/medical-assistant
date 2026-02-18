# Base Image with PyTorch + CUDA 12.1
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# System Dependencies (OpenCV + Glib + Curl)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python Deps
COPY web_app/backend/requirements_web.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 1000 -r requirements_web.txt

# Force Cache Bust
ENV CACHEBUST=2

# Copy Root-Level Inference Scripts
# NOTE: inference.py and inference_skin.py were redundant and deleted
COPY segformer_utils.py .
COPY inference_pytorch.py .
COPY inference_new_models.py .
COPY vis_utils.py .
# COPY saved_models/ ./saved_models/ # We will mount this via Volume for speed

# Copy Backend Code
COPY web_app/backend ./web_app/backend

# Environment Variables
ENV PYTHONPATH=/app
ENV REDIS_URL=redis://redis:6379/0

# Expose API Port
EXPOSE 8000

# Default Command (Overridden by docker-compose)
CMD ["uvicorn", "web_app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
