# Base Image
FROM python:3.10-slim

# System Dependencies (OpenCV + Glib)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python Deps
COPY web_app/backend/requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt

# Force Cache Bust to ensure updated inference scripts are copied
ENV CACHEBUST=1


# Copy Root-Level Inference Scripts (Crucial for importing segformer_utils, etc.)
COPY segformer_utils.py .
COPY inference.py .
COPY inference_skin.py .
COPY inference_sclera.py .
COPY jaundice_combined.py .
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
