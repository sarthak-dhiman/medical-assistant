# syntax=docker/dockerfile:1

# Build the React app first so the runtime image can serve static files with Nginx.
FROM node:22-alpine AS frontend-builder

WORKDIR /frontend

COPY web_app/frontend/package.json web_app/frontend/package-lock.json ./
RUN npm ci --legacy-peer-deps

COPY web_app/frontend ./
RUN npm run build


# Render runs one Docker web service container. This runtime image starts the
# local broker, Celery workers, FastAPI API, and Nginx static frontend.
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=10000 \
    REDIS_URL=redis://127.0.0.1:6379/0 \
    ALLOW_ALL_ORIGINS=True \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    nginx \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY web_app/backend/requirements_web.txt /tmp/requirements_web.txt
RUN pip install --no-cache-dir --retries 10 --timeout 1000 -r /tmp/requirements_web.txt

# Some GPU ONNXRuntime builds look up libnvrtc.so without the version suffix.
RUN if [ -e /opt/conda/lib/libnvrtc.so.11.2 ]; then \
      ln -sf /opt/conda/lib/libnvrtc.so.11.2 /opt/conda/lib/libnvrtc.so; \
    fi

COPY segformer_utils.py inference_pytorch.py inference_new_models.py vis_utils.py posture_rules.py ./
COPY saved_models ./saved_models
RUN mkdir -p saved_models_onnx
COPY web_app/backend ./web_app/backend

COPY --from=frontend-builder /frontend/dist /usr/share/nginx/html
COPY render-start.sh /usr/local/bin/render-start
RUN chmod +x /usr/local/bin/render-start

EXPOSE 10000

CMD ["render-start"]
