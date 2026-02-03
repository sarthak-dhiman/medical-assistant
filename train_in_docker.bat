@echo off
echo Starting Jaundice Training in Docker (GPU Enabled)...
docker run --gpus all --rm -v "%CD%":/app jaundice-trainer
pause
