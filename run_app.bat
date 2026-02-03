@echo off
TITLE Medical Assistant Launcher
echo ==================================================
echo   Starting Medical Assistant (Local Mode)
echo ==================================================

echo [1/3] Starting Backend API...
start cmd /k "cd /d d:\Disease Prediction && uvicorn web_app.backend.main:app --reload"

echo [2/3] Starting AI Worker...
echo NOTE: Ensure Redis is running on localhost:6379!
start cmd /k "cd /d d:\Disease Prediction && celery -A web_app.backend.celery_app worker --pool=solo --loglevel=info"

echo [3/3] Starting Frontend...
start cmd /k "cd /d d:\Disease Prediction\web_app\frontend && npm run dev"

echo.
echo ==================================================
echo   All components launched in new windows.
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8000
echo ==================================================
echo.
pause
