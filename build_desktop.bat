@echo off
echo ==========================================
echo   Building Medical AI Desktop App (ONNX)
echo ==========================================

REM 1. Check Python
python --version
if %errorlevel% neq 0 (
    echo Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

REM 2. Install Dependencies
echo.
echo --- Installing Dependencies ---
pip install pywebview onnxruntime opencv-python-headless pyinstaller numpy

REM 3. Clean Previous Builds
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"

REM 4. Verify Frontend Build
if not exist "web_app\frontend\dist\index.html" (
    echo Error: Frontend build not found!
    echo Please run: docker run --rm -v "%%cd%%\web_app\frontend:/app" node:22-alpine sh -c "cd /app && npm install && npm run build"
    pause
    exit /b 1
)

REM 5. Verify Models
if not exist "saved_models\onnx\jaundice_body.onnx" (
    echo Error: ONNX models not found!
    echo Please run the export script first.
    pause
    exit /b 1
)
if not exist "saved_models\onnx\burns.onnx" (
    echo Error: burns.onnx not found!
    pause
    exit /b 1
)
if not exist "saved_models\onnx\nail_disease.onnx" (
    echo Error: nail_disease.onnx not found!
    pause
    exit /b 1
)
if not exist "saved_models\nail_disease_mapping.json" (
    echo Error: nail_disease_mapping.json not found!
    pause
    exit /b 1
)

REM 6. Run PyInstaller
echo.
echo --- Packaging Executable (Excluding heavy unused modules) ---
python -m PyInstaller --noconfirm --onefile --windowed ^
    --name "MedicalAssistant" ^
    --paths "desktop" ^
    --add-data "web_app/frontend/dist;dist" ^
    --add-data "saved_models/onnx;models" ^
    --add-data "saved_models/skin_disease_mapping.json;models" ^
    --add-data "saved_models/nail_disease_mapping.json;models" ^
    --hidden-import "inference" ^
    --hidden-import "onnxruntime" ^
    --hidden-import "cv2" ^
    --hidden-import "numpy" ^
    --exclude-module "tensorflow" ^
    --exclude-module "torch" ^
    --exclude-module "tensorboard" ^
    --exclude-module "keras" ^
    --exclude-module "matplotlib" ^
    --exclude-module "ipython" ^
    --exclude-module "notebook" ^
    desktop/main.py

echo.
if %errorlevel% equ 0 (
    echo ==========================================
    echo   BUILD SUCCESS!
    echo   Executable: dist\MedicalAssistant.exe
    echo ==========================================
) else (
    echo   BUILD FAILED!
)
pause
