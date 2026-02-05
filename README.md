# Disease Prediction System

## Overview
This is a comprehensive multi-modal disease prediction system capable of detecting:
*   **Skin Diseases**: 38 different categories of skin conditions.
*   **Jaundice**: Using both skin tone analysis and sclera (eye) segmentation.

The system features a **FastAPI** backend for high-performance inference and a **React** frontend for a premium user experience.

---

## 🧠 AI Models (Unified PyTorch Architecture)

All models now use **EfficientNet-B4** via PyTorch for maximum accuracy and unified GPU memory management.

### Skin Disease Detection Model
*   **Architecture**: EfficientNet-B4 (PyTorch)
*   **Training Hardware**: Google Colab TPU v5e-1 (High Performance)
*   **Input**: 380x380 RGB Images (High Res)
*   **Classes**: 38 Categories
*   **Status**: Migrated from Keras B0 to PyTorch B4 for superior accuracy.

### Jaundice Body Detection
*   **Architecture**: EfficientNet-B4 (PyTorch)
*   **Training Hardware**: Local RTX 3050 GPU
*   **Input**: 380x380 RGB Images
*   **Focus**: Analyzes skin tone for bilirubin-induced yellowing.
*   **Status**: Migrated to PyTorch to share VRAM with Eye model.

### Jaundice Eye Detection
*   **Architecture**: EfficientNet-B4 (PyTorch) / SegFormer
*   **Approach**: Sclera segmentation + Color Analysis
*   **Status**: Stable and High Performance.

---

## 🛠️ Project Configuration

### Tech Stack
*   **Backend**: Python, FastAPI, Uvicorn
*   **Machine Learning**: TensorFlow/Keras, PyTorch, Transformers (HuggingFace), OpenCV
*   **Frontend**: React, Vite, TailwindCSS
*   **DB/Storage**: Local filesystem for datasets and models.

### Directory Structure
*   `app/`: Main FastAPI backend application.
*   `client/`: React frontend application.
*   `Dataset/`: Contains training data (Ignored in Git).
    *   `skin/`: Organized 38-class dataset.
*   `dataset_jaundice_eyes/`: Training data for eye segmentation.
*   `saved_models/`: Stores `.keras` and `.pt` model files.

### Initial Setup Flow

**Prerequisites**
*   **Python**: Ensure you have Python 3.10 installed.
*   **Git**: For cloning the repository.
*   **CUDA (Optional)**: For GPU acceleration (recommended for faster inference).

**Installation**

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create & Activate Virtual Environment**:
    ```bash
    # Windows
    python -m venv venv_310
    .\venv_310\Scripts\Activate.ps1
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

**Model Acquisition (Auto-Setup)**
*   The system uses a **SegFormer** model for eye segmentation (`jonathandinu/face-parsing`).
*   **You do NOT need to manually download this.**
*   The first time you run the application (or `segformer_utils.py`), the script will **automatically download** the necessary model files from HuggingFace and save them to `saved_models/segformer/`.
*   *Note: Ensure you have an internet connection for this first run.*

**Running the Application**

**Backend Server (Dev Mode)**:
```bash
uvicorn app.main:app --reload
```

**Frontend (React)**:
```bash
cd client
npm install  # First time only
npm run dev
```

**Training**
*   **Skin Model**: `python train_skin_model.py`
*   **Jaundice Model**: `python train_jaundice_pytorch.py`

### Environment
*   Configured to prefer **PyTorch** backend for Keras `os.environ["KERAS_BACKEND"] = "torch"`.
*   GPU Acceleration enabled (NVIDIA CUDA). 

### Standalone App
*   `build_app.py`: Creates a standalone executable using PyInstaller.  
    *   `dist/MedicalAssistant/MedicalAssistant.exe`
    *   `venv_310`: Virtual environment for dependencies.
    *   `saved_models`: Model files.
    *   `Dataset`: Training data (Ignored in Git).
    *   `dataset_jaundice_eyes`: Training data for eye segmentation.
*   The goal of the standalone app is to provide a simple way to run the app without requiring the user to install Python and other dependencies. It is also a way to distribute the app to users who may not have Python installed. basically it packs the whole project into a single zipped format allowing for just a quick download standing between the user and his diagnosis. 

*   The app even though not perfect is a MVP (Minimum Viable Product) and is a work in progress. I plan to add more features and improve the app in the future , like more datasets in the future to increas the number of diseases that can be detected by the application. though its verdict should not be taken as the final verdict and the app should be used as a tool to help the user understand the possible diseases that can be detected by the application and not as a  medical diagnosis, for those please visit the doctors near you.

### 🚀 Async Architecture & Docker (Redis + Celery)
For production-grade performance, the system uses **Redis** and **Celery** to handle heavy AI inference tasks asynchronously.

*   **Redis**: Acts as the message broker and result backend.
*   **Celery**: Worker process that picks up inference tasks from the queue and processes them in the background, preventing the main API from blocking.

**Running with Docker (Recommended for Full Stack):**
The project includes a `docker-compose.yml` that orchestrates the entire stack (Frontend, Backend, Redis, Worker).

```bash
docker-compose up --build
```
*   **Frontend**: `http://localhost:5173`
*   **Backend**: `http://localhost:8000`
*   **Worker**: Runs in background (Monitor logs: `docker-compose logs -f worker`)

**Running Manually (Dev Mode):**
If you want to run the async stack manually:
1.  **Start Redis**: Ensure Redis is running on port 6379.
2.  **Start Worker**: 
    ```bash
    celery -A web_app.backend.celery_app worker --loglevel=info
    ```
3.  **Start Backend**:
    ```bash
    uvicorn app.main:app --reload
    ```
### Usage
*   `MedicalAssistant.exe`: Run the standalone app.
*   `webcam_app.py`: Run the app from source code.
*   `build_app.py`: Rebuild the app.

### Disclaimer

*   This project is a work in progress and is not intended to be used as a medical diagnosis tool. It is a proof of concept and should not be used as a medical diagnosis tool. For medical diagnosis, please visit the doctors near you.

*   This app is still being actively developed and is undergoing changes on a regular basis. 
