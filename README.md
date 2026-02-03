# Disease Prediction System

## Overview
This is a comprehensive multi-modal disease prediction system capable of detecting:
1.  **Skin Diseases**: 38 different categories of skin conditions.
2.  **Jaundice**: Using both skin tone analysis and sclera (eye) segmentation.

The system features a **FastAPI** backend for high-performance inference and a **React** frontend for a premium user experience.

---

## 🧠 AI Models

### 1. Skin Disease Detection Model
*   **Architecture**: EfficientNetB0 (Transfer Learning)
*   **Backend**: Keras (TensorFlow/PyTorch)
*   **Input**: 224x224 RGB Images
*   **Training Status**: Trained on a massive merged dataset of ~32,000 images.
*   **Classes (38 Types)**:
    1)  Acne
    2)  Actinic Keratosis
    3)  Atopic Dermatitis
    4)  Benign Tumors
    5)  Bullous
    6)  Candidiasis
    7)  Cellulitis Impetigo
    8)  Contact Dermatitis
    9)  Drug Eruption
    10) Dry Skin
    11) Eczema
    12) Herpes HPV STD
    13) Infestations Bites
    14) Lichen
    15) Lupus
    16) Moles
    17) Monkeypox
    18) Normal
    19) Oily Skin
    20) Perioral Dermatitis
    21) Pigment Disorders
    22) Psoriasis
    23) Rosacea
    24) Scabies
    25) Sebaceous Glands
    26) Seborrheic Keratoses
    27) Seborrheic Dermatitis
    28) Skin Cancer
    29) Sun Sunlight Damage
    30) Systemic Disease
    31) TineaS
    32) Tinea Fungal
    33) Unknown Normal
    34) Urticaria Hives
    35) Vascular Tumors
    36) Vasculitis
    37) Vitiligo
    38) Warts

### 2. Jaundice Detection (Combined)
*   **Approach**: Multi-Input Model
*   **Input A (Skin)**: Facial skin analysis for yellow discoloration.
*   **Input B (Sclera)**: **SegFormer** model extracts the sclera (white part of the eye) to analyze pixel color values (yellow vs white ratio).
*   **Classes**:
    *   Jaundice
    *   Normal

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

**1. Prerequisites**
*   **Python**: Ensure you have Python 3.10 installed.
*   **Git**: For cloning the repository.
*   **CUDA (Optional)**: For GPU acceleration (recommended for faster inference).

**2. Installation**

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

**3. Model Acquisition (Auto-Setup)**
*   The system uses a **SegFormer** model for eye segmentation (`jonathandinu/face-parsing`).
*   **You do NOT need to manually download this.**
*   The first time you run the application (or `segformer_utils.py`), the script will **automatically download** the necessary model files from HuggingFace and save them to `saved_models/segformer/`.
*   *Note: Ensure you have an internet connection for this first run.*

**4. Running the Application**

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

**5. Training**
*   **Skin Model**: `python train_skin_model.py`
*   **Jaundice Model**: `python train_jaundice_with_sclera.py`

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
