# AI-Powered Jaundice & Skin Disease Assistant 🏥

A containerized, mobile-first web application for the early detection of **Neonatal Jaundice** and **dermatological conditions** using advanced computer vision and deep learning.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Python](https://img.shields.io/badge/Python-3.10-yellow)
![React](https://img.shields.io/badge/React-Vite-cyan)

---

## 🚀 Key Features

### 1. Jaundice Eye Model (Flagship) 👁️
-   **Method:** Uses **Google MediaPipe Face Mesh** (468 landmarks) to precisely locate and mask the **sclera** (white of the eye).
-   **Innovation:** Replaced unreliable Hough Circles with Face Mesh for sub-pixel accuracy.
-   **Model:** EfficientNet-B4 trained on segmented sclera crops.
-   **Output:** Jaundice / Normal (with confidence score).

### 2. Jaundice Body Model 👶
-   **Method:** Uses **MediaPipe Selfie Segmentation** to isolate the baby from the background (bedsheets, walls), followed by skin color filtering.
-   **Benefit:** Prevents background noise (yellow walls/blankets) from triggering false positives.
-   **Model:** EfficientNet-B4 trained on skin patches.

### 3. Skin Disease Model 🩹
-   **Capability:** Detects 38+ skin conditions (Acne, Eczema, Melanoma, etc.).
-   **Method:** Hybrid Segmentation.
    -   **Primary:** Semantic Segmentation (SegFormer) for face/neck.
    -   **Fallback:** Robust **HSV+YCbCr Color Detection** for arms, legs, and other body parts where semantic models fail.

### 4. Technical Highlights
-   **"Nerd Mode"**: Real-time debug overlay showing segmentation masks, bounding boxes, and model confidence.
-   **Asynchronous Inference**: Celery + Redis architecture eliminates API timeouts for heavy AI tasks.
-   **Hot-Reloading**: Backend code changes reflect instantly in Docker containers.

---

## 🛠️ Tech Stack

-   **Frontend**: React, Vite, TailwindCSS (Dark Mode / Glassmorphism UI).
-   **Backend**: FastAPI, Uvicorn, Python 3.10.
-   **AI Engine**: PyTorch, TensorFlow, Google MediaPipe, OpenCV.
-   **Worker/Queue**: Celery, Redis.
-   **Infrastructure**: Docker Compose (Microservices Architecture).
-   **Reverse Proxy**: Nginx.

---

## ⚡ Quick Start

### Prerequisites
-   Docker Desktop installed and running.
-   NVIDIA GPU (Recommended) with Drivers installed.

### Installation

1.  **Clone the Repository**
    ```bash
    git clone <repo-url>
    cd "Disease Prediction"
    ```

2.  **Start the Application**
    ```bash
    docker compose up --build
    ```
    *Note: The first build may take a few minutes to install dependencies (PyTorch, MediaPipe).*

3.  **Access the App**
    -   **Frontend**: [http://localhost:5173](http://localhost:5173)
    -   **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
    -   **Flower (Task Monitor)**: [http://localhost:5555](http://localhost:5555) (if enabled)

---

## 🐛 Troubleshooting

| Issue | Solution |
| :--- | :--- |
| `AttributeError: no attribute 'solutions'` | MediaPipe Dependency fixed. Run `docker compose up --build`. |
| "No Skin" Result | Ensure good lighting. The model falls back to color detection if face is not found. |
| API Timeout | The first request might be slow as models "warm up". Subsequent requests are fast. |

---

## 📂 Project Structure

```
├── web_app
│   ├── backend
│   │   ├── main.py           # FastAPI Entrypoint
│   │   ├── tasks.py          # Celery Tasks (AI Inference)
│   │   ├── Dockerfile        # Backend Environment
│   ├── frontend              # React App
├── segformer_utils.py        # MediaPipe & Segmentation Logic
├── inference_pytorch.py      # PyTorch Model Definitions
├── docker-compose.yml        # Orchestration Config
```

---

**Developed by:** Sarthak Dhiman
