# AI-Powered Jaundice & Skin Disease Assistant

A containerized, mobile-first web application for the early detection of **Neonatal Jaundice**, **dermatological conditions**, **Burns**, **Hairloss**, and **Nail Diseases** using advanced computer vision and deep learning. (Total: **6 Detection Modes**)

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Python](https://img.shields.io/badge/Python-3.10-yellow)
![React](https://img.shields.io/badge/React-Vite-cyan)

---

## Technical Architecture

The system is built as a microservices architecture to ensure scalability and maintainability:

-   **Frontend:** React (Vite) + TailwindCSS. Configured as a Progressive Web App (PWA) with mobile-first design principles.
-   **Backend:** FastAPI (Python 3.10). Handles high-concurrency requests via Uvicorn.
-   **AI Inference:** Deployed on a Celery Worker layout with Redis as the message broker. This asynchronous architecture prevents API timeouts during heavy model inference.
-   **Infrastructure:** Fully Dockerized with Nginx as a reverse proxy/load balancer.

---

## AI Models & Methodology

### 1. Jaundice Eye Model (Adults)
Target: **Jaundice Adult**
-   **Methodology:** Utilizes **Google MediaPipe Face Mesh** to extract 468 facial landmarks.
-   **Process:**
    1.  Detects face and eye landmarks in real-time.
    2.  Extracts the eye region and generates a semantic mask to isolate the **Sclera** (white region) from the Iris and Eyelids.
    3.  Feeds the masked sclera crop into an **EfficientNet-B4** classifier.
-   **Advantage:** Eliminates noise from skin tone or iris color, providing a pure analysis of scleral yellowing.

### 2. Jaundice Body Model (Neonates)
Target: **Jaundice Baby**
-   **Methodology:** Combines **MediaPipe Selfie Segmentation** with Color Space Analysis.
-   **Process:**
    1.  Generates a binary mask of the human subject (removing background walls/bedsheets).
    2.  Filters skin pixels using HSV/YCbCr thresholds within the subject mask.
    3.  analyzes the skin tone for hyperbilirubinemia indicators using a custom trained Deep Learning model.
-   **Advantage:** Prevents false positives caused by yellow ambient environments (e.g., painted walls or blankets).

### 3. Skin Disease Classifer
Target: **General Dermatology**
-   **Capability:** Classifies **38 distinct skin conditions** with high granularity.
-   **Methodology:** Uses a Hybrid Segmentation approach.
    -   **Primary:** SegFormer (Semantic Segmentation) for face/neck analysis.
    -   **Fallback:** Adaptive Color Detection for limbs and body parts where facial models fail.
    
#### Recognized Skin Classes:
The model is trained to identify the following 38 conditions:
1.  Acne
2.  Actinic Keratosis
3.  Atopic Dermatitis
4.  Benign Tumors
5.  Bullous Diseases
6.  Candidiasis
7.  Cellulitis & Impetigo
8.  Contact Dermatitis
9.  Drug Eruption
10. Dry Skin
11. Eczema
12. Herpes, HPV & STD
13. Infestations & Bites
14. Lichen
15. Lupus
16. Moles
17. Monkeypox
18. Normal
19. Oily Skin
20. Perioral Dermatitis
21. Pigment Disorders
22. Psoriasis
23. Rosacea
24. Scabies
25. Sebaceous Glands
26. Seborrheic Keratoses
27. Seborrheic Dermatitis
28. Skin Cancer
29. Sun & Sunlight Damage
30. Systemic Disease
31. Tinea (Fungal)
32. Tinea (General)
33. Unknown / Normal
34. Urticaria & Hives
35. Vascular Tumors
36. Vasculitis
37. Vitiligo
38. Warts

---

### 4. Burns Detection Model
Target: **Acute Burn Injuries**
-   **Methodology:** Binary classification using **EfficientNet-B4**.
-   **Process:**
    1.  Segments skin area using HSV color thresholding.
    2.  Extracts Region of Interest (ROI) containing the potential burn.
    3.  Classifies the ROI as **Burn** or **Healthy Skin**.
-   **Capability:** Distinguishes between healthy skin and burn injuries with high accuracy.

### 5. Hairloss Analysis Model
Target: **Alopecia & Hair Health**
-   **Methodology:** Full-frame analysis using **EfficientNet-B3**.
-   **Training:** Optimized with **Focal Loss** to handle class imbalance between bald and non-bald datasets.
-   **Capability:** Detects signs of significant hair loss and baldness patterns.

### 6. Nail Disease Model
Target: **Onychology**
-   **Methodology:** Multi-class classification using **EfficientNet-B4**.
-   **Classes:** Detects 8 specific conditions:
    1.  Acral Lentiginous Melanoma
    2.  Blue Finger
    3.  Clubbing
    4.  Healthy Nail
    5.  Nail Psoriasis
    6.  Onychogryphosis
    7.  Onychomycosis
    8.  Pitting

---

## Installation & Deployment

### Prerequisites
-   Docker Desktop
-   NVIDIA GPU (Recommended for Inference acceleration)

### Quick Start

1.  **Clone the Repository**
    ```bash
    git clone <repo-url>
    cd "Disease Prediction"
    ```

2.  **Launch via Docker Compose**
    ```bash
    docker compose up --build
    ```

3.  **Access the Application**
    -   Frontend: `http://localhost:5173`
    -   API Documentation: `http://localhost:8000/docs`

---

## Desktop App (Windows)

You can run the application as a standalone Windows executable (`.exe`) without needing Docker or Python installed on the target machine.

### Prerequisites (for building)
-   Python 3.10+
-   Node.js (for building frontend) or Docker (to build frontend via container)

### How to Build & Run locally

1.  **Install Python Dependencies**:
    ```bash
    pip install pywebview onnxruntime opencv-python-headless pyinstaller numpy
    ```

2.  **Build the Executable**:
    Run the included batch script:
    ```bash
    build_desktop.bat
    ```
    *This script automatically:*
    -   Builds the React Frontend.
    -   Exports PyTorch models to ONNX.
    -   Packages everything into `dist/MedicalAssistant.exe`.

3.  **Run the App**:
    Navigate to the `dist` folder and double-click `MedicalAssistant.exe`.

---

## Development Notes

### Hot-Reloading
The `docker-compose.yml` is configured to mount the source code directories directly. Changes made to `web_app/backend`, `segformer_utils.py`, or `inference_pytorch.py` will be reflected immediately in the container without requiring a rebuild (restart may be required for worker processes).

### Troubleshooting
-   **"No Skin Detected"**: Ensure the subject is well-lit. The fallback color detector requires decent lighting to distinguish skin from background.
-   **GPU Usage**: By default, the `docker-compose.yml` requests NVIDIA GPU access. If running on CPU, remove the `deploy` and `environment` sections related to NVIDIA.
