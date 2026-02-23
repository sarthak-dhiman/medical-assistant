# Project Roadmap / Future Improvements

## Phase 1: Performance & Reliability
- [x] ONNX Conversion: Convert PyTorch .pth models to ONNX to reduce memory and increase speed. (Completed for 7 classifiers + SegFormer)
- [ ] Clinical Confidence Calibration: Implement temperature scaling for realistic confidence scores.

## Phase 1.5: Observability & Robustness
- [ ] Prometheus Metrics: Expose `/metrics` on FastAPI using `prometheus-fastapi-instrumentator`.
- [ ] Latency Tracking: Wrap calls in `inference_service.py` to record per-model inference latencies.
- [ ] Celery Exporter: Run `celery-prometheus-exporter` to track background tasks queue depth.
- [ ] Grafana Dashboards: Add a Grafana container with pre-built dashboards monitoring API health, model latency, and worker load.
- [ ] Tiered API Routing (Asynchronous Queues): Configure multiple Celery queues (`q_lightweight`, `q_heavy_cv`) to prevent fast tasks (Posture) from lagging behind heavy diagnostic tasks (Derm1M). Route backend endpoints accordingly.

## Phase 2: Edge Computing & Data Privacy
- [ ] Client-Side Pre-Processing (Edge AI Shift): Integrate `MediaPipe for Web` or `ONNX.js` into the React frontend. Run face mesh locally to crop eyes/mouth off user's webcam, so only a tiny `224x224` tensor is transmitted over the network (massively reducing Redis/Bandwidth chokes).
- [ ] Zero-Trace Medical Anonymization (HIPAA/GDPR): Build a post-inference Celery pipeline that automatically detects and permanently blitzes/blurs faces in the background of skin/chest/arm diagnostic photos before any theoretical storage to an S3 bucket (Data Flywheel).

## Phase 3: The "Assistant" Experience & Clinical Calibration
- [ ] Neonatal & Lighting Calibration Pipeline: Introduce Color Constancy algorithms (Gray World / Shades of Gray) to mathematically strip lighting temperatures before inference. Add an `Infant / Adult` demographic toggle to route accurately for Jaundice vs normal adult Skin models.
- [ ] Decision Layer (Clinical Triage Engine): Rule-based system for cross-model reasoning and high-priority referral handling.
- [ ] LLM Medical Explanation Layer: Heavily constrained medical LLM for plain text explanations.

## Phase 4: Advanced AI Capabilities
- [ ] Multi-Modal AI Fusion: Add patient history questionnaire.
- [ ] Dataset Bias Auditing: Skin tone fairness analysis.
- [ ] Model Uncertainty Estimation: Monte Carlo dropout and ensembles.
- [ ] Active Learning Loop: Route uncertain samples for human review.
