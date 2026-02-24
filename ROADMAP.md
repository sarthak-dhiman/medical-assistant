# Project Roadmap / Future Improvements

## Phase 1: Performance & Reliability
- [x] ONNX Conversion: Convert PyTorch .pth models to ONNX to reduce memory and increase speed. (Completed for 7 classifiers + SegFormer)
- [x] Clinical Confidence Calibration: Implement temperature scaling for realistic confidence scores.

## Phase 1.5: Observability & Robustness
- [x] Prometheus Metrics: Expose `/metrics` on FastAPI using `prometheus-fastapi-instrumentator`.
- [x] Latency Tracking: Wrap calls in `inference_service.py` to record per-model inference latencies.
- [x] Celery Exporter: Run `celery-prometheus-exporter` to track background tasks queue depth.
- [x] Grafana Dashboards: Add a Grafana container with pre-built dashboards monitoring API health, model latency, and worker load.
- [x] Tiered API Routing (Asynchronous Queues): Configure multiple Celery queues (`q_lightweight`, `q_heavy_cv`) to prevent fast tasks (Posture) from lagging behind heavy diagnostic tasks (Derm1M). Route backend endpoints accordingly.

## Phase 2: Edge Computing & Data Privacy
- [x] Client-Side Pre-Processing (Edge AI Shift): Integrate `MediaPipe for Web` or `ONNX.js` into the React frontend. Run face mesh locally to crop eyes/mouth off user's webcam, so only a tiny `224x224` tensor is transmitted over the network (massively reducing Redis/Bandwidth chokes).
- [ ] Zero-Trace Medical Anonymization (HIPAA/GDPR): Build a post-inference Celery pipeline that automatically detects and permanently blitzes/blurs faces in the background of skin/chest/arm diagnostic photos before any theoretical storage to an S3 bucket (Data Flywheel).

## Phase 3: The "Assistant" Experience & Clinical Calibration
- [x] Neonatal & Lighting Calibration Pipeline: Introduce Color Constancy algorithms (Gray World / Shades of Gray) to mathematically strip lighting temperatures before inference. Add an `Infant / Adult` demographic toggle to route accurately for Jaundice vs normal adult Skin models.
- [ ] Decision Layer (Clinical Triage Engine): Rule-based system for cross-model reasoning and high-priority referral handling.
- [ ] LLM Medical Explanation Layer: Heavily constrained medical LLM for plain text explanations.

## Phase 4: Advanced AI Capabilities
- [ ] Multi-Modal AI Fusion: Add patient history questionnaire.
- [ ] Dataset Bias Auditing: Skin tone fairness analysis.
- [ ] Model Uncertainty Estimation: Monte Carlo dropout and ensembles.
- [ ] Active Learning Loop: Route uncertain samples for human review.

## Phase 5: Synthetic Logic Pipeline for Postural Conditions (Explainable, No-Training Approach)

- Summary: When large, labelled clinical datasets are unavailable (or commercially restricted), implement a deterministic inference-layer pipeline that converts high-quality pose keypoints into clinically meaningful angles and rule-based classifications. This is an engineering-first alternative to training an end-to-end "Kyphosis" classifier and provides fully explainable metrics for clinicians and "Nerd Mode" dashboards.

- Feasibility: Highly viable. Off-the-shelf pose estimators (MMPose, YOLOv8-Pose or MediaPipe Pose) reliably output joint (x,y) coordinates for standard landmarks. Simple trigonometry on these coordinates yields reproducible angles (e.g., head-to-C7 angle, thoracic curvature proxies). Because this approach uses deterministic math and established thresholds, it avoids the need for large labelled datasets, reduces medical licensing concerns, and produces explainable outputs that can be audited clinically.

- Implementation steps:
	- Integrate a robust pose keypoint detector into the inference worker (prefer MMPose / YOLOv8-Pose running in the heavy Celery worker or a GPU-capable service). Keep the detector output standardized (normalized x,y relative to image or absolute pixels + image size).
	- Build a lightweight geometry module `posture_rules.py` that:
		- Accepts keypoints (ear, eye, nose, C7, shoulders, mid-hip) and image dims.
		- Computes vectors and angles with numerically stable math (e.g., arctan2, dot-product, acos for angle between vectors).
		- Implements clinically-sourced thresholds (configurable) to classify: Forward Head Posture, Thoracic Kyphosis Proxy, Pelvic Tilt, etc.
		- Outputs: {condition, severity, angle_values, supporting_keypoints} for display and audit.
	- Expose endpoint/worker result fields for `nerd_mode` showing raw angles, keypoint overlays (SVG or normalized bbox), and the exact rule triggered.
	- Add unit tests with synthetic keypoint examples and a small validation set (10–200 expert-labelled cases) to sanity-check threshold choices.

- Pros:
	- No costly medical dataset required.
	- Explainable and auditable decisions — each classification links to an exact measured angle.
	- Fast, deterministic, and easy to iterate (adjust thresholds without retraining).

- Cons / Limitations:
	- Pose detectors can fail under occlusion, camera angle, or non-standard clothing — include confidence gating and fallbacks.
	- Clinical thresholds vary; integrate a validation step with clinicians before production use.
	- Not a substitute for full clinical imaging (X-ray) where exact vertebral measures are needed.

- Validation & rollout:
	- Start in `nerd_mode` only, surface angles and rule details for clinician review.
	- Collect flagged cases and optionally build a small supervised dataset later if desired (hybrid approach).
	- Provide a configurable dashboard of false-positive / false-negative rates and a simple threshold tuning UI.

This phase provides a fast path to actionable postural screening while preserving explainability and minimizing data/regulatory risk.
