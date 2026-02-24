"""
segformer_utils.py  —  ONNX Runtime Edition
=============================================
Replaces the HuggingFace/PyTorch SegFormer inference with a lightweight
onnxruntime session.

The ONNX model was exported via:  scripts/export_segformer_to_onnx.py

Input:  (1, 3, 512, 512)  float32  — ImageNet-normalised RGB
Output: (1, 19, 128, 128) float32  — raw logits (1/4 resolution)

Post-processing done in numpy/cv2:
  1. Argmax over class dim → (1, 128, 128) uint8
  2. Resize to original image size via cv2 (nearest-neighbour, preserves label IDs)
"""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import sys
import os
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
MIN_IMAGE_SIZE = 10
MAX_IMAGE_SIZE = 4096

# SegFormer standard input size (must match export)
SF_H, SF_W = 512, 512

# ImageNet normalisation (same as SegformerImageProcessor defaults)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

ONNX_PATH = BASE_DIR / "saved_models_onnx" / "segformer.onnx"

# ─────────────────────────────────────────────────────────────
# Session (thread-safe singleton)
# ─────────────────────────────────────────────────────────────
_session = None
_session_lock = threading.Lock()

def _get_session() -> ort.InferenceSession | None:
    global _session
    if _session: return _session
    with _session_lock:
        if _session: return _session
        if not ONNX_PATH.exists():
            logger.error(
                f"SegFormer ONNX model not found at {ONNX_PATH}. "
                "Run scripts/export_segformer_to_onnx.py to generate it."
            )
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        try:
            import torch
            _cuda_ok = torch.cuda.is_available()
        except Exception:
            _cuda_ok = False
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if _cuda_ok else ["CPUExecutionProvider"]
        try:
            _session = ort.InferenceSession(str(ONNX_PATH), sess_options=opts, providers=providers)
            logger.info(f"SegFormer ONNX session: {_session.get_providers()[0]}")
        except Exception as e:
            logger.error(f"Failed to create SegFormer ONNX session: {e}")
            _session = None
    return _session

# ─────────────────────────────────────────────────────────────
# Pre/Post processing helpers
# ─────────────────────────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR numpy → (1, 3, 512, 512) float32 NCHW tensor."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SF_W, SF_H))
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return img.transpose(2, 0, 1)[np.newaxis]  # NCHW

def _postprocess(logits: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """(1, 19, 128, 128) logits → (H, W) uint8 mask at original resolution."""
    seg = np.argmax(logits[0], axis=0).astype(np.uint8)   # (128, 128)
    # nearest-neighbour to preserve label integer values exactly
    seg = cv2.resize(seg, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return seg

# ─────────────────────────────────────────────────────────────
# SegFormerWrapper — same public API as the old PyTorch version
# ─────────────────────────────────────────────────────────────

class SegFormerWrapper:
    """
    Drop-in replacement for the HuggingFace SegFormerWrapper.
    All public method signatures are identical to the original.
    """

    def __init__(self, model_name: str = "jonathandinu/face-parsing"):
        self._ready = _get_session() is not None
        if self._ready:
            logger.info("SegFormerWrapper (ONNX): ready.")
        else:
            logger.warning(
                "SegFormerWrapper (ONNX): session not ready. "
                "Run export_segformer_to_onnx.py to generate the ONNX model."
            )

        # Face label map — identical to original
        self.labels = {
            'background': 0, 'skin': 1, 'l_brow': 2, 'r_brow': 3,
            'l_eye': 4, 'r_eye': 5, 'eye_g': 6, 'l_ear': 7,
            'r_ear': 8, 'ear_r': 9, 'nose': 10, 'mouth': 11,
            'u_lip': 12, 'l_lip': 13, 'neck': 14, 'neck_l': 15,
            'cloth': 16, 'hair': 17, 'hat': 18,
        }

        # MediaPipe (kept as-is, no changes to eye/mouth helpers)
        self.face_mesh = None
        self.mp_face_mesh = None

    @property
    def is_ready(self) -> bool:
        return _get_session() is not None

    # ── Core prediction ──────────────────────────────────────

    def validate_image(self, image, context="image"):
        if image is None:
            return False, f"{context} is None"
        if not isinstance(image, (np.ndarray, Image.Image)):
            return False, f"{context} type invalid: {type(image)}"
        if isinstance(image, np.ndarray):
            if image.size == 0: return False, f"{context} is empty"
            h, w = image.shape[:2]
            if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE: return False, f"{context} too small: {w}x{h}"
            if h > MAX_IMAGE_SIZE or w > MAX_IMAGE_SIZE: return False, f"{context} too large: {w}x{h}"
            if np.isnan(image).any() or np.isinf(image).any(): return False, f"{context} has NaN/Inf"
        return True, "Valid"

    def predict(self, image) -> np.ndarray:
        """
        Run ONNX segmentation on image (BGR numpy or PIL RGB).
        Returns class mask (H, W) uint8.
        """
        is_valid, msg = self.validate_image(image, "predict input")
        if not is_valid:
            logger.error(f"SegFormer predict validation: {msg}")
            if isinstance(image, np.ndarray) and image.ndim >= 2:
                return np.zeros(image.shape[:2], dtype=np.uint8)
            return np.zeros((480, 640), dtype=np.uint8)

        sess = _get_session()
        if sess is None:
            logger.warning("SegFormer ONNX session is None, returning empty mask.")
            if isinstance(image, np.ndarray): return np.zeros(image.shape[:2], dtype=np.uint8)
            w, h = image.size; return np.zeros((h, w), dtype=np.uint8)

        try:
            # Normalise to BGR numpy
            if isinstance(image, Image.Image):
                img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            elif image.ndim == 2:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                img_bgr = image

            orig_h, orig_w = img_bgr.shape[:2]
            tensor = _preprocess(img_bgr)
            input_name = sess.get_inputs()[0].name
            logits = sess.run(None, {input_name: tensor})[0]
            return _postprocess(logits, orig_h, orig_w)

        except Exception as e:
            logger.error(f"SegFormer predict error: {e}")
            if isinstance(image, np.ndarray) and image.ndim >= 2:
                return np.zeros(image.shape[:2], dtype=np.uint8)
            return np.zeros((480, 640), dtype=np.uint8)

    # ── Eye / Mouth helpers (unchanged from original) ────────

    def get_eye_rois(self, mask, image_bgr):
        rois = []
        h, w = mask.shape
        for name, idx in {'Left Eye': 4, 'Right Eye': 5}.items():
            eye_mask = (mask == idx).astype(np.uint8)
            contours, _ = cv2.findContours(eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(c)
                pad = int(max(cw, ch) * 0.5)
                x1 = max(0, x - pad); y1 = max(0, y - int(pad * 0.5))
                x2 = min(w, x + cw + pad); y2 = min(h, y + ch + int(pad * 0.5))
                crop = image_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    rois.append((crop, name, (x1, y1, x2, y2)))
        return rois

    def apply_iris_mask(self, eye_img):
        if eye_img is None or eye_img.size == 0: return eye_img, ""
        h, w = eye_img.shape[:2]
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=w/2,
            param1=50, param2=30,
            minRadius=int(min(h, w) * 0.10), maxRadius=int(min(h, w) * 0.55)
        )
        mask = np.ones_like(eye_img) * 255
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for cx, cy, r in circles[0]:
                if abs(cx - w//2) < w*0.4 and abs(cy - h//2) < h*0.4:
                    cv2.circle(mask, (int(cx), int(cy)), int(r * 1.1), (0, 0, 0), -1)
                    return cv2.bitwise_and(eye_img, mask), ""
        # Fallback contour method
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred2 = cv2.GaussianBlur(enhanced, (7, 7), 0)
        _, thresh = cv2.threshold(blurred2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if cv2.countNonZero(thresh) / thresh.size > 0.6:
            _, thresh = cv2.threshold(blurred2, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_c, max_score, center = None, 0, (w//2, h//2)
        for c in contours:
            area = cv2.contourArea(c)
            if area < h*w*0.02 or area > h*w*0.55: continue
            perim = cv2.arcLength(c, True)
            if perim == 0: continue
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx2 = int(M["m10"] / M["m00"]); cy2 = int(M["m01"] / M["m00"])
                if abs(cx2 - center[0]) < w*0.4:
                    score = area * (1 + 4*np.pi*area/(perim**2))
                    if score > max_score: max_score, best_c = score, c
        if best_c is not None:
            cv2.drawContours(mask, [best_c], -1, (0, 0, 0), -1)
            (x, y), radius = cv2.minEnclosingCircle(best_c)
            cv2.circle(mask, (int(x), int(y)), int(min(radius*1.1, min(h,w)*0.45)), (0, 0, 0), -1)
        return cv2.bitwise_and(eye_img, mask), ""

    def get_skin_mask(self, mask):
        skin_mask = np.isin(mask, [1, 7, 8, 14]).astype(np.uint8) * 255
        kernel_c = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_c)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filtered = np.zeros_like(skin_mask)
            min_area = skin_mask.shape[0] * skin_mask.shape[1] * 0.005
            found = False
            for c in contours:
                if cv2.contourArea(c) > min_area:
                    cv2.drawContours(filtered, [c], -1, 255, -1); found = True
            if found: return filtered
        return skin_mask

    def get_skin_mask_color(self, image_bgr):
        hsv   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 15, 50], np.uint8), np.array([20, 255, 255], np.uint8))
        ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        mask2 = cv2.inRange(ycbcr, np.array([0, 133, 77], np.uint8), np.array([255, 173, 127], np.uint8))
        combined = cv2.bitwise_and(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    def init_mediapipe(self):
        if hasattr(self, 'face_mesh') and self.face_mesh: return
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5
            )
        except ImportError:
            self.face_mesh = None

    def get_eyes_mediapipe(self, image_bgr):
        self.init_mediapipe()
        if not self.face_mesh: return []
        h, w = image_bgr.shape[:2]
        results = self.face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks: return []
        landmarks = results.multi_face_landmarks[0].landmark
        LEFT_EYE   = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        LEFT_IRIS  = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]
        rois = []
        for name, eye_idxs, iris_idxs in [('Left Eye', LEFT_EYE, LEFT_IRIS), ('Right Eye', RIGHT_EYE, RIGHT_IRIS)]:
            eye_pts  = np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye_idxs])
            iris_pts = np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in iris_idxs])
            x, y, ew, eh = cv2.boundingRect(eye_pts)
            px, py = int(ew*0.5), int(eh*0.8)
            x1, y1 = max(0, x-px), max(0, y-py)
            x2, y2 = min(w, x+ew+px), min(h, y+eh+py)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0: continue
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [eye_pts - [x1, y1]], 255)
            (cx, cy), r = cv2.minEnclosingCircle(iris_pts - [x1, y1])
            cv2.circle(mask, (int(cx), int(cy)), int(r), 0, -1)
            rois.append((cv2.bitwise_and(crop, crop, mask=mask), name, (x1, y1, x2, y2)))
        return rois

    def get_body_segmentation(self, image_bgr):
        try:
            import mediapipe as mp
            if not hasattr(self, 'segmenter'):
                self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            results = self.segmenter.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            if results.segmentation_mask is not None:
                return np.where(results.segmentation_mask > 0.5, 255, 0).astype(np.uint8)
        except Exception as e:
            logger.warning(f"Body segmentation failed: {e}")
        return np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255
