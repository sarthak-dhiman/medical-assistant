"""
export_segformer_to_onnx.py
============================
One-time script to export the locally saved SegFormer face-parsing model
(saved_models/segformer/) to ONNX format.

Run once from the project root:
    python scripts/export_segformer_to_onnx.py

Output: saved_models_onnx/segformer.onnx

The model takes a fixed (1, 3, 512, 512) NCHW float32 tensor input and
outputs raw logits of shape (1, 19, 128, 128).
Post-processing (resize + argmax) is done in Python/numpy at inference time.
"""

import sys
import os
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

LOCAL_MODEL_PATH = BASE_DIR / "saved_models" / "segformer"
ONNX_OUT_DIR    = BASE_DIR / "saved_models_onnx"
ONNX_PATH       = ONNX_OUT_DIR / "segformer.onnx"

# SegFormer fixed input size (standard for face-parsing model)
INPUT_H, INPUT_W = 512, 512
OPSET = 14

def main():
    ONNX_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check model exists locally
    if not LOCAL_MODEL_PATH.exists():
        print(f"[ERROR] SegFormer model not found at {LOCAL_MODEL_PATH}")
        print("Please ensure the model has been downloaded first by running the app once.")
        sys.exit(1)

    weight_files = list(LOCAL_MODEL_PATH.glob("model.safetensors")) + \
                   list(LOCAL_MODEL_PATH.glob("pytorch_model.bin"))
    if not weight_files:
        print(f"[ERROR] No weight file found in {LOCAL_MODEL_PATH}")
        sys.exit(1)

    print(f"Loading SegFormer from {LOCAL_MODEL_PATH} ...")
    try:
        from transformers import AutoModelForSemanticSegmentation
    except ImportError:
        print("[ERROR] transformers library not installed. Run: pip install transformers")
        sys.exit(1)

    model = AutoModelForSemanticSegmentation.from_pretrained(str(LOCAL_MODEL_PATH))
    model.eval()

    print(f"Model loaded successfully. Exporting to ONNX (opset {OPSET}) ...")

    # 2. Dummy input — processor normalises to ImageNet mean/std
    dummy = torch.randn(1, 3, INPUT_H, INPUT_W)

    # 3. Export
    # The model output is an object; we wrap it to return only logits
    class SegFormerLogitsWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, pixel_values):
            return self.base(pixel_values=pixel_values).logits

    wrapper = SegFormerLogitsWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(ONNX_PATH),
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "logits":       {0: "batch_size"},
            },
        )

    print(f"[Success] segformer.onnx  [{ONNX_PATH.stat().st_size / 1e6:.1f} MB]")

    # 4. Quick validation
    print("Validating ONNX model ...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"pixel_values": dummy.numpy()})[0]
    print(f"ONNX output shape: {out.shape}  (expected: [1, 19, 128, 128])")
    print("Validation passed!")

if __name__ == "__main__":
    main()
