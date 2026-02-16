
import sys
import os

print("--- DEBUGGING MODEL LOADING ---")

print("\n1. Testing SegFormer (Should work)...")
try:
    from segformer_utils import SegFormerWrapper
    print("🛫 SegFormer Imported.")
except Exception as e:
    print(f"🛬 SegFormer Failed: {e}")

print("\n2. Testing Jaundice Eye (PyTorch)...")
try:
    # Set CPU to avoid CUDA crashes
    import torch
    print(f"   Torch Version: {torch.__version__}")
    from inference_pytorch import get_model
    model = get_model()
    if model:
        print("🛫 PyTorch Model Loaded.")
    else:
        print("🛬 PyTorch Model returned None.")
except Exception as e:
    print(f"🛬 PyTorch Import Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing Skin Disease (Keras)...")
try:
    from inference_skin import predict_skin_disease, get_model_and_mapping
    model, mapping = get_model_and_mapping()
    if model:
        print("🛫 Skin Disease Model Loaded.")
    else:
        print("🛬 Skin Disease Model returned None.")
except Exception as e:
    print(f"🛬 Skin Disease Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n--- END DEBUG ---")
