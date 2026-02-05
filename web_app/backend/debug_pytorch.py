
import sys
import os

print("--- DEBUGGING PYTORCH ---")
try:
    import torch
    print(f"Torch Version: {torch.__version__}")
    
    # Force CPU
    device = torch.device('cpu')
    print(f"Using Device: {device}")
    
    from inference_pytorch import get_model
    model = get_model()
    
    if model:
        print("✅ PyTorch Model Loaded Successfully.")
    else:
        print("❌ PyTorch Model returned None.")
        
except Exception as e:
    print(f"❌ PyTorch Critical Failure: {e}")
    import traceback
    traceback.print_exc()

print("--- END PYTORCH DEBUG ---")
