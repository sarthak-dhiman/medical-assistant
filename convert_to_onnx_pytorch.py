import torch
import torch.nn as nn
import timm
import os
import json
import sys
from pathlib import Path

# Import architectures from inference_pytorch (or redefine them if imports are tricky)
# To avoid import issues with relative paths if running as script, we'll just redefine/import carefully.
try:
    from inference_pytorch import JaundiceModel, JaundiceBodyModel, SkinDiseaseModel
except ImportError:
    # Fallback: redefine if inference_pytorch isn't in path
    print("WARNING: Could not import architectures from inference_pytorch. Using local definitions.")
    exit(1)

# Force CPU for export to avoid VRAM conflict with training
DEVICE = torch.device('cpu')
SAVE_DIR = Path("saved_models/onnx")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def export_body_model():
    print("--- Exporting Jaundice Body Model ---")
    path = Path("saved_models/jaundice_body_pytorch.pth")
    if not path.exists():
        print(f"Skipping Body: {path} not found")
        return

    model = JaundiceBodyModel().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()

    # Dummy Input: (1, 3, 380, 380) EfficientNetB4
    dummy_input = torch.randn(1, 3, 380, 380).to(DEVICE)
    
    output_path = SAVE_DIR / "jaundice_body.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Exported to {output_path}")

def export_eye_model():
    print("--- Exporting Jaundice Eye Model ---")
    path = Path("saved_models/jaundice_with_sclera_torch.pth")
    if not path.exists():
        print(f"Skipping Eye: {path} not found")
        return

    model = JaundiceModel().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()

    # Dummy Input: Two inputs (Skin, Sclera)
    # Skin: (1, 3, 380, 380)
    # Sclera: (1, 3, 64, 64)
    dummy_skin = torch.randn(1, 3, 380, 380).to(DEVICE)
    dummy_sclera = torch.randn(1, 3, 64, 64).to(DEVICE)
    
    output_path = SAVE_DIR / "jaundice_eye.onnx"
    torch.onnx.export(
        model, 
        (dummy_skin, dummy_sclera), 
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['skin_input', 'sclera_input'],
        output_names=['output'],
        dynamic_axes={
            'skin_input': {0: 'batch_size'}, 
            'sclera_input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ Exported to {output_path}")

def export_skin_model():
    print("--- Exporting Skin Disease Model ---")
    path = Path("saved_models/skin_disease_pytorch.pth")
    # Need to know num_classes (load from mapping)
    map_path = Path("saved_models/skin_disease_mapping.json")
    
    if not path.exists() or not map_path.exists():
        print(f"Skipping Skin: Files not found")
        return

    with open(map_path, 'r') as f:
        mapping = json.load(f)
        num_classes = len(mapping)

    model = SkinDiseaseModel(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()

    dummy_input = torch.randn(1, 3, 380, 380).to(DEVICE)
    
    output_path = SAVE_DIR / "skin_disease.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Exported to {output_path}")

if __name__ == "__main__":
    print(f"Exporting on device: {DEVICE}")
    export_body_model()
    export_eye_model()
    export_skin_model()
    print("Done!")
