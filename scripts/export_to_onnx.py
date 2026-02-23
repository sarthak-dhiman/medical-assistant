"""
export_to_onnx.py
=================
Converts PyTorch (.pth) models to ONNX format.
Includes Nails, Oral Cancer, Teeth, Skin, Jaundice, and Posture models.
"""

import os
import json
import torch
import torch.nn as nn
import timm
from pathlib import Path

SAVED_MODELS_DIR = Path("D:/Disease Prediction/saved_models")
ONNX_MODELS_DIR  = Path("D:/Disease Prediction/saved_models_onnx")
ONNX_MODELS_DIR.mkdir(exist_ok=True, parents=True)

# -----------------------------------------------------------------------------
# ARCHITECTURES
# -----------------------------------------------------------------------------

class NailDiseaseModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

class TeethDiseaseModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

class SkinDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

class JaundiceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.skin_backbone = timm.create_model('tf_efficientnet_b4', pretrained=False, num_classes=0)
        self.sclera_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(self.skin_backbone.num_features + 64, 256),
            nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
    def forward(self, skin, sclera):
        x1 = self.skin_backbone(skin)
        x2 = self.sclera_branch(sclera)
        x2 = x2.view(x2.size(0), -1)
        return self.classifier(torch.cat((x1, x2), dim=1))

class JaundiceBodyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

class PostureClassifier(nn.Module):
    def __init__(self, input_size=24, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# -----------------------------------------------------------------------------
# EXPORT LOGIC
# -----------------------------------------------------------------------------

def convert_to_onnx(model, dummy_input, pth_path, onnx_name, input_names=['input'], dynamic_axes=None):
    onnx_path = ONNX_MODELS_DIR / onnx_name
    if not pth_path.exists():
        print(f"Skipping: {onnx_name} (File not found {pth_path})")
        return
        
    try:
        sd = torch.load(pth_path, map_location='cpu')
        # Handle DataParallel if necessary
        if isinstance(sd, dict) and len(sd) > 0 and 'module.' in list(sd.keys())[0]:
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
        # Handle entire model save vs state dict
        if isinstance(sd, nn.Module):
            model = sd
        else:
            model.load_state_dict(sd)
            
        model.eval()
        
        if dynamic_axes is None:
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=input_names, output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        print(f"[Success] {onnx_name}")
    except Exception as e:
        print(f"[Failed] {onnx_name}: {e}")

def main():
    print(f"Starting Multi-Model ONNX Conversion...\nReading from: {SAVED_MODELS_DIR}")
    
    # 1. Nails
    convert_to_onnx(
        NailDiseaseModel(8),
        torch.randn(1, 3, 380, 380, requires_grad=False),
        SAVED_MODELS_DIR / "nail_disease_pytorch.pth",
        "nail_disease.onnx"
    )
    
    # 2. Oral Cancer (saved as object, pass empty module)
    convert_to_onnx(
        nn.Module(), # will be replaced by object load
        torch.randn(1, 3, 224, 224, requires_grad=False),
        SAVED_MODELS_DIR / "oral_cancer_model.pth",
        "oral_cancer.onnx"
    )
    
    # 3. Teeth
    convert_to_onnx(
        TeethDiseaseModel(7),
        torch.randn(1, 3, 224, 224, requires_grad=False),
        SAVED_MODELS_DIR / "teeth_model.pth",
        "teeth_disease.onnx"
    )
    
    # 4. Skin Disease
    skin_map_path = SAVED_MODELS_DIR / "skin_disease_mapping.json"
    if skin_map_path.exists():
        with open(skin_map_path) as f:
            num_skin_classes = len(json.load(f))
        convert_to_onnx(
            SkinDiseaseModel(num_skin_classes),
            torch.randn(1, 3, 380, 380, requires_grad=False),
            SAVED_MODELS_DIR / "skin_disease_pytorch.pth",
            "skin_disease.onnx"
        )
    else:
        print("Skipping Skin Disease: mapping JSON missing")

    # 5. Jaundice Body
    convert_to_onnx(
        JaundiceBodyModel(),
        torch.randn(1, 3, 380, 380, requires_grad=False),
        SAVED_MODELS_DIR / "jaundice_body_pytorch.pth",
        "jaundice_body.onnx"
    )
    
    # 6. Jaundice Eye (Multi-input)
    dummy_skin = torch.randn(1, 3, 380, 380, requires_grad=False)
    dummy_sclera = torch.randn(1, 3, 64, 64, requires_grad=False)
    convert_to_onnx(
        JaundiceModel(),
        (dummy_skin, dummy_sclera),
        SAVED_MODELS_DIR / "jaundice_with_sclera_torch.pth",
        "jaundice_eye.onnx",
        input_names=['skin_input', 'sclera_input'],
        dynamic_axes={'skin_input': {0: 'batch_size'}, 'sclera_input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # 7. Posture (Vector input)
    convert_to_onnx(
        PostureClassifier(input_size=24, num_classes=2),
        torch.randn(1, 24, requires_grad=False),
        SAVED_MODELS_DIR / "posture_classifier_mmpose.pth",
        "posture.onnx"
    )

if __name__ == "__main__":
    main()
