import torch
import sys
import os

PATHS = [
    r"D:\Disease Prediction\cataract_model.pth",
    r"D:\Disease Prediction\saved_models\cataract_model.pth"
]

def inspect_shapes():
    for MODEL_PATH in PATHS:
        if not os.path.exists(MODEL_PATH):
            continue
            
        print(f"Inspecting shapes in {MODEL_PATH}...")
        try:
            data = torch.load(MODEL_PATH, map_location='cpu')
            
            if isinstance(data, dict):
                print("First layer shape:", data['features.0.0.weight'].shape)
                # Check classifier input
                if 'classifier.1.1.weight' in data:
                     print("Classifier weight shape:", data['classifier.1.1.weight'].shape)
                elif 'classifier.1.weight' in data:
                     print("Classifier weight shape:", data['classifier.1.weight'].shape)
            else:
                print("Object is not a dict.")
                
            return
                
        except Exception as e:
            print(f"Error loading {MODEL_PATH}: {e}")

if __name__ == "__main__":
    inspect_shapes()
