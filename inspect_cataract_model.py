import torch
import sys
import os

PATHS = [
    r"D:\Disease Prediction\cataract_model.pth",
    r"D:\Disease Prediction\saved_models\cataract_model.pth"
]

def inspect_model():
    for MODEL_PATH in PATHS:
        if not os.path.exists(MODEL_PATH):
            continue
            
        print(f"Inspecting {MODEL_PATH}...")
        try:
            data = torch.load(MODEL_PATH, map_location='cpu')
            
            if isinstance(data, dict):
                keys = sorted(list(data.keys()))
                with open("model_keys.txt", "w") as f:
                    f.write(f"Total Keys: {len(keys)}\n")
                    f.write("All Keys:\n")
                    for k in keys:
                        f.write(f"{k}\n")
                print("Keys dumped to model_keys.txt")
                return
            else:
                print("Object is not a dict.")
                print(type(data))
                
        except Exception as e:
            print(f"Error loading {MODEL_PATH}: {e}")

if __name__ == "__main__":
    inspect_model()
