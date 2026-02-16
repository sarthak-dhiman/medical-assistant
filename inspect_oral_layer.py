import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from inference_new_models import get_oral_cancer_model

def inspect_layers():
    print("--- Inspecting Oral Cancer Model Layers ---")
    try:
        model = get_oral_cancer_model()
        if model is None:
            print("Failed to load model.")
            return

        print(f"Model type: {type(model)}")
        
        if hasattr(model, 'conv_head'):
            print(f"Has conv_head: {type(model.conv_head)}")
            print(model.conv_head)
        else:
            print("No conv_head")
            
        if hasattr(model, 'blocks'):
            print(f"Has blocks: {len(model.blocks)} blocks")
            print(f"Last block: {type(model.blocks[-1])}")
            print(model.blocks[-1])
        else:
            print("No blocks")
            
        # Check children
        print("\nNamed Children (last 3):")
        children = list(model.named_children())
        for name, child in children[-3:]:
             print(f"{name}: {type(child)}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_layers()
