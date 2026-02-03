import PyInstaller.__main__
import shutil
import os

def build():
    print("Building Medical Assistant...")
    
    # Clean dist/build
    if os.path.exists("dist"): shutil.rmtree("dist")
    if os.path.exists("build"): shutil.rmtree("build")

    # PyInstaller Arguments
    args = [
        'webcam_app.py',                     # Script
        '--name=MedicalAssistant',           # Name
        '--onedir',                          # Folder Mode (Important for updates)
        '--noconfirm',                       # Overwrite
        '--windowed',                        # Windowed mode (Reduces console lag)
        # '--add-data=saved_models;saved_models', # Bundle models? No, let's copy them manually for easy updates
        
        # Hidden Imports (Common ML missing links)
        '--hidden-import=sklearn.utils._typedefs',
        '--hidden-import=sklearn.neighbors._partition_nodes',
        '--hidden-import=sklearn.metrics._pairwise_distances_reduction',
        '--hidden-import=PIL._tkinter_finder',
        '--hidden-import=timm',
        '--hidden-import=torch',
        '--hidden-import=torchvision',
    ]
    
    PyInstaller.__main__.run(args)
    
    print("\nBuild Complete. Post-processing...")
    
    # Copy 'saved_models' to the dist folder so it's accessible and updatable
    src_models = "saved_models"
    dest_models = "dist/MedicalAssistant/saved_models"
    
    if os.path.exists(src_models):
        print(f"Copying models from {src_models} to {dest_models}...")
        if os.path.exists(dest_models): shutil.rmtree(dest_models)
        shutil.copytree(src_models, dest_models)
    else:
        print("Warning: saved_models not found in source!")

    print("\nDONE! You can find the app in: dist/MedicalAssistant/MedicalAssistant.exe")

if __name__ == "__main__":
    build()
