import os
import shutil
from pathlib import Path
from collections import Counter

# Define paths
BASE_DIR = Path(r"D:\Disease Prediction\Dataset\teeth\test")
LABELS_DIR = BASE_DIR / "labels"
IMAGES_DIR = BASE_DIR / "images"
SORTED_DIR = BASE_DIR / "sorted_by_class"

# Ensure sorted directory exists
if not SORTED_DIR.exists():
    SORTED_DIR.mkdir()

# Class mapping assumption (Alphabetical)
ALPHABETICAL_CLASSES = ['Calculus', 'Caries', 'Gingivitis', 'Healthy', 'Hypodontia', 'Mouth Ulcer', 'Tooth Discoloration']

def get_class_id(label_file):
    try:
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return None
            # First token is the class ID
            parts = content.split()
            return int(parts[0])
    except Exception as e:
        print(f"Error reading {label_file}: {e}")
        return None

def main():
    if not LABELS_DIR.exists() or not IMAGES_DIR.exists():
        print("Labels or Images directory not found!")
        return

    label_files = list(LABELS_DIR.glob("*.txt"))
    print(f"Found {len(label_files)} label files.")

    moved_count = 0
    class_Filename_sample = {}

    for label_file in label_files:
        # Find corresponding image
        image_file = IMAGES_DIR / (label_file.stem + ".jpg")
        if not image_file.exists():
            # Try .jpeg or .png just in case
            image_file = IMAGES_DIR / (label_file.stem + ".jpeg")
            if not image_file.exists():
                 image_file = IMAGES_DIR / (label_file.stem + ".png")
        
        if not image_file.exists():
            # print(f"Image not found for label: {label_file.name}")
            continue

        class_id = get_class_id(label_file)
        if class_id is None:
            continue

        # Track filename samples for verification
        if class_id not in class_Filename_sample:
            class_Filename_sample[class_id] = []
        if len(class_Filename_sample[class_id]) < 5:
            class_Filename_sample[class_id].append(image_file.name)

        # Create destination folder
        # We use Class_ID as folder name first to be safe
        class_folder_name = f"Class_{class_id}"
        
        # Heuristic mapping if confirmed (Logic: if definitive match found)
        # For now, let's stick to IDs to explain to user.
        
        dest_folder = SORTED_DIR / class_folder_name
        dest_folder.mkdir(exist_ok=True)

        target_path = dest_folder / image_file.name
        
        # Copy instead of move for safety first? User said "sort", usually implies move. 
        # I'll Move.
        try:
            shutil.move(str(image_file), str(target_path))
            moved_count += 1
        except Exception as e:
            print(f"Failed to move {image_file.name}: {e}")

    print(f"Successfully moved {moved_count} images to {SORTED_DIR}")
    
    print("\n--- Class Verification ---")
    sorted_ids = sorted(class_Filename_sample.keys())
    for cid in sorted_ids:
        samples = class_Filename_sample[cid]
        print(f"Class {cid}: {samples}")
        
    print("\n--- Proposed Mapping vs Alphabetical ---")
    for i, name in enumerate(ALPHABETICAL_CLASSES):
        print(f"ID {i}: {name}")

if __name__ == "__main__":
    main()
