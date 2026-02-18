import json
import cv2
import os
import argparse
import numpy as np
from pathlib import Path

def draw_skeleton(img, keypoints, bbox=None, label=None):
    """
    Draws keypoints and skeleton connections on the image.
    Assumes standard COCO format or the specific 6-point posture format.
    
    Keypoints: list of [x, y, v]
    """
    h, w = img.shape[:2]
    
    # Define connections (indices)
    # 0: Nose, 1: L_Sho, 2: R_Sho, 3: L_Hip, 4: R_Hip, 5: Mid/Spine (if present)
    # Basic Box connections
    connections = [
        (1, 2), # Shoulders
        (1, 3), # Left Side
        (2, 4), # Right Side
        (3, 4), # Hips
    ]
    
    # Optional connections
    if len(keypoints) >= 6 * 3:
         connections.extend([
             (0, 1), (0, 2), # Nose to shoulders
             (5, 1), (5, 2), (5, 3), (5, 4) # Mid point connections if applicable
         ])
    elif len(keypoints) >= 5 * 3:
        connections.extend([
             (0, 1), (0, 2)
        ])

    # Convert raw keypoints list to struct
    pts = []
    for i in range(0, len(keypoints), 3):
        x = int(keypoints[i])
        y = int(keypoints[i+1])
        v = keypoints[i+2]
        pts.append((x, y, v))

    # Draw Connections (Limbs)
    for i, j in connections:
        if i < len(pts) and j < len(pts):
            pt1 = pts[i]
            pt2 = pts[j]
            
            # Check visibility (v > 0 means visible or labeled)
            if pt1[2] > 0 and pt2[2] > 0:
                cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 255), 2) # Yellow lines

    # Draw Keypoints (Joints)
    for i, pt in enumerate(pts):
        if pt[2] > 0:
            color = (0, 0, 255) if i == 0 else (0, 255, 0) # Red for nose, Green for others
            cv2.circle(img, (pt[0], pt[1]), 5, color, -1)
            
    # Draw Bounding Box if provided
    if bbox:
        x, y, bw, bh = [int(v) for v in bbox]
        cv2.rectangle(img, (x, y), (x+bw, y+bh), (255, 0, 0), 2)
        if label:
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return img

def main():
    parser = argparse.ArgumentParser(description="Apply skeleton keypoints to images.")
    parser.add_argument("--json_path", required=True, help="Path to COCO annotations JSON")
    parser.add_argument("--image_dir", required=True, help="Directory containing source images")
    parser.add_argument("--output_dir", required=True, help="Directory to save visualized images")
    
    args = parser.parse_args()

    # Load JSON
    print(f"Loading annotations from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    # Prepare directories
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Index images by ID
    images_map = {img['id']: img for img in data['images']}
    
    # Group annotations by image_id
    anns_by_img = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)

    # Process
    count = 0
    total = len(images_map)
    print(f"Processing {total} images...")
    
    for img_id, img_info in images_map.items():
        file_name = img_info['file_name']
        src_path = image_dir / file_name
        
        if not src_path.exists():
            # Try just filename if src_path fails (some datasets have subdirs in json but flat dir structure)
            src_path = image_dir / Path(file_name).name
            if not src_path.exists():
                 # Try searching in subdirectories (depth 1)
                 found = False
                 for sub in image_dir.iterdir():
                     if sub.is_dir():
                         possible_path = sub / Path(file_name).name
                         if possible_path.exists():
                             src_path = possible_path
                             found = True
                             break
                 
                 if not found:
                    # print(f"Warning: Image not found: {src_path}")
                    continue

        # Load Image
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"Warning: Failed to load image: {src_path}")
            continue

        # Get Annotations
        anns = anns_by_img.get(img_id, [])
        
        # Iterate annotations
        for ann in anns:
            kps = ann.get('keypoints', [])
            bbox = ann.get('bbox', [])
            cat_id = ann.get('category_id', '')
            
            if kps:
                draw_skeleton(img, kps, bbox, f"Class {cat_id}")
            elif bbox: # Draw bbox even if no keypoints
                 x, y, bw, bh = [int(v) for v in bbox]
                 cv2.rectangle(img, (x, y), (x+bw, y+bh), (255, 0, 0), 2)


        # Save
        out_path = output_dir / Path(file_name).name
        cv2.imwrite(str(out_path), img)
        count += 1
        
        if count % 10 == 0:
            print(f"Processed {count}/{total} images...")

    print(f"Done. Saved {count} visualized images to {output_dir}")

if __name__ == "__main__":
    main()
