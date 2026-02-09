#!/usr/bin/env python3
"""
Copy and rename images from the jaundice_sclera dataset into a verification folder.

Usage:
    python scripts/rename_jaundice_images.py \
        --src "D:/Disease Prediction/Dataset/jaundice_sclera" \
        --dst "D:/Disease Prediction/renamed_jaundice_sclera"

Default behavior: uses the Dataset/jaundice_sclera folder in the repo root and
writes renamed copies (does not modify originals). A CSV mapping file is created
in the destination folder for manual verification.

Filename format used: {prefix}_{relative_folder}_s.no{counter:06d}{ext}
Example: jaundice_JPEGImages_s.no000123.jpg

The script is conservative (skips non-image files) and prints a short summary.
"""
from pathlib import Path
import argparse
import shutil
import csv
import re
import sys

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def sanitize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', s)


def main():
    parser = argparse.ArgumentParser(description='Copy & rename jaundice_sclera images')
    parser.add_argument('--src', type=str, default=r'D:/Disease Prediction/Dataset/jaundice_sclera',
                        help='Source dataset folder')
    parser.add_argument('--dst', type=str, default=r'D:/Disease Prediction/renamed_jaundice_sclera',
                        help='Destination folder to place renamed copies')
    parser.add_argument('--prefix', type=str, default='jaundice', help='Prefix for new filenames')
    parser.add_argument('--dry-run', action='store_true', help='Show planned actions but do not copy')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    prefix = sanitize(args.prefix)

    if not src.exists():
        print(f"Source folder does not exist: {src}")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    mapping_path = dst / 'mapping.csv'

    counter = 1
    mappings = []

    for root, dirs, files in sorted(os_walk(src)):
        root_path = Path(root)
        try:
            rel = root_path.relative_to(src)
            rel_str = rel.as_posix().replace('/', '_') if str(rel) != '.' else 'root'
        except Exception:
            rel_str = root_path.name or 'root'
        rel_str = sanitize(rel_str)

        for fname in sorted(files):
            p = root_path / fname
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext not in IMAGE_EXTS:
                continue

            # Build new name
            new_name = f"{prefix}_{rel_str}_s.no{counter:06d}{ext}"
            new_path = dst / new_name
            # Avoid collisions by incrementing counter
            while new_path.exists():
                counter += 1
                new_name = f"{prefix}_{rel_str}_s.no{counter:06d}{ext}"
                new_path = dst / new_name

            if args.dry_run:
                print(f"[DRY] {p} -> {new_path}")
            else:
                shutil.copy2(p, new_path)

            mappings.append((str(p.resolve()), str(new_path.resolve())))
            counter += 1

    # Write mapping CSV
    if not args.dry_run:
        with mapping_path.open('w', newline='', encoding='utf-8') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['original_path', 'renamed_path'])
            for a, b in mappings:
                writer.writerow([a, b])

    print(f"Processed {len(mappings)} images")
    print(f"Renamed copies (and mapping) written to: {dst}")


# small helper to ensure deterministic walk ordering across platforms
def os_walk(src_path: Path):
    # yields tuples like os.walk
    for root, dirs, files in __import__('os').walk(src_path):
        dirs.sort()
        files.sort()
        yield root, dirs, files


if __name__ == '__main__':
    main()
