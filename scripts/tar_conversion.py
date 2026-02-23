"""
tar_conversion.py  —  Direct Derm1M → WebDataset Shard Packer
=============================================================

This script skips the intermediate "sorted_by_class" folder stage.
It directly reads `Derm1M_v2_pretrain.csv` and `Derm1M_v2_validation.csv`,
locates the raw images in their respective directories (public, pubmed, edu, etc),
and packs them directly into WebDataset `.tar` shards for training.

Output:
    shards/
        derm1m_train_00000.tar  …
        derm1m_val_00000.tar    …
        class_mapping.json

Usage:
    python scripts/tar_conversion.py
"""

import csv
import json
import re
import os
from pathlib import Path
import webdataset as wds

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"D:\Disease Prediction\Dataset\Derm1M")

PRETRAIN_CSV  = DATASET_ROOT / "Derm1M_v2_pretrain.csv"
VALID_CSV     = DATASET_ROOT / "Derm1M_v2_validation.csv"
ONTOLOGY_PATH = DATASET_ROOT / "ontology.json"

# Source image directories
PRETRAIN_SUBDIRS    = ["public", "pubmed", "edu", "note", "IIYI", "youtube"]
VALIDATION_DATA_DIR = DATASET_ROOT / "validation_data"

SHARD_DIR      = DATASET_ROOT / "shards"
IMAGES_PER_TAR = 1000

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
# ─────────────────────────────────────────────────────────────────────────────


def is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes(200).startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def load_csv_index(csv_path: Path) -> dict[str, str]:
    """Return {filename_stem → original_label} from CSV."""
    index = {}
    if not csv_path.exists() or is_lfs_pointer(csv_path):
        return index
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fname = row.get("filename", "").strip()
            label = (row.get("disease_label", "").strip() or row.get("label", "").strip())
            if fname and label:
                index[Path(fname).stem] = label
    return index


def build_ontology_lookup(path: Path):
    """Builds a fallback lookup list of (lowercase_class, original_class)."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        classes = [k for k in json.load(f) if k.lower() != "root"]
    return sorted([(c.lower(), c) for c in classes], key=lambda t: len(t[0]), reverse=True)


_SEP = re.compile(r"[\W_]+")
def classify_by_filename(stem: str, lookup) -> str:
    """Fallback: guess class from filename string matching ontology."""
    text = _SEP.sub(" ", stem).lower()
    for lc, orig in lookup:
        if lc in text:
            return orig
    return "__unclassified__"


def sanitise_label(name: str) -> str:
    """Clean up labels to be consistent."""
    name = re.sub(r'[\r\n\t]+', ' ', name)
    name = name.strip()
    return name.lower()  # lowercase normalisation for mapping


def iter_images(subdirs: list[Path]):
    for d in subdirs:
        if d.exists():
            for p in d.rglob("*"):
                if p.suffix.lower() in IMAGE_EXTS:
                    yield p


def pack_split(
    image_dirs: list[Path],
    csv_index: dict[str, str],
    ontology_lookup: list[tuple],
    class_to_idx: dict[str, int],
    shard_pattern: str,
    images_per_tar: int,
) -> int:
    total = 0
    with wds.ShardWriter(shard_pattern, maxcount=images_per_tar) as sink:
        for img_path in iter_images(image_dirs):
            raw_label = csv_index.get(img_path.stem)
            if not raw_label:
                raw_label = classify_by_filename(img_path.stem, ontology_lookup)
                
            if raw_label == "__unclassified__":
                continue

            cleaned_label = sanitise_label(raw_label)
            
            if cleaned_label not in class_to_idx:
                class_to_idx[cleaned_label] = len(class_to_idx)
                
            class_idx = class_to_idx[cleaned_label]

            try:
                image_bytes = img_path.read_bytes()
            except OSError:
                continue

            sink.write({
                "__key__": f"{class_idx:04d}_{total:07d}",
                "jpg":     image_bytes,
                "cls":     str(class_idx).encode(),
                "txt":     cleaned_label.encode(),
            })
            total += 1

            if total % 10_000 == 0:
                print(f"    {total:>8,} images packed …")

    return total


def main():
    print("=" * 64)
    print("  Direct Derm1M → WebDataset Shard Packer  (No sort step)")
    print("=" * 64)

    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    print("  Loading CSVs and Ontology …")
    train_idx = load_csv_index(PRETRAIN_CSV)
    val_idx   = load_csv_index(VALID_CSV)
    ontology  = build_ontology_lookup(ONTOLOGY_PATH)

    print(f"    Train labels: {len(train_idx):,}")
    print(f"    Val labels  : {len(val_idx):,}")

    class_to_idx = {}  # Discovered dynamically as we process images

    # ── 1. Pack Train Split ───────────────────────────────────────────────────
    print("\n  Packing Train split …")
    train_dirs = [DATASET_ROOT / s for s in PRETRAIN_SUBDIRS]
    train_pattern = os.path.relpath(SHARD_DIR / "derm1m_train_%05d.tar").replace("\\", "/")
    train_total = pack_split(
        train_dirs, train_idx, ontology, class_to_idx, train_pattern, IMAGES_PER_TAR
    )
    print(f"  ✓ Train: {train_total:,} images packed.")

    # ── 2. Pack Val Split ─────────────────────────────────────────────────────
    print("\n  Packing Validation split …")
    val_pattern = os.path.relpath(SHARD_DIR / "derm1m_val_%05d.tar").replace("\\", "/")
    val_total = pack_split(
        [VALIDATION_DATA_DIR], val_idx, ontology, class_to_idx, val_pattern, IMAGES_PER_TAR
    )
    print(f"  ✓ Val  : {val_total:,} images packed.")

    # ── 3. Save Master Class Mapping ──────────────────────────────────────────
    mapping_path = SHARD_DIR / "class_mapping.json"
    idx_to_class = {str(idx): name for name, idx in class_to_idx.items()}
    with open(mapping_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    
    print(f"\n  Final Class Count: {len(class_to_idx)}")
    print(f"  Class mapping saved → {mapping_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()