"""
tar_conversion.py  —  Pack Derm1M sorted dataset into WebDataset shards
==========================================================================

Converts the folder structure produced by sort_derm1m.py:

    sorted_by_class/
        train/<class_name>/<image>.jpg
        val/<class_name>/<image>.png

into WebDataset-compatible .tar shards:

    sorted_by_class/shards/
        derm1m_train_00000.tar  …
        derm1m_val_00000.tar    …

Each sample inside the tar contains:
    <key>.jpg    — raw image bytes
    <key>.cls    — integer class index (as ASCII text, e.g. "42\n")
    <key>.txt    — class name string (e.g. "psoriasis\n")

A class mapping JSON is also saved so you can recover {index → class_name}.

Usage
-----
    python scripts/tar_conversion.py
"""

import json
import webdataset as wds
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT   = Path("Dataset/Derm1M/sorted_by_class")
SHARD_DIR      = DATASET_ROOT / "shards"
IMAGES_PER_TAR = 1000          # how many samples per .tar file

SKIP_DIRS      = {"__unclassified__", "__invalid__", "shards"}
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# ─────────────────────────────────────────────────────────────────────────────


def get_sorted_classes(split_dir: Path) -> list[str]:
    """Return sorted class names found in a split directory."""
    return sorted(
        d.name
        for d in split_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    )


def pack_split(
    split_dir: Path,
    class_names: list[str],
    shard_pattern: str,
    images_per_tar: int,
) -> int:
    """Pack one split (train or val) and return total images written."""
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    total = 0

    with wds.ShardWriter(shard_pattern, maxcount=images_per_tar) as sink:
        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            class_idx = class_to_idx[class_name]

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                try:
                    image_bytes = img_path.read_bytes()
                except OSError as e:
                    print(f"  [WARN] Could not read {img_path}: {e}")
                    continue

                # Normalise extension to .jpg for simplicity inside the tar
                sink.write({
                    "__key__": f"{class_idx:04d}_{total:07d}",
                    "jpg":     image_bytes,            # raw image bytes
                    "cls":     str(class_idx).encode(),# integer index as bytes
                    "txt":     class_name.encode(),    # class name as bytes
                })
                total += 1

                if total % 10_000 == 0:
                    print(f"    {total:>8,} images written …")

    return total


def main():
    print("=" * 64)
    print("  Derm1M → WebDataset Shard Packer")
    print("=" * 64)

    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    # ── Discover classes from train/ (authoritative) ──────────────────────────
    train_dir = DATASET_ROOT / "train"
    val_dir   = DATASET_ROOT / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"train/ not found at {train_dir}\n"
            "Run sort_derm1m.py first."
        )

    class_names = get_sorted_classes(train_dir)
    print(f"  Classes found : {len(class_names)}")
    print(f"  Output dir    : {SHARD_DIR}\n")

    # Save class mapping
    mapping = {str(i): name for i, name in enumerate(class_names)}
    mapping_path = SHARD_DIR / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Class mapping saved → {mapping_path}\n")

    # ── Pack train ────────────────────────────────────────────────────────────
    print("  Packing train split …")
    train_pattern = str(SHARD_DIR / "derm1m_train_%05d.tar")
    train_total = pack_split(train_dir, class_names, train_pattern, IMAGES_PER_TAR)
    print(f"  ✓ Train: {train_total:,} images packed.\n")

    # ── Pack val ──────────────────────────────────────────────────────────────
    if val_dir.exists():
        print("  Packing val split …")
        val_pattern = str(SHARD_DIR / "derm1m_val_%05d.tar")
        val_total = pack_split(val_dir, class_names, val_pattern, IMAGES_PER_TAR)
        print(f"  ✓ Val  : {val_total:,} images packed.\n")
    else:
        print("  val/ directory not found — skipping val split.\n")

    print("=" * 64)
    print("  Done! Shards saved to:", SHARD_DIR)
    print("=" * 64)


if __name__ == "__main__":
    main()