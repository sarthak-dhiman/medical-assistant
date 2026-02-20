"""
sort_derm1m.py  —  Derm1M Dataset Organiser (v4)
===================================================

Reads both CSVs, classifies every image by its 'disease_label', copies images
into:

    sorted_by_class/
        train/
            <disease_label>/
                image1.png  …
        val/
            <disease_label>/
                image1.png  …

• Pretrain CSV  → train split
• Validation CSV → val split
• Images not covered by either CSV get fallback filename-keyword classification
  and are placed in the train split as unvalidated data.

Output at end:
  - printed class distribution table (train | val | total)
  - derm1m_distribution.csv next to this script

Usage
-----
    python sort_derm1m.py

Edit the CONFIGURATION section before running.
"""

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"D:\Disease Prediction\Dataset\Derm1M")

PRETRAIN_CSV  = DATASET_ROOT / "Derm1M_v2_pretrain.csv"
VALID_CSV     = DATASET_ROOT / "Derm1M_v2_validation.csv"
ONTOLOGY_PATH = DATASET_ROOT / "ontology.json"

# Source image sub-directories (for pretrain / uncovered images):
PRETRAIN_SUBDIRS = ["public", "pubmed", "edu", "note", "IIYI", "youtube"]

# Validation images live in validation_data/ alongside the CSV:
VALIDATION_DATA_DIR = DATASET_ROOT / "validation_data"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

OUTPUT_DIR  = DATASET_ROOT / "sorted_by_class"
TRAIN_DIR   = OUTPUT_DIR / "train"
VAL_DIR     = OUTPUT_DIR / "val"

MOVE_FILES     = True  # True = move instead of copy (destructive!)
MAX_IMAGES     = None   # None = all; set e.g. 10_000 for a quick test run
PROGRESS_EVERY = 10_000
# ────────────────────────────────────────────────────────────────────────────


# ── Helpers ─────────────────────────────────────────────────────────────────

def is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes(200).startswith(
            b"version https://git-lfs.github.com/spec/v1"
        )
    except Exception:
        return False


def load_csv_index(
    csv_path: Path,
    label_col: str = "disease_label",
    filename_col: str = "filename",
) -> dict[str, str]:
    """Return {filename_stem → label} from a CSV that has the given columns."""
    index: dict[str, str] = {}
    if not csv_path.exists() or is_lfs_pointer(csv_path):
        return index
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fname = row.get(filename_col, "").strip()
            label = (
                row.get(label_col, "").strip()
                or row.get("label", "").strip()
            )
            if fname and label:
                index[Path(fname).stem] = label  # key = stem only
    return index


def load_ontology_classes(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        o = json.load(f)
    return [k for k in o if k.lower() != "root"]


def build_lookup(classes: list[str]):
    return sorted(
        [(c.lower(), c) for c in classes],
        key=lambda t: len(t[0]),
        reverse=True,
    )


_SEP = re.compile(r"[\W_]+")


def classify_by_filename(stem: str, lookup) -> str:
    text = _SEP.sub(" ", stem).lower()
    for lc, orig in lookup:
        if lc in text:
            return orig
    return ""


def sanitise(name: str) -> str:
    # Remove newlines and unprintable chars
    name = re.sub(r'[\r\n\t]+', ' ', name)
    
    # Remove invalid Windows folder characters
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
        
    name = name.strip()
    
    # Windows doesn't like trailing dots
    while name.endswith('.'):
        name = name[:-1].strip()
        
    # Truncate to a safe length for Windows path limits
    # (e.g., 60 characters should be more than enough for actual disease names)
    if len(name) > 60:
        name = name[:60].strip()
        
    return name or "__invalid__"


def copy_or_move(src: Path, dst: Path) -> bool:
    """Copy (or move) src → dst. Returns True on success."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return False   # already handled
        if MOVE_FILES:
            shutil.move(str(src), dst)
        else:
            shutil.copy2(src, dst)
        return True
    except Exception as exc:
        print(f"  [ERR] {src}: {exc}")
        return False


def iter_images(subdirs: list[Path]):
    for d in subdirs:
        if not d.exists():
            print(f"  [WARN] not found, skipping: {d}")
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS:
                yield p


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 72
    print(SEP)
    print("  Derm1M Dataset Sorter  (v4 — with train / val split)")
    print(SEP)

    # ── Load label indices ────────────────────────────────────────────────────
    print("  Loading pretrain CSV …")
    pretrain_idx = load_csv_index(PRETRAIN_CSV)
    if pretrain_idx:
        print(f"  ✓ Pretrain: {len(pretrain_idx):,} labelled entries")
    else:
        print("  ⚠ Pretrain CSV not available or still a LFS pointer.")

    print("  Loading validation CSV …")
    val_idx = load_csv_index(VALID_CSV)
    if val_idx:
        print(f"  ✓ Validation: {len(val_idx):,} labelled entries")
    else:
        print("  ⚠ Validation CSV not available or still a LFS pointer.")

    # ── Ontology fallback ─────────────────────────────────────────────────────
    lookup = []
    if ONTOLOGY_PATH.exists():
        lookup = build_lookup(load_ontology_classes(ONTOLOGY_PATH))
        print(f"  Ontology: {len(lookup)} classes loaded (for fallback).")
    else:
        print(f"  ⚠ Ontology not found: {ONTOLOGY_PATH}")

    # ── Counters ──────────────────────────────────────────────────────────────
    train_dist: Counter = Counter()
    val_dist:   Counter = Counter()
    stats = defaultdict(int)  # matched_csv, matched_fn, unclassified, errors

    total_processed = 0

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 1 — VAL SPLIT  (images listed in validation CSV)
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("  ── Pass 1: Validation images ──────────────────────────────────")

    if val_idx and VALIDATION_DATA_DIR.exists():
        val_stems = {stem: label for stem, label in val_idx.items()}
        for img_path in iter_images([VALIDATION_DATA_DIR]):
            if MAX_IMAGES and total_processed >= MAX_IMAGES:
                break

            label = val_stems.get(img_path.stem, "")
            if label:
                stats["matched_csv"] += 1
            else:
                label = classify_by_filename(img_path.stem, lookup)
                if label:
                    stats["matched_fn"] += 1
                else:
                    label = "__unclassified__"
                    stats["unclassified"] += 1

            dest = VAL_DIR / sanitise(label) / img_path.name
            if copy_or_move(img_path, dest):
                total_processed += 1
                val_dist[label] += 1
            else:
                val_dist[label] += 1  # count even if dup

            if total_processed % PROGRESS_EVERY == 0 and total_processed:
                print(f"    {total_processed:>8,} images done …")
    else:
        print("    validation_data/ folder or index not found — skipping.")

    val_total = sum(val_dist.values())
    print(f"    Done. {val_total:,} validation images sorted into {len(val_dist)} classes.")

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 2 — TRAIN SPLIT  (all images in source subdirs via pretrain CSV)
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("  ── Pass 2: Pretrain / train images ────────────────────────────")

    subdirs = [DATASET_ROOT / s for s in PRETRAIN_SUBDIRS]
    for img_path in iter_images(subdirs):
        if MAX_IMAGES and total_processed >= MAX_IMAGES:
            break

        # Build a relative key the same way the pretrain CSV does:
        # e.g.  "IIYI/26549_2.png"  → stem "26549_2"
        label = pretrain_idx.get(img_path.stem, "")
        if label:
            stats["matched_csv"] += 1
        else:
            label = classify_by_filename(img_path.stem, lookup)
            if label:
                stats["matched_fn"] += 1
            else:
                label = "__unclassified__"
                stats["unclassified"] += 1

        dest = TRAIN_DIR / sanitise(label) / img_path.name
        if copy_or_move(img_path, dest):
            total_processed += 1
            train_dist[label] += 1
        else:
            train_dist[label] += 1

        if total_processed % PROGRESS_EVERY == 0 and total_processed:
            print(f"    {total_processed:>8,} images done …")

    train_total = sum(train_dist.values())
    print(f"    Done. {train_total:,} train images sorted into {len(train_dist)} classes.")

    # ══════════════════════════════════════════════════════════════════════════
    # DISTRIBUTION TABLE
    # ══════════════════════════════════════════════════════════════════════════
    all_classes = sorted(set(train_dist) | set(val_dist))
    grand_train = sum(train_dist.values())
    grand_val   = sum(val_dist.values())
    grand_total  = grand_train + grand_val

    print()
    print(SEP)
    print(f"  CLASS DISTRIBUTION  (train={grand_train:,}  |  val={grand_val:,}  |  total={grand_total:,})")
    print(SEP)
    print(f"  {'Class':<45} {'Train':>8}  {'Val':>7}  {'Total':>8}")
    print(f"  {'-'*45} {'-'*8}  {'-'*7}  {'-'*8}")

    sorted_classes = sorted(
        all_classes,
        key=lambda c: -(train_dist[c] + val_dist[c])
    )
    for cls in sorted_classes:
        tr = train_dist[cls]
        va = val_dist[cls]
        print(f"  {cls:<45} {tr:>8,}  {va:>7,}  {tr+va:>8,}")

    print(f"  {'-'*45} {'-'*8}  {'-'*7}  {'-'*8}")
    print(f"  {'TOTAL':<45} {grand_train:>8,}  {grand_val:>7,}  {grand_total:>8,}")

    print()
    print(f"  ✓ Label CSV matched  : {stats['matched_csv']:,}")
    print(f"  ~ Filename matched   : {stats['matched_fn']:,}")
    print(f"  ? Unclassified       : {stats['unclassified']:,}")
    print(f"  ✗ Errors             : {stats['errors']:,}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    dist_csv = Path(__file__).parent / "derm1m_distribution.csv"
    with open(dist_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "train_count", "val_count", "total", "train_%", "val_%"])
        for cls in sorted_classes:
            tr, va = train_dist[cls], val_dist[cls]
            tot = tr + va
            w.writerow([
                cls, tr, va, tot,
                f"{tr/grand_train*100:.2f}%" if grand_train else "0%",
                f"{va/grand_val*100:.2f}%"   if grand_val   else "0%",
            ])
    print(f"\n  Distribution saved → {dist_csv}")
    print(SEP)


if __name__ == "__main__":
    main()
