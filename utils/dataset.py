"""Dataset preparation utilities for wind turbine blade damage detection.

Provides:
  - Stratified train/val/test splitting (preserving positive/negative ratio)
  - Label sanitisation (clip bbox coords, drop invalid rows)
  - CLAHE preprocessing for UAV imagery (low-contrast turbine surfaces)
  - Dataset statistics reporting
"""

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

def split_dataset(
    pairs: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified split preserving the positive/negative class ratio.

    Args:
        pairs:       List of dicts with keys ``img``, ``lbl_in``, ``has_obj``.
        train_ratio: Fraction of data for training.
        val_ratio:   Fraction of data for validation (remainder → test).
        seed:        Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) lists.
    """
    pos = [p for p in pairs if p["has_obj"]]
    neg = [p for p in pairs if not p["has_obj"]]

    def _split(items: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        items = items.copy()
        random.Random(seed).shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]

    tr_p, va_p, te_p = _split(pos)
    tr_n, va_n, te_n = _split(neg)
    return tr_p + tr_n, va_p + va_n, te_p + te_n


def index_images(img_dir: Path, lbl_dir: Path) -> List[Dict]:
    """Index every image and check whether its YOLO label is non-empty.

    Args:
        img_dir: Directory containing image files.
        lbl_dir: Directory containing YOLO ``.txt`` label files.

    Returns:
        List of dicts: ``{"img": Path, "lbl_in": Path | None, "has_obj": bool}``.
    """
    images = sorted(
        [*img_dir.glob("*.png"), *img_dir.glob("*.jpg"), *img_dir.glob("*.jpeg")]
    )
    pairs: List[Dict] = []
    for im in images:
        lb = lbl_dir / f"{im.stem}.txt"
        if lb.exists():
            try:
                txt = lb.read_text().strip()
                pairs.append({"img": im, "lbl_in": lb, "has_obj": bool(txt)})
            except OSError:
                pairs.append({"img": im, "lbl_in": None, "has_obj": False})
        else:
            pairs.append({"img": im, "lbl_in": None, "has_obj": False})
    return pairs


# ---------------------------------------------------------------------------
# Copy & sanitise
# ---------------------------------------------------------------------------

def copy_split(split_name: str, records: List[Dict], out_dir: Path) -> None:
    """Copy images and labels into the working dataset directory.

    Missing or empty labels are written as empty ``.txt`` files (background).

    Args:
        split_name: One of ``"train"``, ``"val"``, ``"test"``.
        records:    Output of :func:`index_images` (subset for this split).
        out_dir:    Root output directory (``out_dir/images/split_name`` etc.).
    """
    img_dst = out_dir / "images" / split_name
    lbl_dst = out_dir / "labels" / split_name
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    copied, empty = 0, 0
    for rec in records:
        im: Path = rec["img"]
        lb_in: Optional[Path] = rec["lbl_in"]

        shutil.copy2(im, img_dst / im.name)
        dst_lbl = lbl_dst / f"{im.stem}.txt"

        if lb_in is not None:
            txt = lb_in.read_text().strip()
            if txt:
                shutil.copy2(lb_in, dst_lbl)
                copied += 1
            else:
                dst_lbl.write_text("")
                empty += 1
        else:
            dst_lbl.write_text("")
            empty += 1

    print(f"[{split_name:5s}] total={len(records):4d}  labelled={copied:4d}  background={empty:4d}")


def sanitize_labels(lbl_dir: Path, class_count: int = 2) -> Tuple[int, int]:
    """Sanitise YOLO labels in-place: clip coords to [0, 1], drop bad rows.

    Args:
        lbl_dir:     Directory of YOLO ``.txt`` labels to sanitise.
        class_count: Expected number of classes (rows with out-of-range class
                     IDs are dropped).

    Returns:
        Tuple ``(kept, dropped)`` box counts.
    """
    kept, dropped = 0, 0
    for f in lbl_dir.glob("*.txt"):
        lines_in = f.read_text().strip().splitlines() if f.exists() else []
        lines_out: List[str] = []
        for ln in lines_in:
            parts = ln.strip().split()
            if len(parts) != 5:
                dropped += 1
                continue
            try:
                cid = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
            except ValueError:
                dropped += 1
                continue
            if not (0 <= cid < class_count):
                dropped += 1
                continue
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            if w <= 0 or h <= 0:
                dropped += 1
                continue
            lines_out.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            kept += 1
        f.write_text("\n".join(lines_out))
    return kept, dropped


# ---------------------------------------------------------------------------
# CLAHE preprocessing
# ---------------------------------------------------------------------------

def apply_clahe(
    src_dir: Path,
    dst_dir: Path,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> int:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to images.

    UAV turbine blade imagery often has low local contrast in overcast conditions.
    CLAHE enhances fine surface texture which aids both dirt and damage detection.

    Args:
        src_dir:    Source image directory.
        dst_dir:    Destination directory (created if absent).
        clip_limit: CLAHE clip limit.
        tile_grid:  Grid size for local histogram equalisation.

    Returns:
        Number of images processed.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    count = 0
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for img_path in src_dir.glob(ext):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            # Apply CLAHE per channel in LAB space to preserve colour balance
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = clahe.apply(l_ch)
            enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
            cv2.imwrite(str(dst_dir / img_path.name), enhanced)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def dataset_stats(lbl_dir: Path, names: List[str]) -> Dict:
    """Compute per-class box counts from a YOLO label directory.

    Args:
        lbl_dir: Directory with YOLO ``.txt`` files.
        names:   Class name list (index → name).

    Returns:
        Dict with keys ``total_images``, ``labelled_images``,
        ``boxes_per_class``, ``total_boxes``.
    """
    counts: Dict[int, int] = {i: 0 for i in range(len(names))}
    total, labelled = 0, 0
    for f in lbl_dir.glob("*.txt"):
        total += 1
        lines = f.read_text().strip().splitlines()
        if lines:
            labelled += 1
        for ln in lines:
            parts = ln.strip().split()
            if parts:
                cid = int(float(parts[0]))
                counts[cid] = counts.get(cid, 0) + 1
    return {
        "total_images": total,
        "labelled_images": labelled,
        "boxes_per_class": {names[k]: v for k, v in counts.items()},
        "total_boxes": sum(counts.values()),
    }
