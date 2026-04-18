"""Visualisation utilities for wind turbine blade damage detection."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Colour palette: index → BGR colour
PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 180, 255),    # dirt  – orange
    1: (0, 0, 220),      # damage – red
}

CLASS_NAMES = {0: "dirt", 1: "damage"}


def draw_detections(
    img: np.ndarray,
    detections: List[Dict],
    conf_threshold: float = 0.25,
    line_thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        img:            BGR image array.
        detections:     List of dicts with keys ``cls``, ``conf``,
                        ``bbox_xyxy`` (absolute pixel coords [x1,y1,x2,y2]).
        conf_threshold: Draw only boxes above this confidence.
        line_thickness: Rectangle line thickness.
        font_scale:     Label font scale.

    Returns:
        Annotated image (copy, original unchanged).
    """
    out = img.copy()
    for det in detections:
        if det["conf"] < conf_threshold:
            continue
        cls_id = det.get("cls_id", next(
            (k for k, v in CLASS_NAMES.items() if v == det["cls"]), 0
        ))
        color = PALETTE.get(cls_id, (200, 200, 200))
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)

        label = f"{det['cls']} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            out, label, (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
        )
    return out


def severity_bar(
    img: np.ndarray,
    severity: float,
    max_severity: float = 0.10,
    bar_height: int = 20,
) -> np.ndarray:
    """Append a severity heat-bar below the image.

    Args:
        img:          Input BGR image.
        severity:     Severity score in [0, ∞).
        max_severity: Score considered 'maximum' (saturates bar).
        bar_height:   Pixel height of the bar strip.

    Returns:
        Image with severity bar appended at the bottom.
    """
    h, w = img.shape[:2]
    bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    ratio = min(severity / max(max_severity, 1e-9), 1.0)
    fill_w = int(ratio * w)

    # Gradient: green → yellow → red
    for x in range(fill_w):
        t = x / max(w - 1, 1)
        r = int(255 * min(t * 2, 1.0))
        g = int(255 * min((1 - t) * 2, 1.0))
        bar[:, x] = (0, g, r)

    label = f"severity: {severity:.4f}"
    cv2.putText(
        bar, label, (4, bar_height - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
    )
    return np.vstack([img, bar])


def save_prediction_grid(
    image_paths: List[Path],
    detections_map: Dict[str, List[Dict]],
    out_path: Path,
    cols: int = 4,
    thumb_size: Tuple[int, int] = (293, 186),
    conf_threshold: float = 0.25,
) -> None:
    """Save a grid of prediction thumbnails to a single PNG.

    Args:
        image_paths:     Ordered list of image paths to include.
        detections_map:  Mapping ``image_name → detection list``.
        out_path:        Output PNG file path.
        cols:            Grid columns.
        thumb_size:      ``(width, height)`` of each thumbnail.
        conf_threshold:  Minimum confidence for drawn boxes.
    """
    thumbs: List[np.ndarray] = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            img = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
        dets = detections_map.get(p.name, [])
        annotated = draw_detections(img, dets, conf_threshold)
        annotated = cv2.resize(annotated, thumb_size)
        thumbs.append(annotated)

    rows_needed = (len(thumbs) + cols - 1) // cols
    # Pad to full grid
    while len(thumbs) < rows_needed * cols:
        thumbs.append(np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8))

    row_imgs = [
        np.hstack(thumbs[r * cols : (r + 1) * cols]) for r in range(rows_needed)
    ]
    grid = np.vstack(row_imgs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"Prediction grid saved → {out_path}")
