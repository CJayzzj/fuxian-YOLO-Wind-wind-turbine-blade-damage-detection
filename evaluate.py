"""evaluate.py – Comprehensive evaluation for wind turbine blade damage detection.

Loads a trained checkpoint and produces:
  • Val / test mAP (standard + TTA)
  • Per-class AP table
  • Confusion matrix
  • Per-image severity scores and a risk-level histogram
  • Prediction thumbnails grid
  • Inference FPS benchmark

Usage
-----
  python evaluate.py --weights /path/to/best.pt --data /path/to/data.yaml
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wind turbine damage detection – evaluation")
    p.add_argument("--weights", required=True, help="Path to best.pt checkpoint")
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.65)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out-dir", default="eval_output",
                   help="Directory for evaluation artifacts")
    p.add_argument("--no-cbam", action="store_true",
                   help="Set if the checkpoint was trained WITHOUT the custom CBAM model")
    p.add_argument("--grid-samples", type=int, default=32,
                   help="Number of test images to include in the prediction grid")
    return p.parse_args()


def load_model(weights: str, use_cbam: bool = True):
    if use_cbam:
        from models import register_custom_modules
        register_custom_modules()
    from ultralytics import YOLO
    return YOLO(weights)


def run_val(model, data: str, split: str, imgsz: int, batch: int,
            augment: bool = False, out_dir: str = "eval_output",
            tag: str = "eval") -> dict:
    m = model.val(
        split=split,
        data=data,
        imgsz=imgsz,
        batch=batch,
        augment=augment,
        save_json=True,
        plots=True,
        project=out_dir,
        name=tag,
    )
    return {
        "tag": tag,
        "mAP50_95": float(m.box.map),
        "mAP50": float(m.box.map50),
        "mAP75": float(m.box.map75),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "maps_per_class": getattr(m.box, "maps", None),
    }


def risk_level(score: float) -> str:
    if score > 0.05:
        return "high"
    if score >= 0.01:
        return "moderate"
    return "low"


def build_severity_report(
    model,
    test_dir: Path,
    imgsz: int,
    conf: float,
    names: list,
    img_h: int = 371,
    img_w: int = 586,
) -> pd.DataFrame:
    from tqdm import tqdm

    records = []
    for im in tqdm(sorted(test_dir.glob("*.*")), desc="Scoring"):
        results = model.predict(source=str(im), imgsz=imgsz, conf=conf, verbose=False)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item())
                c = float(b.conf.item())
                x1, y1, x2, y2 = b.xyxy.cpu().numpy().tolist()[0]
                dets.append({"cls": names[cls_id], "cls_id": cls_id, "conf": c,
                             "bbox_xyxy": [x1, y1, x2, y2]})
        sev = sum(
            (2.0 if d["cls"] == "damage" else 1.0)
            * max(0, d["bbox_xyxy"][2] - d["bbox_xyxy"][0])
            * max(0, d["bbox_xyxy"][3] - d["bbox_xyxy"][1])
            / (img_w * img_h)
            * d["conf"]
            for d in dets
        )
        records.append({
            "image": im.name,
            "num_dirt": sum(1 for d in dets if d["cls"] == "dirt"),
            "num_damage": sum(1 for d in dets if d["cls"] == "damage"),
            "severity_score": round(sev, 6),
            "risk_level": risk_level(sev),
            "detections_json": json.dumps(dets),
        })
    return pd.DataFrame(records)


def save_prediction_grid(
    model,
    test_imgs: list,
    out_path: Path,
    imgsz: int,
    conf: float,
    n: int = 32,
) -> None:
    """Save sample predictions as an annotated PNG grid."""
    try:
        import cv2
        from utils.visualize import draw_detections, save_prediction_grid as _grid

        imgs = test_imgs[:n]
        dets_map = {}
        for im in imgs:
            results = model.predict(source=str(im), imgsz=imgsz, conf=conf, verbose=False)
            dets = []
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls_id = int(b.cls.item())
                    dets.append({
                        "cls": ["dirt", "damage"][cls_id],
                        "cls_id": cls_id,
                        "conf": float(b.conf.item()),
                        "bbox_xyxy": b.xyxy.cpu().numpy().tolist()[0],
                    })
            dets_map[im.name] = dets
        _grid(imgs, dets_map, out_path, cols=4, conf_threshold=conf)
    except Exception as exc:
        print(f"[Grid] Skipped prediction grid: {exc}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights, use_cbam=not args.no_cbam)

    # --- Validation & test metrics ---
    print("\n[1/4] Running validation …")
    val_row = run_val(model, args.data, "val", args.imgsz, args.batch,
                      out_dir=str(out_dir), tag="val_std")

    print("\n[2/4] Running test evaluation (standard) …")
    test_row = run_val(model, args.data, "test", args.imgsz, args.batch,
                       out_dir=str(out_dir), tag="test_std")

    print("\n[2b] Running test evaluation (TTA) …")
    test_tta = run_val(model, args.data, "test", args.imgsz, args.batch,
                       augment=True, out_dir=str(out_dir), tag="test_tta")

    rows = []
    for row in (val_row, test_row, test_tta):
        rows.append({k: v for k, v in row.items() if k != "maps_per_class"})
    df_metrics = pd.DataFrame(rows)
    print("\n[Metrics]\n", df_metrics.to_string(index=False))
    df_metrics.to_csv(out_dir / "summary_metrics.csv", index=False)

    # Per-class AP
    names = ["dirt", "damage"]
    per_cls = test_row.get("maps_per_class")
    if per_cls is not None and len(per_cls) == 2:
        pd.DataFrame({"class": names, "AP50_95": list(per_cls)}).to_csv(
            out_dir / "per_class_ap.csv", index=False
        )

    # --- Severity scoring ---
    print("\n[3/4] Building severity report …")
    import yaml
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    data_root = Path(data_cfg.get("path", "."))
    test_dir = data_root / data_cfg.get("test", "images/test")
    if not test_dir.exists():
        test_dir = data_root.parent / data_cfg.get("test", "images/test")

    sev_df = build_severity_report(model, test_dir, args.imgsz, args.conf, names)
    sev_df.to_csv(out_dir / "detections_test.csv", index=False)

    risk_counts = sev_df["risk_level"].value_counts().to_dict()
    pd.DataFrame([{
        "images_tested":      len(sev_df),
        "images_with_damage": int((sev_df["num_damage"] > 0).sum()),
        "risk_high":          risk_counts.get("high", 0),
        "risk_moderate":      risk_counts.get("moderate", 0),
        "risk_low":           risk_counts.get("low", 0),
        "mean_severity":      float(sev_df["severity_score"].mean()),
        "p95_severity":       float(sev_df["severity_score"].quantile(0.95)),
    }]).to_csv(out_dir / "executive_summary.csv", index=False)

    # --- Prediction grid ---
    print("\n[4/4] Saving prediction grid …")
    test_imgs = sorted(test_dir.glob("*.*"))
    save_prediction_grid(
        model, test_imgs,
        out_dir / "prediction_grid.png",
        args.imgsz, args.conf, n=args.grid_samples,
    )

    # --- FPS ---
    t0 = time.time()
    n = min(200, len(test_imgs))
    for im in test_imgs[:n]:
        model.predict(source=str(im), imgsz=args.imgsz, conf=args.conf, verbose=False)
    fps = n / (time.time() - t0)
    pd.DataFrame([{"fps": fps, "images": n, "imgsz": args.imgsz}]).to_csv(
        out_dir / "inference_fps.csv", index=False
    )
    print(f"[FPS] {fps:.2f} FPS on {n} images")

    print(f"\nAll evaluation artifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
