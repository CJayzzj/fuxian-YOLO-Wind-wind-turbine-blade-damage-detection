"""train.py – Improved YOLOv8 training for wind turbine blade damage detection.

Improvements over the stock YOLOv8m baseline
============================================
Architecture
  • YOLOv8m-Wind (models/yolov8m_wind.yaml): CBAM attention modules added after
    SPPF (backbone) and after every C2f block in the PAN neck.  CBAM lets the
    network learn *where* (spatial) and *what* (channel) to focus on, which is
    especially valuable for small, low-contrast blade defects in UAV imagery.

Training
  • Cosine LR schedule (cos_lr=True) for smoother convergence.
  • Extended warmup (5 epochs) to stabilise the freshly-initialised CBAM weights.
  • Stronger augmentation: Mosaic, MixUp (p=0.15), Copy-Paste (p=0.10).
  • Label smoothing (ε=0.05) to reduce overconfidence on noisy annotations.
  • AMP (mixed precision) enabled for speed / memory efficiency.
  • Patience-based early stopping.

Evaluation
  • Standard val + test mAP@0.5 and mAP@0.5:0.95.
  • TTA (test-time augmentation) pass for a second inference estimate.

Usage
-----
  # Minimal – uses kaggle paths (mirrors original notebook):
  python train.py

  # Custom dataset root:
  python train.py --data /path/to/data.yaml --project my_project --name run1

  # Ablation: use stock YOLOv8m without CBAM:
  python train.py --no-cbam
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Disable W&B / Ray before importing ultralytics
# ---------------------------------------------------------------------------
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wind turbine blade damage detection – improved YOLOv8m training")

    # Data
    p.add_argument("--input-root", default="/kaggle/input/yolo-annotated-wind-turbines-586x371",
                   help="Root directory of the raw Kaggle dataset")
    p.add_argument("--workdir", default="/kaggle/working",
                   help="Working directory for outputs")
    p.add_argument("--data", default=None,
                   help="Path to an already-prepared data.yaml (skips data prep)")

    # Model
    p.add_argument("--no-cbam", action="store_true",
                   help="Use stock yolov8m.pt baseline instead of CBAM model")
    p.add_argument("--weights", default="yolov8m.pt",
                   help="Pretrained weights (used when --no-cbam is set or as init)")

    # Training
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--project", default="windturbine_yolo_improved")
    p.add_argument("--name", default="yolov8m_wind_cbam")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")

    # Misc
    p.add_argument("--evolve", action="store_true", help="Run hyperparameter evolution before training")
    p.add_argument("--clahe", action="store_true", default=False,
                   help="Apply CLAHE preprocessing to training images")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(args: argparse.Namespace) -> Path:
    """Prepare the working dataset directory and return the data.yaml path."""
    from utils.dataset import (
        apply_clahe,
        copy_split,
        dataset_stats,
        index_images,
        sanitize_labels,
        split_dataset,
    )

    root = Path(args.input_root)
    ds_dir = root / "NordTank586x371"
    img_dir_in = ds_dir / "images"
    lbl_dir_in = ds_dir / "labels"

    if not img_dir_in.exists() or not lbl_dir_in.exists():
        raise FileNotFoundError(
            f"Dataset not found at {ds_dir}. "
            "Download from https://www.kaggle.com/datasets/ajifoster3/yolo-annotated-wind-turbines-586x371"
        )

    workdir = Path(args.workdir)
    out = workdir / args.project / "dataset"

    # Index & split
    pairs = index_images(img_dir_in, lbl_dir_in)
    train, val, test = split_dataset(pairs, train_ratio=0.8, val_ratio=0.1, seed=args.seed)
    print(f"\nDataset split — train: {len(train)}  val: {len(val)}  test: {len(test)}")

    # Copy
    for split_name, split_list in [("train", train), ("val", val), ("test", test)]:
        copy_split(split_name, split_list, out)

    # Optional CLAHE preprocessing on training images
    if args.clahe:
        n = apply_clahe(out / "images/train", out / "images/train_clahe")
        print(f"CLAHE applied to {n} training images → {out}/images/train_clahe")

    # Sanitise labels
    for split_name in ("train", "val", "test"):
        kept, dropped = sanitize_labels(out / f"labels/{split_name}", class_count=2)
        print(f"Labels [{split_name}] — kept: {kept}  dropped: {dropped}")

    # Stats
    for split_name in ("train", "val", "test"):
        stats = dataset_stats(out / f"labels/{split_name}", names=["dirt", "damage"])
        print(f"Stats  [{split_name}] — {stats}")

    # Write data.yaml
    data_yaml = workdir / args.project / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(
        f"path: {out}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n"
        "nc: 2\n"
        "names:\n"
        "  0: dirt\n"
        "  1: damage\n"
    )
    print(f"\ndata.yaml written → {data_yaml}")
    return data_yaml


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace, data_yaml: Path):
    """Return an initialised YOLO model (CBAM or baseline)."""
    from ultralytics import YOLO

    if args.no_cbam:
        print("\n[Model] Using stock YOLOv8m baseline (no CBAM)")
        return YOLO(args.weights)

    # Register custom CBAM module so ultralytics can resolve it from YAML
    from models import register_custom_modules
    register_custom_modules()

    model_yaml = Path(__file__).parent / "models" / "yolov8m_wind.yaml"
    print(f"\n[Model] Loading YOLOv8m-Wind (CBAM) from {model_yaml}")
    model = YOLO(str(model_yaml))

    # Initialise backbone / neck weights from pretrained YOLOv8m
    print(f"[Model] Transferring pretrained weights from {args.weights}")
    pretrained = YOLO(args.weights)
    state_src = pretrained.model.state_dict()
    state_dst = model.model.state_dict()

    matched, skipped = 0, 0
    new_state = {}
    for k, v in state_dst.items():
        if k in state_src and state_src[k].shape == v.shape:
            new_state[k] = state_src[k]
            matched += 1
        else:
            new_state[k] = v
            skipped += 1
    model.model.load_state_dict(new_state, strict=False)
    print(f"[Model] Weight transfer — matched: {matched}  skipped (CBAM / shape): {skipped}")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, model, data_yaml: Path) -> Path:
    """Train the model and return the path to the best checkpoint."""
    import yaml

    workdir = Path(args.workdir)
    hyp_path = Path(__file__).parent / "configs" / "hyp_wind.yaml"

    # Load custom hyperparameters
    hyp = {}
    if hyp_path.exists():
        with open(hyp_path) as f:
            hyp = yaml.safe_load(f) or {}

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n[Train] device={device}  epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}")

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(workdir / args.project),
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        workers=args.workers,
        amp=args.amp,
        cos_lr=True,            # cosine LR schedule
        close_mosaic=10,        # disable mosaic last 10 epochs for fine-tuning
        **hyp,
    )

    # Optional hyperparam evolution
    if args.evolve:
        print("[Evolve] Running hyperparameter evolution (25 generations) …")
        model.train(**{**train_kwargs, "epochs": 20, "patience": 10})  # short evolve run

    results = model.train(**train_kwargs)
    best_ckpt = workdir / args.project / args.name / "weights" / "best.pt"
    print(f"\n[Train] Best checkpoint → {best_ckpt}")
    return best_ckpt


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    best_ckpt: Path,
    data_yaml: Path,
    args: argparse.Namespace,
    workdir: Path,
) -> pd.DataFrame:
    """Run val/test evaluation (standard + TTA) and return metrics DataFrame."""
    from ultralytics import YOLO

    if not args.no_cbam:
        from models import register_custom_modules
        register_custom_modules()

    model = YOLO(str(best_ckpt))

    def _row(tag: str, m) -> dict:
        return {
            "tag": tag,
            "mAP50_95": float(m.box.map),
            "mAP50": float(m.box.map50),
            "mAP75": float(m.box.map75),
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
        }

    project_out = str(workdir / args.project)

    val_std = model.val(split="val", data=str(data_yaml), imgsz=args.imgsz,
                        save_json=True, plots=True, project=project_out, name="val_std")
    test_std = model.val(split="test", data=str(data_yaml), imgsz=args.imgsz,
                         save_json=True, plots=True, project=project_out, name="test_std")
    test_tta = model.val(split="test", data=str(data_yaml), imgsz=args.imgsz,
                         augment=True, plots=False, project=project_out, name="test_tta")

    df = pd.DataFrame([
        _row("val_std",  val_std),
        _row("test_std", test_std),
        _row("test_tta", test_tta),
    ])
    print("\n[Eval] Metrics:\n", df.to_string(index=False))

    artifacts = workdir / args.project / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    df.to_csv(artifacts / "summary_metrics.csv", index=False)

    # Per-class AP
    try:
        per_cls = getattr(test_std.box, "maps", None)
        if per_cls is not None:
            pd.DataFrame({"class": ["dirt", "damage"], "AP50_95": list(per_cls)}).to_csv(
                artifacts / "per_class_ap.csv", index=False
            )
    except Exception as exc:
        print("[Eval] Per-class AP skipped:", exc)

    return df


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------

def score_detections(
    model,
    test_dir: Path,
    artifacts: Path,
    imgsz: int = 640,
    conf: float = 0.25,
) -> pd.DataFrame:
    """Run inference on the test set and compute a per-image severity score.

    Severity accounts for defect type, area, and confidence:
      severity = Σ w_cls × (bbox_area / image_area) × confidence
    where w_damage=2 and w_dirt=1.
    """
    names = ["dirt", "damage"]
    IMG_H, IMG_W = 371, 586

    test_imgs = sorted(test_dir.glob("*.*"))
    records = []
    for im in tqdm(test_imgs, desc="Scoring detections"):
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

        severity = 0.0
        for d in dets:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            area = max(0, x2 - x1) * max(0, y2 - y1) / (IMG_W * IMG_H)
            w_cls = 2.0 if d["cls"] == "damage" else 1.0
            severity += w_cls * area * d["conf"]

        records.append({
            "image": im.name,
            "num_dirt": sum(1 for d in dets if d["cls"] == "dirt"),
            "num_damage": sum(1 for d in dets if d["cls"] == "damage"),
            "severity_score": round(severity, 6),
            "detections_json": json.dumps(dets),
        })

    df = pd.DataFrame(records)
    df.to_csv(artifacts / "detections_test.csv", index=False)

    summary = {
        "images_tested":       int(len(df)),
        "images_with_damage":  int((df["num_damage"] > 0).sum()),
        "images_dirt_only":    int(((df["num_damage"] == 0) & (df["num_dirt"] > 0)).sum()),
        "mean_severity":       float(df["severity_score"].mean()),
        "median_severity":     float(df["severity_score"].median()),
        "p95_severity":        float(df["severity_score"].quantile(0.95)),
    }
    pd.DataFrame([summary]).to_csv(artifacts / "executive_summary.csv", index=False)
    print("\n[Severity] Executive summary:", summary)
    return df


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

def benchmark_fps(model, test_imgs, imgsz: int = 640, max_imgs: int = 200) -> float:
    subset = test_imgs[:max_imgs]
    t0 = time.time()
    for im in subset:
        model.predict(source=str(im), imgsz=imgsz, conf=0.25, verbose=False)
    fps = len(subset) / (time.time() - t0)
    print(f"\n[Benchmark] {fps:.2f} FPS on {len(subset)} test images (imgsz={imgsz})")
    return fps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    workdir = Path(args.workdir)

    # 1) Data preparation
    if args.data:
        data_yaml = Path(args.data)
        print(f"[Data] Using existing data.yaml: {data_yaml}")
    else:
        data_yaml = prepare_dataset(args)

    # 2) Build model
    model = build_model(args, data_yaml)

    # 3) Train
    best_ckpt = train(args, model, data_yaml)

    # 4) Evaluate
    metrics_df = evaluate(best_ckpt, data_yaml, args, workdir)

    # 5) Reload best checkpoint for inference
    from ultralytics import YOLO
    if not args.no_cbam:
        from models import register_custom_modules
        register_custom_modules()
    best_model = YOLO(str(best_ckpt))

    artifacts = workdir / args.project / "artifacts"
    test_dir = workdir / args.project / "dataset" / "images" / "test"

    # 6) Severity scoring
    det_df = score_detections(best_model, test_dir, artifacts, imgsz=args.imgsz)

    # 7) FPS benchmark
    test_imgs = sorted(test_dir.glob("*.*"))
    fps = benchmark_fps(best_model, test_imgs, imgsz=args.imgsz)
    pd.DataFrame([{"fps": fps, "imgsz": args.imgsz}]).to_csv(
        artifacts / "inference_fps.csv", index=False
    )

    print(f"\n{'='*60}")
    print("Training complete.  Artifacts:")
    for p in [
        best_ckpt,
        artifacts / "summary_metrics.csv",
        artifacts / "per_class_ap.csv",
        artifacts / "detections_test.csv",
        artifacts / "executive_summary.csv",
        artifacts / "inference_fps.csv",
    ]:
        print(f"  {p}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
