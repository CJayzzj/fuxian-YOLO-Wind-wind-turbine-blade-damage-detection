# Wind Turbine Blade Damage Detection — Improved YOLOv8 Baseline

机器学习论文复现 — YOLOv8 baseline 改进版（风力涡轮机叶片损伤检测）

---

## Overview

This project reproduces and improves a **YOLOv8-based baseline** for detecting
surface damage on wind turbine blades from UAV/drone imagery.

**Dataset** — *YOLO-annotated Wind Turbine Surface Damage* (Foster et al., IEEE IVMSP 2022):
- 586 × 371 px images from a NordTank turbine
- 2 classes: `dirt` (surface contamination) and `damage` (cracks / erosion)

---

## Improvements over the YOLOv8m Baseline

| Aspect | Baseline | Improved (`YOLOv8m-Wind`) |
|---|---|---|
| **Architecture** | YOLOv8m (stock) | YOLOv8m + **CBAM** attention in neck |
| **Attention** | None | Channel + Spatial attention (CBAM, ECCV 2018) |
| **LR schedule** | Linear decay | **Cosine** with 5-epoch warmup |
| **Augmentation** | Default mosaic | Mosaic + **MixUp** (p=0.15) + **Copy-Paste** (p=0.10) |
| **Label smoothing** | None | ε = 0.05 |
| **Training length** | 100 epochs | **150 epochs** with early stopping (patience=30) |
| **Preprocessing** | None | Optional **CLAHE** (contrast enhancement for UAV imagery) |
| **Evaluation** | Standard val | Standard + **TTA** (test-time augmentation) |

### CBAM (Convolutional Block Attention Module)

CBAM is inserted after the **SPPF** layer (backbone output) and after every
C2f block in the **PAN neck** (P3 / P4 / P5 feature maps).  It applies
sequential **channel** and **spatial** attention to focus on defect-relevant
features, which is critical for small blade cracks that occupy only a few
percent of the image area.

```
Feature map → Channel Attention → Spatial Attention → refined feature map
                  (avg+max pool + shared MLP + sigmoid)
                                     (7×7 conv on concat[avg,max] + sigmoid)
```

---

## Repository Structure

```
.
├── train.py                         # Improved training script
├── evaluate.py                      # Comprehensive evaluation script
├── requirements.txt
│
├── models/
│   ├── __init__.py                  # CBAM registration helper
│   ├── cbam.py                      # CBAM, ChannelAttention, SpatialAttention
│   └── yolov8m_wind.yaml            # Custom model YAML (YOLOv8m + CBAM)
│
├── configs/
│   ├── wind_turbine_data.yaml       # Dataset configuration
│   └── hyp_wind.yaml                # Training hyperparameters
│
├── utils/
│   ├── dataset.py                   # Splitting, label sanitisation, CLAHE
│   └── visualize.py                 # Bounding-box drawing, prediction grids
│
└── 5-3-yolo-annotated-wind-turbine-surface-damage.ipynb  # Kaggle notebook
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (Kaggle / local with dataset)

```bash
# Improved YOLOv8m-Wind (CBAM + better hyperparameters)
python train.py \
    --input-root /path/to/yolo-annotated-wind-turbines-586x371 \
    --workdir ./output \
    --epochs 150 \
    --batch 16

# Ablation: stock YOLOv8m baseline (no CBAM)
python train.py --no-cbam \
    --input-root /path/to/yolo-annotated-wind-turbines-586x371 \
    --workdir ./output_baseline
```

### 3. Evaluate a saved checkpoint

```bash
python evaluate.py \
    --weights output/windturbine_yolo_improved/yolov8m_wind_cbam/weights/best.pt \
    --data    output/windturbine_yolo_improved/data.yaml \
    --out-dir eval_results/
```

### 4. Kaggle notebook

Open `5-3-yolo-annotated-wind-turbine-surface-damage.ipynb` in Kaggle (GPU T4 × 2 recommended).  
The notebook mirrors the full pipeline including optional BLIP scene captioning and FLAN-T5
inspection report generation.

---

## Model Architecture (`models/yolov8m_wind.yaml`)

```
Backbone  →  [Conv×2] → C2f×3 → [Conv] → C2f×6 → [Conv] → C2f×6
              → [Conv] → C2f×3 → SPPF → CBAM(1024)
                                                  ↓
Head      →  Upsample → Concat(P4) → C2f(512) → CBAM(512)
              → Upsample → Concat(P3) → C2f(256) → CBAM(256)   ← P3 small
              → Conv↓ → Concat → C2f(512)                       ← P4 medium
              → Conv↓ → Concat → C2f(1024)                      ← P5 large
              → Detect([P3, P4, P5])
```

---

## Key Hyperparameters (`configs/hyp_wind.yaml`)

| Parameter | Value | Rationale |
|---|---|---|
| `lr0` | 0.01 | Standard SGD initial LR |
| `cos_lr` | True | Smooth convergence |
| `warmup_epochs` | 5 | Stabilises CBAM weights |
| `mixup` | 0.15 | Helps with inter-class variation |
| `copy_paste` | 0.10 | Augments small-object coverage |
| `label_smoothing` | 0.05 | Reduces annotation noise effects |
| `close_mosaic` | 10 | Fine-tune without mosaic at end |

---

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{foster2022drone,
  title={Drone Footage Wind Turbine Surface Damage Detection},
  author={Foster, Ashley and Best, Oscar and Gianni, Mario and Khan, Asiya
          and Collins, Kerry and Sharma, Sanjay},
  booktitle={IEEE IVMSP Workshop},
  year={2022}
}

@data{hd96prn3nc.2,
  author  = {SHIHAVUDDIN, A.S.M. and Chen, Xiang},
  title   = {DTU - Drone inspection images of wind turbine},
  year    = {2018},
  publisher = {Mendeley Data},
  version = {2},
  doi     = {10.17632/hd96prn3nc.2}
}
```
