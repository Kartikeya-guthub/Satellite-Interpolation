# 🛰️ Satellite Interpolation — Frequency-Aware

Temporal interpolation of GOES-19 satellite imagery using frequency-aware video frame interpolation (RIFE fine-tuned on satellite sequences).

## Project Structure

```
satellite-interpolation/
├── app/                    # Streamlit demo app
├── configs/                # YAML config files for training & inference
├── data/
│   ├── raw/                # Downloaded GOES-19 NetCDF + PNG frames
│   ├── processed/          # Normalized, cropped tiles
│   ├── train/              # Training triplets (frame0, frame1, gt)
│   ├── validation/         # Validation sequences
│   └── test/               # Hold-out test sequences
├── models/
│   ├── rife/               # RIFE model source
│   ├── checkpoints/        # Training checkpoints
│   └── finetuned/          # Final fine-tuned weights
├── notebooks/              # Colab notebooks (one per phase)
├── outputs/
│   ├── baseline/           # RIFE zero-shot results
│   ├── finetuned/          # Fine-tuned model results
│   ├── metrics/            # JSON / CSV metric logs
│   ├── heatmaps/           # Error / difference heatmaps
│   ├── animations/         # GIF / MP4 animations
│   └── comparisons/        # Side-by-side visual comparisons
├── scripts/                # Standalone pipeline scripts
├── utils/                  # Shared utility modules
├── requirements.txt
└── README.md
```

## Colab Notebooks

| Notebook | Description |
|---|---|
| `notebooks/Phase1_Setup.ipynb` | GPU setup, Drive mount, package install |
| `notebooks/Phase2_DataCollection.ipynb` | GOES-19 download & inspection |
| `notebooks/Phase3_Preprocessing.ipynb` | Tile generation & train/val splits |
| `notebooks/Phase4_Baseline.ipynb` | RIFE zero-shot baseline evaluation |
| `notebooks/Phase5_Finetune.ipynb` | Fine-tuning on satellite data |
| `notebooks/Phase6_Evaluation.ipynb` | Metrics, heatmaps, animations |

## Google Drive Structure

```
Drive/SatelliteInterpolation/
├── data/        # Mirrors local data/ (large files)
├── outputs/     # All generated results
├── checkpoints/ # Saved model weights
├── logs/        # Training logs
└── models/      # Model source files
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download GOES-19 data
python scripts/download_data.py

# (Colab) Open Phase1_Setup.ipynb and run all cells
```

## Pipeline Scripts

| Script | Description |
|---|---|
| `scripts/download_data.py` | Download raw GOES-19 frames via AWS S3 |
| `scripts/preprocess.py` | Preprocess frames → train/val/test splits |
| `scripts/run_baseline.py` | RIFE zero-shot baseline inference |
| `scripts/train.py` | Fine-tune RIFE on satellite data |
| `scripts/evaluate.py` | Compute PSNR, SSIM, tOF metrics |
| `scripts/export_assets.py` | Generate heatmaps & animations |
