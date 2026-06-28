# -*- coding: utf-8 -*-
"""
run_baseline.py
===============
Phase 4: Run zero-shot RIFE baseline on the satellite test set.

Pipeline
--------
  Clone RIFE  ->  Download weights  ->  Load model
  ->  Infer on data/test/ triplets  ->  Compute PSNR/SSIM/MSE
  ->  Save outputs/baseline/  ->  Generate GIF + MP4 + comparison PNGs
  ->  Save outputs/metrics/baseline_metrics.json

Usage (local / Colab)
---------------------
  python scripts/run_baseline.py
  python scripts/run_baseline.py --test-dir data/test --limit 20
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
RIFE_DIR    = ROOT / "models" / "rife"
WEIGHTS_DIR = RIFE_DIR / "train_log"
TEST_DIR    = ROOT / "data" / "test"
OUT_DIR     = ROOT / "outputs" / "baseline"
METRICS_DIR = ROOT / "outputs" / "metrics"
ANIM_DIR    = ROOT / "outputs" / "animations"

for d in [OUT_DIR, METRICS_DIR, ANIM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RIFE_REPO = "https://github.com/hzwer/Practical-RIFE.git"

# Google Drive file ID for Practical-RIFE v4.25 (avoids quota limit of older versions)
WEIGHTS_GDRIVE_ID = "1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg"


# ── setup RIFE ────────────────────────────────────────────────────────────────
def clone_rife():
    if (RIFE_DIR / "model").exists():
        print("  RIFE already cloned.")
        return
    print(f"  Cloning Practical-RIFE into {RIFE_DIR} ...")
    RIFE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", RIFE_REPO, str(RIFE_DIR)],
        check=True
    )
    print("  Done.")


def download_weights():
    if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.glob("*.pkl")):
        print("  Weights already present.")
        return
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Try gdown (available on Colab)
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={WEIGHTS_GDRIVE_ID}"
        zip_path = RIFE_DIR / "train_log.zip"
        print(f"  Downloading weights via gdown ...")
        gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
        shutil.unpack_archive(str(zip_path), str(RIFE_DIR))
        zip_path.unlink(missing_ok=True)
        print("  Weights extracted.")
    except Exception as e:
        print(f"  [WARN] gdown failed: {e}")
        print("  Manual download required:")
        print(f"    gdown {WEIGHTS_GDRIVE_ID} -O models/rife/train_log.zip")
        print("    unzip models/rife/train_log.zip -d models/rife/")
        sys.exit(1)


def load_rife_model(device):
    sys.path.insert(0, str(RIFE_DIR))
    try:
        from model.RIFE_HDv3 import Model
    except ImportError:
        from model.RIFE_HD import Model

    model = Model()
    model.load_model(str(RIFE_DIR), -1)
    model.eval()
    model.device()
    return model


# ── image helpers ─────────────────────────────────────────────────────────────
def load_img_tensor(path: Path, device):
    import torch
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    # Repeat grayscale -> 3-channel (RIFE expects RGB)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    t = t.repeat(1, 3, 1, 1)                               # [1,3,H,W]
    return t.to(device)


def tensor_to_uint8(t) -> np.ndarray:
    """Convert [1,3,H,W] or [1,1,H,W] tensor -> uint8 HxW numpy array."""
    arr = t.squeeze().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.mean(axis=0)   # RGB -> gray (they're identical for us)
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    pred_f = pred.astype(np.float64)
    gt_f   = gt.astype(np.float64)
    mse    = float(np.mean((pred_f - gt_f) ** 2))
    psnr   = float(peak_signal_noise_ratio(gt_f, pred_f, data_range=255.0))
    ssim   = float(structural_similarity(gt, pred, data_range=255))
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


# ── visualization ─────────────────────────────────────────────────────────────
def save_comparison(img0: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                    path: Path, metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d1117")
    fig.suptitle(
        f"PSNR={metrics['psnr']:.2f} dB   SSIM={metrics['ssim']:.4f}   MSE={metrics['mse']:.2f}",
        color="white", fontsize=11
    )
    titles = ["Input (frame0)", "RIFE Prediction", "Ground Truth"]
    imgs   = [img0, pred, gt]
    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, color="#c9d1d9", fontsize=9)
        ax.axis("off")
    for spine in [s for ax in axes for s in ax.spines.values()]:
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_animation(frames: list[np.ndarray], path: Path, fps: int = 4) -> None:
    """Save frames as GIF and (if imageio-ffmpeg available) MP4."""
    import imageio.v2 as imageio

    gif_path = path.with_suffix(".gif")
    imageio.mimsave(str(gif_path), frames, fps=fps)
    print(f"  GIF saved -> {gif_path}")

    try:
        import imageio_ffmpeg  # noqa
        mp4_path = path.with_suffix(".mp4")
        writer = imageio.get_writer(str(mp4_path), fps=fps)
        for f in frames:
            writer.append_data(np.stack([f, f, f], axis=-1))
        writer.close()
        print(f"  MP4 saved -> {mp4_path}")
    except Exception:
        pass


# ── inference loop ────────────────────────────────────────────────────────────
def run_inference(model, test_dir: Path, out_dir: Path,
                  limit: int, device) -> list[dict]:
    import torch

    triplets = sorted(test_dir.iterdir())
    if limit > 0:
        triplets = triplets[:limit]

    all_metrics = []
    anim_frames = []   # for GIF/MP4 (collect pred frames)

    print(f"\n  Running inference on {len(triplets)} triplets ...\n")

    for i, tri in enumerate(triplets):
        img0_path = tri / "img0.png"
        img1_path = tri / "img1.png"
        gt_path   = tri / "gt.png"

        if not (img0_path.exists() and img1_path.exists() and gt_path.exists()):
            continue

        with torch.no_grad():
            t0  = load_img_tensor(img0_path, device)
            t1  = load_img_tensor(img1_path, device)
            mid = model.inference(t0, t1)

        pred_arr = tensor_to_uint8(mid)
        gt_arr   = np.array(Image.open(gt_path).convert("L"))
        img0_arr = np.array(Image.open(img0_path).convert("L"))

        metrics = compute_metrics(pred_arr, gt_arr)
        metrics["triplet"] = tri.name
        all_metrics.append(metrics)

        # save predicted frame
        pred_out = out_dir / f"{tri.name}_pred.png"
        Image.fromarray(pred_arr, "L").save(pred_out)

        # save comparison panel
        cmp_out = out_dir / f"{tri.name}_comparison.png"
        save_comparison(img0_arr, pred_arr, gt_arr, cmp_out, metrics)

        anim_frames.append(pred_arr)

        if (i + 1) % 10 == 0 or (i + 1) == len(triplets):
            avg_psnr = np.mean([m["psnr"] for m in all_metrics])
            avg_ssim = np.mean([m["ssim"] for m in all_metrics])
            print(f"  [{i+1:>4}/{len(triplets)}]  "
                  f"PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}  "
                  f"| avg PSNR={avg_psnr:.2f}  avg SSIM={avg_ssim:.4f}")

    return all_metrics, anim_frames


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", type=Path, default=TEST_DIR)
    p.add_argument("--out-dir",  type=Path, default=OUT_DIR)
    p.add_argument("--limit",    type=int,  default=0,
                   help="Max triplets to evaluate (0 = all)")
    p.add_argument("--skip-setup", action="store_true",
                   help="Skip clone+download if already done")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*62)
    print("  Phase 4 -- RIFE Baseline Inference")
    print("="*62)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── setup ──────────────────────────────────────────────────────────────
    if not args.skip_setup:
        print("\n[1/4] Setting up RIFE ...")
        clone_rife()
        download_weights()
    else:
        print("\n[1/4] Skipping setup (--skip-setup).")

    print("\n[2/4] Loading RIFE model ...")
    model = load_rife_model(device)
    print("  Model loaded.")

    # ── inference ──────────────────────────────────────────────────────────
    print("\n[3/4] Running inference ...")
    all_metrics, anim_frames = run_inference(
        model, args.test_dir, args.out_dir, args.limit, device
    )

    # ── save metrics ───────────────────────────────────────────────────────
    print("\n[4/4] Saving metrics and animations ...")

    summary = {
        "model":    "RIFE_HDv3_baseline",
        "n_samples": len(all_metrics),
        "avg_psnr": float(np.mean([m["psnr"] for m in all_metrics])),
        "avg_ssim": float(np.mean([m["ssim"] for m in all_metrics])),
        "avg_mse":  float(np.mean([m["mse"]  for m in all_metrics])),
        "std_psnr": float(np.std( [m["psnr"] for m in all_metrics])),
        "std_ssim": float(np.std( [m["ssim"] for m in all_metrics])),
        "per_sample": all_metrics,
    }

    metrics_path = METRICS_DIR / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Metrics -> {metrics_path}")

    # animation
    if anim_frames:
        make_animation(anim_frames[:30], ANIM_DIR / "baseline_prediction")

    print(f"""
{'='*62}
  BASELINE RESULTS
  Samples  : {summary['n_samples']}
  Avg PSNR : {summary['avg_psnr']:.3f} dB
  Avg SSIM : {summary['avg_ssim']:.4f}
  Avg MSE  : {summary['avg_mse']:.2f}
  Outputs  : {args.out_dir}
  Metrics  : {metrics_path}
{'='*62}
""")


if __name__ == "__main__":
    main()
