# -*- coding: utf-8 -*-
"""
evaluate.py
===========
Phase 6: Comprehensive Evaluation & Asset Generation

Generates all required assets for the interactive dashboard:
- Computes SSIM, PSNR, MSE for Baseline vs Fine-tuned on data/test/
- Generates metrics.csv and metrics.json
- Generates difference heatmaps
- Generates comparison images
- Generates GIF animations
"""

import argparse
import json
import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from PIL import Image
import cv2

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
RIFE_DIR     = ROOT / "models" / "rife"
WEIGHTS_DIR  = RIFE_DIR / "train_log"
FINETUNE_DIR = ROOT / "models" / "finetuned"
TEST_DIR     = ROOT / "data" / "test"
OUT_DIR      = ROOT / "outputs" / "evaluation"

for d in ["metrics", "heatmaps", "comparisons", "animations"]:
    (OUT_DIR / d).mkdir(parents=True, exist_ok=True)


def load_rife_model(weights_path: Path, device):
    sys.path.insert(0, str(RIFE_DIR))
    try:
        from train_log.RIFE_HDv3 import Model
    except ImportError:
        try:
            from train_log.RIFE_HD import Model
        except ImportError:
            from train_log.RIFE_HDv2 import Model

    model = Model()
    model.load_model(str(weights_path.parent), -1)

    import torch
    if weights_path.name != "flownet.pkl" and weights_path.exists():
        print(f"  Loading fine-tuned weights: {weights_path.name}")
        ckpt = torch.load(weights_path, map_location=device)
        model.flownet.load_state_dict(ckpt["state_dict"])

    model.eval()
    model.device()
    return model


def load_img_tensor(path: Path, device):
    import torch
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = t.repeat(1, 3, 1, 1)
    return t.to(device)


def tensor_to_uint8(t) -> np.ndarray:
    arr = t.squeeze().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    pred_f = pred.astype(np.float64)
    gt_f   = gt.astype(np.float64)
    mse    = float(np.mean((pred_f - gt_f) ** 2))
    psnr   = float(peak_signal_noise_ratio(gt_f, pred_f, data_range=255.0))
    ssim   = float(structural_similarity(gt, pred, data_range=255))
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


def generate_heatmap(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    diff = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
    # Normalize diff for visualization (cap at ~50 intensity difference)
    diff = np.clip((diff / 50.0) * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def save_comparison_image(img0, gt, base_pred, ft_pred, heat_base, heat_ft, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#0d1117")
    fig.suptitle("Evaluation Comparison", color="white", fontsize=16)

    titles = [
        "Input Frame 0", "Baseline RIFE", "Difference (Baseline vs GT)",
        "Ground Truth",  "Fine-Tuned RIFE", "Difference (Fine-Tuned vs GT)"
    ]
    imgs = [
        img0, base_pred, heat_base,
        gt,   ft_pred,   heat_ft
    ]

    for i, (ax, title, img) in enumerate(zip(axes.flat, titles, imgs)):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title, color="#c9d1d9", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_animation(frames: list[np.ndarray], path: Path, fps: int = 4):
    import imageio.v2 as imageio
    gif_path = path.with_suffix(".gif")
    imageio.mimsave(str(gif_path), frames, fps=fps)


def evaluate(test_dir: Path, limit: int, device):
    import torch

    base_weights = WEIGHTS_DIR / "flownet.pkl"
    ft_weights   = FINETUNE_DIR / "frequency_rife.pth"

    print("\n[1/4] Loading models...")
    model_base = load_rife_model(base_weights, device)
    
    if ft_weights.exists():
        model_ft = load_rife_model(ft_weights, device)
    else:
        print("[WARN] Fine-tuned model not found. Using baseline for both to avoid crashing.")
        model_ft = model_base

    triplets = sorted(test_dir.iterdir())
    if limit > 0:
        triplets = triplets[:limit]

    print(f"\n[2/4] Evaluating {len(triplets)} triplets...")

    results = []
    base_anim_frames = []
    ft_anim_frames   = []

    for i, tri in enumerate(triplets):
        img0_path = tri / "img0.png"
        img1_path = tri / "img1.png"
        gt_path   = tri / "gt.png"

        if not (img0_path.exists() and img1_path.exists() and gt_path.exists()):
            continue

        gt_arr   = np.array(Image.open(gt_path).convert("L"))
        img0_arr = np.array(Image.open(img0_path).convert("L"))

        with torch.no_grad():
            t0 = load_img_tensor(img0_path, device)
            t1 = load_img_tensor(img1_path, device)
            
            pred_base = tensor_to_uint8(model_base.inference(t0, t1))
            pred_ft   = tensor_to_uint8(model_ft.inference(t0, t1))

        # Metrics
        m_base = compute_metrics(pred_base, gt_arr)
        m_ft   = compute_metrics(pred_ft, gt_arr)

        row = {
            "frame": tri.name,
            "baseline_psnr": m_base["psnr"],
            "baseline_ssim": m_base["ssim"],
            "baseline_mse":  m_base["mse"],
            "finetuned_psnr": m_ft["psnr"],
            "finetuned_ssim": m_ft["ssim"],
            "finetuned_mse":  m_ft["mse"],
        }
        results.append(row)

        # Heatmaps
        heat_base = generate_heatmap(pred_base, gt_arr)
        heat_ft   = generate_heatmap(pred_ft, gt_arr)
        
        # Save individual images for dashboard
        Image.fromarray(img0_arr, "L").save(OUT_DIR / "comparisons" / f"{tri.name}_input.png")
        Image.fromarray(gt_arr, "L").save(OUT_DIR / "comparisons" / f"{tri.name}_gt.png")
        Image.fromarray(pred_base, "L").save(OUT_DIR / "comparisons" / f"{tri.name}_pred_baseline.png")
        Image.fromarray(pred_ft, "L").save(OUT_DIR / "comparisons" / f"{tri.name}_pred_finetuned.png")
        Image.fromarray(heat_base).save(OUT_DIR / "heatmaps" / f"{tri.name}_heat_baseline.png")
        Image.fromarray(heat_ft).save(OUT_DIR / "heatmaps" / f"{tri.name}_heat_finetuned.png")

        # Save composite image
        cmp_path = OUT_DIR / "comparisons" / f"{tri.name}_panel.png"
        save_comparison_image(img0_arr, gt_arr, pred_base, pred_ft, heat_base, heat_ft, cmp_path)

        base_anim_frames.append(pred_base)
        ft_anim_frames.append(pred_ft)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(triplets)}")

    print("\n[3/4] Exporting Metrics & Data...")
    df = pd.DataFrame(results)
    
    csv_path = OUT_DIR / "metrics" / "evaluation_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    summary = {
        "baseline": {
            "psnr": df["baseline_psnr"].mean(),
            "ssim": df["baseline_ssim"].mean(),
            "mse":  df["baseline_mse"].mean()
        },
        "finetuned": {
            "psnr": df["finetuned_psnr"].mean(),
            "ssim": df["finetuned_ssim"].mean(),
            "mse":  df["finetuned_mse"].mean()
        }
    }
    
    json_path = OUT_DIR / "metrics" / "evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[4/4] Creating Animations...")
    if base_anim_frames:
        make_animation(base_anim_frames[:30], OUT_DIR / "animations" / "baseline")
        make_animation(ft_anim_frames[:30],   OUT_DIR / "animations" / "finetuned")

    print(f"\n=======================================================")
    print(f"  EVALUATION COMPLETE")
    print(f"  Baseline PSNR:  {summary['baseline']['psnr']:.3f} dB")
    print(f"  Finetuned PSNR: {summary['finetuned']['psnr']:.3f} dB")
    print(f"  Outputs saved to: {OUT_DIR}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluate(TEST_DIR, args.limit, device)
