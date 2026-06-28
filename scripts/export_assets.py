# -*- coding: utf-8 -*-
"""
export_assets.py
================
Phase 6: Final Sequence Generation using Fine-Tuned RIFE

Pipeline
--------
  Load raw GOES-19 sequence (data/raw/)
  -> Recursively interpolate frames using fine-tuned RIFE
  -> Output high-FPS smooth MP4 video (outputs/animations/final_sequence.mp4)

Usage
-----
  python scripts/export_assets.py
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
RIFE_DIR     = ROOT / "models" / "rife"
RAW_DIR      = ROOT / "data" / "raw"
FINETUNE_DIR = ROOT / "models" / "finetuned"
WEIGHTS_DIR  = RIFE_DIR / "train_log"
ANIM_DIR     = ROOT / "outputs" / "animations"

ANIM_DIR.mkdir(parents=True, exist_ok=True)


def load_rife_model(weights_path: Path, device):
    sys.path.insert(0, str(RIFE_DIR))
    try:
        from train_log.RIFE_HDv3 import Model
    except ImportError:
        from train_log.RIFE_HD import Model

    model = Model()
    model.load_model(str(WEIGHTS_DIR), -1)
    
    import torch
    # Override with our fine-tuned weights if they exist
    if weights_path.exists():
        print(f"  Loading fine-tuned weights: {weights_path.name}")
        ckpt = torch.load(weights_path, map_location=device)
        model.flownet.load_state_dict(ckpt["state_dict"])
    else:
        print(f"  [WARN] Fine-tuned weights not found. Using baseline.")

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


def make_inference(model, I0, I1, num_intermediates, device):
    """Recursively predict intermediate frames to reach num_intermediates."""
    import torch
    
    if num_intermediates == 0:
        return []
    
    # 1 intermediate = 2x frame rate
    # 3 intermediates = 4x frame rate
    # 7 intermediates = 8x frame rate
    
    if num_intermediates == 1:
        with torch.no_grad():
            middle = model.inference(I0, I1)
        return [middle]
        
    elif num_intermediates == 3:
        with torch.no_grad():
            mid = model.inference(I0, I1)
            left = model.inference(I0, mid)
            right = model.inference(mid, I1)
        return [left, mid, right]
        
    elif num_intermediates == 7:
        with torch.no_grad():
            mid = model.inference(I0, I1)
            mid_l = model.inference(I0, mid)
            mid_r = model.inference(mid, I1)
            
            l1 = model.inference(I0, mid_l)
            l2 = model.inference(mid_l, mid)
            r1 = model.inference(mid, mid_r)
            r2 = model.inference(mid_r, I1)
        return [l1, mid_l, l2, mid, r1, mid_r, r2]
    else:
        raise ValueError("num_intermediates must be 1, 3, or 7")


def export_video(frames: list[np.ndarray], out_path: Path, fps: int = 24):
    import imageio.v2 as imageio
    try:
        import imageio_ffmpeg  # noqa
        writer = imageio.get_writer(str(out_path), fps=fps, macro_block_size=1)
        for f in frames:
            writer.append_data(np.stack([f, f, f], axis=-1))
        writer.close()
        print(f"  MP4 saved -> {out_path}")
    except ImportError:
        print("  [WARN] imageio-ffmpeg not found. Saving as GIF.")
        gif_path = out_path.with_suffix(".gif")
        imageio.mimsave(str(gif_path), frames, fps=fps)
        print(f"  GIF saved -> {gif_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=int, default=30, help="Output video FPS")
    p.add_argument("--multiplier", type=int, default=8, choices=[2, 4, 8], 
                   help="Frame rate multiplier (2x, 4x, 8x)")
    args = p.parse_args()

    print("\n" + "="*62)
    print("  Phase 6 -- Final Sequence Generation (Fine-Tuned)")
    print("="*62)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load Model
    weights = FINETUNE_DIR / "frequency_rife.pth"
    model = load_rife_model(weights, device)

    # Load raw frames
    raw_files = sorted(RAW_DIR.glob("*.png"))
    if not raw_files:
        print(f"  [ERROR] No raw frames found in {RAW_DIR}")
        sys.exit(1)

    print(f"\n  Found {len(raw_files)} raw frames.")
    print(f"  Interpolating at {args.multiplier}x speed...")
    
    num_intermediates = args.multiplier - 1
    final_frames = []

    for i in range(len(raw_files) - 1):
        print(f"    Processing interval: {raw_files[i].name} -> {raw_files[i+1].name}")
        
        t0 = load_img_tensor(raw_files[i], device)
        t1 = load_img_tensor(raw_files[i+1], device)
        
        preds = make_inference(model, t0, t1, num_intermediates, device)
        
        # Add frame 0
        final_frames.append(tensor_to_uint8(t0))
        # Add predictions
        for p in preds:
            final_frames.append(tensor_to_uint8(p))
            
    # Add very last frame
    t_last = load_img_tensor(raw_files[-1], device)
    final_frames.append(tensor_to_uint8(t_last))

    print(f"\n  Generated {len(final_frames)} total frames (up from {len(raw_files)}).")

    # Export
    print("  Exporting video...")
    out_mp4 = ANIM_DIR / "satellite_timelapse_smooth.mp4"
    export_video(final_frames, out_mp4, fps=args.fps)

    print(f"\n{'='*62}")
    print(f"  ALL PHASES COMPLETE!")
    print(f"  Final video saved to: {out_mp4}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
