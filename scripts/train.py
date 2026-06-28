# -*- coding: utf-8 -*-
"""
train.py
========
Phase 5: Frequency-Aware Fine-Tuning of RIFE on satellite imagery.

Key Innovation — Frequency Loss
--------------------------------
Standard RIFE uses pixel-space Charbonier loss.
We add a 2D FFT magnitude loss that penalises errors in:
  - High-frequency components  (cloud edges, fine texture)
  - Mid-frequency components   (frontal structure, convective cells)
  - Low-frequency baseline     (scene brightness, large-scale gradients)

  L_total = alpha * L_pixel  +  beta * L_freq
  L_freq  = L1( |FFT(pred)| , |FFT(gt)| )

Pipeline
--------
  Load dataset  ->  Profile GPU (10 iters)  ->  Adjust batch/tile size
  ->  Train loop (checkpoint /50 iters)  ->  Validate  ->  Save best
  ->  Final model: models/finetuned/frequency_rife.pth

Usage (run in Colab or locally)
--------------------------------
  python scripts/train.py
  python scripts/train.py --epochs 30 --batch 4 --tile 256
  python scripts/train.py --profile-only      # just measure GPU stats
"""

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
RIFE_DIR      = ROOT / "models" / "rife"
TRAIN_DIR     = ROOT / "data" / "train"
VAL_DIR       = ROOT / "data" / "validation"
CKPT_DIR      = ROOT / "models" / "checkpoints"
FINETUNE_DIR  = ROOT / "models" / "finetuned"
METRICS_DIR   = ROOT / "outputs" / "metrics"

for d in [CKPT_DIR, FINETUNE_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATASET
# ============================================================================
class SatelliteTripletDataset:
    """
    Loads (img0, img1, gt) triplets from data/train/ or data/validation/.
    Images are grayscale -> repeated to 3 channels for RIFE compatibility.
    Returns float32 tensors in [0, 1].
    """
    def __init__(self, split_dir: Path, tile_size: int = 256,
                 augment: bool = True):
        self.triplets  = sorted(split_dir.iterdir())
        self.tile_size = tile_size
        self.augment   = augment

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        import torch
        tri = self.triplets[idx]

        def load(name):
            arr = np.array(Image.open(tri / name).convert("L"),
                           dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
            return t   # [3, H, W]

        img0 = load("img0.png")
        img1 = load("img1.png")
        gt   = load("gt.png")

        # horizontal flip augmentation
        if self.augment and np.random.rand() > 0.5:
            import torch
            img0 = torch.flip(img0, dims=[2])
            img1 = torch.flip(img1, dims=[2])
            gt   = torch.flip(gt,   dims=[2])

        return img0, img1, gt

    def to_loader(self, batch_size: int, shuffle: bool = True,
                  num_workers: int = 0):
        from torch.utils.data import DataLoader
        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          pin_memory=True, drop_last=True)


# ============================================================================
# FREQUENCY-AWARE LOSS
# ============================================================================
class FrequencyAwareLoss:
    """
    Combined pixel + frequency-domain loss.

    L_total = alpha * L_Charbonier  +  beta * L_freq_magnitude

    L_freq compares 2D FFT magnitude spectra of pred vs gt.
    This forces the model to preserve satellite-specific high-frequency
    structure: cloud edges, convective cell boundaries, frontal gradients.

    Optional: high-frequency emphasis via a radial weight mask that
    amplifies the outer (high-frequency) ring of the spectrum.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 hf_emphasis: float = 1.5, eps: float = 1e-3):
        self.alpha       = alpha
        self.beta        = beta
        self.hf_emphasis = hf_emphasis   # extra weight for HF components
        self.eps         = eps
        self._freq_mask  = None          # cached, built on first call

    def charbonier(self, pred, gt):
        import torch
        diff = pred - gt
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))

    def _build_freq_mask(self, H: int, W: int, device):
        """
        Radial frequency mask: center=low freq (weight 1), edge=high freq
        (weight up to hf_emphasis). Shape: [1, 1, H, W//2+1].
        """
        import torch
        cy, cx = H // 2, (W // 2 + 1) // 2
        ys = torch.arange(H, device=device).float() - cy
        xs = torch.arange(W // 2 + 1, device=device).float() - cx
        r  = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
        r_norm = r / (r.max() + 1e-8)
        mask = 1.0 + (self.hf_emphasis - 1.0) * r_norm
        return mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W//2+1]

    def frequency_loss(self, pred, gt):
        import torch
        B, C, H, W = pred.shape
        # 2D rFFT on first channel only (all channels identical for us)
        p_fft = torch.fft.rfft2(pred[:, 0:1, :, :], norm="ortho")
        g_fft = torch.fft.rfft2(gt  [:, 0:1, :, :], norm="ortho")

        p_mag = torch.abs(p_fft)
        g_mag = torch.abs(g_fft)

        # build / reuse frequency weight mask
        if (self._freq_mask is None
                or self._freq_mask.shape[-2:] != (H, W // 2 + 1)):
            self._freq_mask = self._build_freq_mask(H, W, pred.device)

        mask = self._freq_mask.to(pred.device)
        return torch.mean(mask * torch.abs(p_mag - g_mag))

    def __call__(self, pred, gt):
        l_pix  = self.charbonier(pred, gt)
        l_freq = self.frequency_loss(pred, gt)
        return self.alpha * l_pix + self.beta * l_freq, l_pix, l_freq


# ============================================================================
# GPU PROFILER
# ============================================================================
def profile_gpu(model, train_loader, device, n_iters: int = 10) -> dict:
    """
    Run n_iters forward+backward passes and measure:
      - GPU memory used
      - Iteration time
    Returns suggested batch_size and tile_size adjustments.
    """
    import torch

    print(f"\n  Profiling GPU ({n_iters} iterations) ...")
    criterion = FrequencyAwareLoss()
    optimizer = torch.optim.Adam(model.flownet.parameters(), lr=1e-4)

    times = []
    mem_peaks = []

    loader_iter = iter(train_loader)
    for i in range(n_iters):
        try:
            img0, img1, gt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            img0, img1, gt = next(loader_iter)

        img0, img1, gt = img0.to(device), img1.to(device), gt.to(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()
        pred = model.inference(img0, img1)
        loss, _, _ = criterion(pred, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.perf_counter()

        times.append(t1 - t0)
        if device.type == "cuda":
            mem_peaks.append(torch.cuda.max_memory_allocated(device) / 1e9)

    avg_time = np.mean(times)
    avg_mem  = np.mean(mem_peaks) if mem_peaks else 0.0

    profile = {
        "avg_iter_time_s":  round(avg_time, 3),
        "avg_gpu_mem_gb":   round(avg_mem, 3),
        "iters_per_minute": round(60 / avg_time, 1),
    }
    print(f"    Avg iteration time : {avg_time:.3f}s")
    print(f"    Avg GPU memory     : {avg_mem:.2f} GB")
    print(f"    Iters / minute     : {profile['iters_per_minute']}")

    # simple auto-suggestion
    if avg_mem > 12.0:
        print("    [WARN] High GPU memory — consider --batch 2 or --tile 192")
    elif avg_mem < 4.0:
        print("    [INFO] GPU memory low — could increase --batch 8 or --tile 320")
    else:
        print("    [OK] GPU memory usage looks healthy.")

    return profile


# ============================================================================
# TRAINING LOOP
# ============================================================================
def validate(model, val_loader, device) -> dict:
    import torch
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    model.eval()
    psnrs, ssims = [], []

    with torch.no_grad():
        for img0, img1, gt in val_loader:
            img0, img1, gt = img0.to(device), img1.to(device), gt.to(device)
            pred = model.inference(img0, img1)

            for b in range(pred.shape[0]):
                p = pred[b, 0].cpu().numpy() * 255
                g = gt  [b, 0].cpu().numpy() * 255
                p = np.clip(p, 0, 255).astype(np.uint8)
                g = np.clip(g, 0, 255).astype(np.uint8)
                psnrs.append(peak_signal_noise_ratio(g, p, data_range=255))
                ssims.append(structural_similarity(g, p, data_range=255))

    model.train()
    return {
        "val_psnr": float(np.mean(psnrs)),
        "val_ssim": float(np.mean(ssims)),
    }


def train(model, train_loader, val_loader, args, device) -> dict:
    import torch

    criterion = FrequencyAwareLoss(alpha=args.alpha, beta=args.beta)
    optimizer = torch.optim.Adam(
        model.flownet.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6
    )

    best_psnr   = 0.0
    history     = []
    global_step = 0

    print(f"\n  Starting training: {args.epochs} epochs, "
          f"batch={args.batch}, lr={args.lr}")
    print(f"  Loss: alpha={args.alpha}*pixel + beta={args.beta}*freq\n")

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []

        for img0, img1, gt in train_loader:
            img0, img1, gt = img0.to(device), img1.to(device), gt.to(device)

            pred = model.inference(img0, img1)
            loss, l_pix, l_freq = criterion(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.flownet.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            global_step += 1

            # checkpoint every 50 iterations
            if global_step % 50 == 0:
                ckpt_path = CKPT_DIR / f"iter_{global_step:05d}.pth"
                torch.save({
                    "step":       global_step,
                    "epoch":      epoch,
                    "state_dict": model.flownet.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "loss":       loss.item(),
                }, str(ckpt_path))

        # end-of-epoch validation
        val_metrics = validate(model, val_loader, device)
        avg_loss    = float(np.mean(epoch_losses))

        log = {
            "epoch":    epoch,
            "step":     global_step,
            "loss":     round(avg_loss, 5),
            "val_psnr": round(val_metrics["val_psnr"], 3),
            "val_ssim": round(val_metrics["val_ssim"], 4),
        }
        history.append(log)

        print(f"  Epoch {epoch:>3}/{args.epochs}  "
              f"loss={avg_loss:.5f}  "
              f"val_PSNR={val_metrics['val_psnr']:.3f}  "
              f"val_SSIM={val_metrics['val_ssim']:.4f}")

        # save best
        if val_metrics["val_psnr"] > best_psnr:
            best_psnr = val_metrics["val_psnr"]
            best_path = CKPT_DIR / "best_model.pth"
            torch.save({
                "epoch":      epoch,
                "state_dict": model.flownet.state_dict(),
                "val_psnr":   best_psnr,
            }, str(best_path))
            print(f"    *** New best PSNR: {best_psnr:.3f} dB -> saved ***")

        # epoch checkpoint
        ep_ckpt = CKPT_DIR / f"epoch_{epoch:03d}.pth"
        torch.save({
            "epoch":      epoch,
            "state_dict": model.flownet.state_dict(),
            "history":    history,
        }, str(ep_ckpt))

    return history


# ============================================================================
# MAIN
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Frequency-Aware RIFE Fine-Tuning")
    p.add_argument("--train-dir",     type=Path, default=TRAIN_DIR)
    p.add_argument("--val-dir",       type=Path, default=VAL_DIR)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch",         type=int,   default=4)
    p.add_argument("--tile",          type=int,   default=256)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--alpha",         type=float, default=0.7,
                   help="Pixel loss weight")
    p.add_argument("--beta",          type=float, default=0.3,
                   help="Frequency loss weight")
    p.add_argument("--profile-only",  action="store_true",
                   help="Only run 10-iter GPU profile, then exit")
    p.add_argument("--skip-setup",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*62)
    print("  Phase 5 -- Frequency-Aware RIFE Fine-Tuning")
    print("="*62)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device    : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Loss      : {args.alpha}*pixel + {args.beta}*frequency")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch}")

    # ── load model ─────────────────────────────────────────────────────────
    print("\n[1] Loading RIFE model ...")
    sys.path.insert(0, str(RIFE_DIR))
    try:
        from train_log.RIFE_HDv3 import Model
    except ImportError:
        try:
            from train_log.RIFE_HD import Model
        except ImportError:
            from train_log.RIFE_HDv2 import Model

    model = Model()
    model.load_model(str(RIFE_DIR / "train_log"), -1)
    model.train()

    # ── datasets ───────────────────────────────────────────────────────────
    print("\n[2] Loading datasets ...")
    train_ds  = SatelliteTripletDataset(args.train_dir, args.tile, augment=True)
    val_ds    = SatelliteTripletDataset(args.val_dir,   args.tile, augment=False)

    n_workers = 2 if device.type == "cuda" else 0
    train_loader = train_ds.to_loader(args.batch, shuffle=True,  num_workers=n_workers)
    val_loader   = val_ds.to_loader(  args.batch, shuffle=False, num_workers=n_workers)

    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── GPU profile ────────────────────────────────────────────────────────
    print("\n[3] GPU profiling (10 iterations) ...")
    profile = profile_gpu(model, train_loader, device, n_iters=10)

    prof_path = METRICS_DIR / "gpu_profile.json"
    with open(prof_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"  Profile saved -> {prof_path}")

    if args.profile_only:
        print("\n  --profile-only set. Exiting after profile.\n")
        return

    # ── train ──────────────────────────────────────────────────────────────
    print("\n[4] Training ...")
    history = train(model, train_loader, val_loader, args, device)

    # ── save final model ───────────────────────────────────────────────────
    print("\n[5] Saving final fine-tuned model ...")
    final_path = FINETUNE_DIR / "frequency_rife.pth"
    import torch
    torch.save({
        "state_dict": model.flownet.state_dict(),
        "history":    history,
        "config": {
            "alpha": args.alpha,
            "beta":  args.beta,
            "tile":  args.tile,
            "batch": args.batch,
            "epochs": args.epochs,
        },
    }, str(final_path))
    print(f"  Final model -> {final_path}")

    # save training history JSON
    hist_path = METRICS_DIR / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    best = max(history, key=lambda x: x["val_psnr"])
    print(f"""
{'='*62}
  TRAINING COMPLETE
  Best epoch : {best['epoch']}  PSNR={best['val_psnr']} dB  SSIM={best['val_ssim']}
  Final model: {final_path}
  History    : {hist_path}
  Checkpoints: {CKPT_DIR}
{'='*62}
""")


if __name__ == "__main__":
    main()
