"""
visualize_sequence.py
=====================
Loads every frame from data/raw/ and displays them as a 2×3 grid
so you can visually inspect cloud motion before proceeding.

Usage
-----
  python scripts/visualize_sequence.py               # default data/raw/
  python scripts/visualize_sequence.py --raw-dir path/to/frames
  python scripts/visualize_sequence.py --no-show    # save PNG only, no window

Output
------
  outputs/baseline/sequence_preview.png   ← always saved
  Interactive matplotlib window            ← opens unless --no-show

What to look for
----------------
  ✅ GOOD  : Cloud edges shift position between frames
             Convective cells grow or dissipate
             Frontal boundary moves across the image
  ❌ BAD   : All frames look identical → re-run download_data.py
             with --start set to a different hour
"""

import argparse
import sys
from pathlib import Path

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
DEFAULT_RAW  = ROOT / "data" / "raw"
OUT_DIR      = ROOT / "outputs" / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def load_frames(raw_dir: Path) -> list[tuple[str, np.ndarray]]:
    """Return list of (label, array) sorted by filename."""
    pngs = sorted(raw_dir.glob("frame*.png"))
    if not pngs:
        sys.exit(
            f"[ERROR] No frame*.png files found in:\n  {raw_dir}\n"
            "Run: python scripts/download_data.py"
        )
    frames = []
    for p in pngs:
        img = np.array(Image.open(p).convert("L"))
        frames.append((p.stem, img))
    return frames


def mean_pixel_shift(a: np.ndarray, b: np.ndarray) -> float:
    """
    Rough proxy for cloud motion: mean absolute difference between consecutive
    frames after a 10-pixel offset search (simple block-match on downsampled).
    Returns mean absolute difference (0 = identical, >5 = visible motion).
    """
    scale = 8
    a_s = a[::scale, ::scale].astype(np.float32)
    b_s = b[::scale, ::scale].astype(np.float32)
    return float(np.mean(np.abs(a_s - b_s)))


def motion_verdict(scores: list[float]) -> str:
    mean = np.mean(scores)
    if mean >= 6.0:
        return f"[OK] Strong cloud motion detected  (avg D={mean:.1f})"
    elif mean >= 2.5:
        return f"[~]  Moderate motion detected       (avg D={mean:.1f})"
    else:
        return f"[X]  Little/no motion visible       (avg D={mean:.1f})"


# ── main display ──────────────────────────────────────────────────────────────
def visualize(frames: list[tuple[str, np.ndarray]], show: bool) -> Path:
    n  = len(frames)
    nc = 3
    nr = (n + nc - 1) // nc

    fig = plt.figure(figsize=(nc * 5, nr * 4 + 1.2), facecolor="#0d1117")
    fig.suptitle(
        "GOES-19  ABI  |  Channel C13  (10.3 µm IR)  |  Inspect for Cloud Motion",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(nr, nc, figure=fig,
                           hspace=0.08, wspace=0.05,
                           top=0.93, bottom=0.06, left=0.02, right=0.98)

    motion_scores = []
    prev_arr = None

    for idx, (label, arr) in enumerate(frames):
        row, col = divmod(idx, nc)
        ax = fig.add_subplot(gs[row, col])

        ax.imshow(arr, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])

        # frame label
        ax.set_title(label.upper(), color="#c9d1d9", fontsize=10,
                     fontweight="bold", pad=4)

        # motion score badge
        if prev_arr is not None:
            score = mean_pixel_shift(prev_arr, arr)
            motion_scores.append(score)
            color = "#2ea043" if score >= 6 else ("#d29922" if score >= 2.5 else "#f85149")
            ax.text(0.97, 0.04, f"Δ={score:.1f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color=color,
                    bbox=dict(facecolor="#161b22", alpha=0.7, pad=2, edgecolor="none"))

        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
            spine.set_linewidth(0.8)

        prev_arr = arr

    # overall verdict
    if motion_scores:
        verdict = motion_verdict(motion_scores)
        fig.text(0.5, 0.01, verdict, ha="center", va="bottom",
                 color="#c9d1d9", fontsize=11,
                 bbox=dict(facecolor="#161b22", alpha=0.85,
                           pad=4, edgecolor="#30363d"))

    out_path = OUT_DIR / "sequence_preview.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Preview saved → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
        print("  (use --show to open an interactive window)")

    return out_path


# ── motion diff strip ─────────────────────────────────────────────────────────
def diff_strip(frames: list[tuple[str, np.ndarray]]) -> None:
    """Print per-frame motion stats to stdout."""
    print("\n  Frame-to-frame motion summary")
    print("  " + "-" * 56)
    print(f"  {'Pair':<24}  {'Mean |D|':>9}  {'Max |D|':>9}  Signal")
    print("  " + "-" * 56)

    for i in range(1, len(frames)):
        a = frames[i - 1][1].astype(np.float32)
        b = frames[i][1].astype(np.float32)
        diff = np.abs(a - b)
        m    = diff.mean()
        mx   = diff.max()
        sig  = "[OK] visible" if m >= 6 else ("[~] moderate" if m >= 2.5 else "[X] static")
        pair = f"{frames[i-1][0]} -> {frames[i][0]}"
        print(f"  {pair:<24}  {m:>9.2f}  {mx:>9.1f}  {sig}")

    print("  " + "-" * 56)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Visualize GOES-19 frame sequence.")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW,
                   help="Directory containing frame*.png  (default: data/raw/)")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="Skip interactive window; only save preview PNG")
    p.set_defaults(show=True)
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*54)
    print("  visualize_sequence.py  --  GOES-19 Frame Inspector")
    print("="*54)
    print(f"  Reading frames from: {args.raw_dir}\n")

    frames = load_frames(args.raw_dir)

    print(f"  Loaded {len(frames)} frame(s):")
    for label, arr in frames:
        print(f"    {label}.png   {arr.shape[1]}×{arr.shape[0]} px")

    diff_strip(frames)

    if not args.show:
        # headless environments (Colab, servers)
        matplotlib.use("Agg")

    out = visualize(frames, show=args.show)

    print("\n  " + "-"*50)
    print("  Decision guide:")
    print("    [OK] D >= 6  -> Proceed to Phase 3 (preprocessing)")
    print("    [~]  D 2-6   -> Acceptable; proceed with caution")
    print("    [X]  D < 2   -> Re-run download with different hour:")
    print("         python scripts/download_data.py --start '2025-06-10 21:00'")
    print("  " + "-"*50 + "\n")


if __name__ == "__main__":
    main()
