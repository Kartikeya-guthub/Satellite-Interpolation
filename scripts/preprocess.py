# -*- coding: utf-8 -*-
"""
preprocess.py
=============
Prepares raw GOES-19 PNG frames for RIFE interpolation training.

Pipeline
--------
  raw/frame0N.png
      |
      v
  [1] Build triplets      (frame_A, frame_B -> ground-truth frame_mid)
      |
      v
  [2] Slide & crop        non-overlapping / overlapping tiles
      |                   default: 256x256, stride 192 (64-px overlap)
      v
  [3] Quality filter      discard nearly-blank / cloud-free tiles
      |
      v
  [4] Augment (opt.)      horizontal flip for extra variety
      |
      v
  [5] Split               train / validation / test  (70/15/15 %)
      |
      v
  [6] Save                data/processed/  data/train/  data/validation/  data/test/

Triplet format (RIFE-compatible)
---------------------------------
  Each triplet saved as a folder:
    data/train/triplet_0000/
        img0.png   <- input frame A  (t-1)
        img1.png   <- input frame B  (t+1)
        gt.png     <- ground truth   (t)

Usage
-----
  python scripts/preprocess.py                           # defaults
  python scripts/preprocess.py --tile 512 --stride 384  # larger tiles
  python scripts/preprocess.py --no-augment --split 80 10 10
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
RAW_DIR   = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR   = ROOT / "data" / "validation"
TEST_DIR  = ROOT / "data" / "test"

# ── defaults ──────────────────────────────────────────────────────────────────
TILE_SIZE   = 256
STRIDE      = 192        # overlap = TILE_SIZE - STRIDE = 64 px
SPLIT_RATIO = (70, 15, 15)   # train / val / test  (must sum to 100)
MIN_STD     = 4.0        # discard tiles with std < MIN_STD (flat/empty)
SEED        = 42


# ── helpers ───────────────────────────────────────────────────────────────────
def load_frames(raw_dir: Path) -> list[tuple[str, np.ndarray]]:
    """Return sorted list of (stem, uint8 array) from raw dir."""
    pngs = sorted(raw_dir.glob("frame*.png"))
    if len(pngs) < 3:
        sys.exit(
            f"[ERROR] Need at least 3 frames in {raw_dir}, found {len(pngs)}.\n"
            "Run: python scripts/download_data.py"
        )
    frames = []
    for p in pngs:
        arr = np.array(Image.open(p).convert("L"))
        frames.append((p.stem, arr))
    return frames


def build_triplets(
    frames: list[tuple[str, np.ndarray]]
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Sliding window over frame list with step=1:
      triplet i: (frames[i], frames[i+2], frames[i+1])
                  img0         img1         gt
    Returns list of (name, img0, img1, gt).
    """
    triplets = []
    for i in range(len(frames) - 2):
        name_a, arr_a = frames[i]
        name_b, arr_b = frames[i + 1]   # ground-truth (middle)
        name_c, arr_c = frames[i + 2]
        label = f"{name_a}_{name_c}"
        triplets.append((label, arr_a, arr_c, arr_b))   # img0, img1, gt
    return triplets


def tile_triplet(
    img0: np.ndarray,
    img1: np.ndarray,
    gt:   np.ndarray,
    tile: int,
    stride: int,
    min_std: float,
    augment: bool,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Crop the three arrays into tile x tile patches using sliding window.
    Returns list of (img0_tile, img1_tile, gt_tile) passing the quality filter.
    """
    H, W = gt.shape
    results = []

    ys = list(range(0, H - tile + 1, stride))
    xs = list(range(0, W - tile + 1, stride))

    # ensure we capture the rightmost / bottommost strip
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)

    for y in ys:
        for x in xs:
            t0 = img0[y:y+tile, x:x+tile]
            t1 = img1[y:y+tile, x:x+tile]
            tg = gt  [y:y+tile, x:x+tile]

            # quality filter: skip nearly-flat (no-data / ocean) tiles
            if tg.std() < min_std:
                continue

            results.append((t0, t1, tg))

            # horizontal flip augmentation
            if augment:
                results.append((
                    np.fliplr(t0).copy(),
                    np.fliplr(t1).copy(),
                    np.fliplr(tg).copy(),
                ))

    return results


def save_triplet(
    folder: Path,
    idx:    int,
    t0:     np.ndarray,
    t1:     np.ndarray,
    tg:     np.ndarray,
) -> None:
    dest = folder / f"triplet_{idx:05d}"
    dest.mkdir(parents=True, exist_ok=True)
    Image.fromarray(t0, "L").save(dest / "img0.png")
    Image.fromarray(t1, "L").save(dest / "img1.png")
    Image.fromarray(tg, "L").save(dest / "gt.png")


def split_indices(
    n: int, ratios: tuple[int, int, int], seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Randomly split n indices into train/val/test according to ratios."""
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    tr, va, te = ratios
    total = tr + va + te
    n_tr = round(n * tr / total)
    n_va = round(n * va / total)
    return idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Preprocess GOES-19 frames for RIFE training.")
    p.add_argument("--raw-dir",  type=Path, default=RAW_DIR)
    p.add_argument("--tile",     type=int,  default=TILE_SIZE,
                   help="Tile size in pixels (default: 256)")
    p.add_argument("--stride",   type=int,  default=STRIDE,
                   help="Sliding window stride (default: 192, gives 64-px overlap)")
    p.add_argument("--min-std",  type=float, default=MIN_STD,
                   help="Min std-dev to keep a tile (default: 4.0)")
    p.add_argument("--split",    type=int, nargs=3, default=list(SPLIT_RATIO),
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Split percentages (default: 70 15 15)")
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="Disable horizontal-flip augmentation")
    p.add_argument("--seed",     type=int, default=SEED)
    p.add_argument("--clean",    action="store_true",
                   help="Delete existing processed/train/val/test dirs first")
    p.set_defaults(augment=True)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    tile   = args.tile
    stride = args.stride
    ratios = tuple(args.split)

    print("\n" + "="*62)
    print("  preprocess.py  --  GOES-19 -> RIFE training data")
    print("="*62)
    print(f"  Raw dir     : {args.raw_dir}")
    print(f"  Tile size   : {tile}x{tile} px")
    print(f"  Stride      : {stride} px  (overlap {tile-stride} px)")
    print(f"  Min std     : {args.min_std}")
    print(f"  Augment     : {'yes (+ horizontal flip)' if args.augment else 'no'}")
    print(f"  Split       : train {ratios[0]}% / val {ratios[1]}% / test {ratios[2]}%")
    print()

    # ── clean old outputs ─────────────────────────────────────────────────
    if args.clean:
        for d in [PROC_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
            if d.exists():
                shutil.rmtree(d)
                print(f"  Removed: {d}")

    for d in [PROC_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. load frames ────────────────────────────────────────────────────
    print("[1/5] Loading raw frames ...")
    frames = load_frames(args.raw_dir)
    print(f"  Loaded {len(frames)} frame(s): "
          f"{frames[0][1].shape[1]}x{frames[0][1].shape[0]} px each")

    # ── 2. build triplets ─────────────────────────────────────────────────
    print(f"\n[2/5] Building triplets (sliding window, step=1) ...")
    triplets = build_triplets(frames)
    print(f"  {len(triplets)} triplet(s) from {len(frames)} frames")
    for i, (name, _, _, _) in enumerate(triplets):
        print(f"    [{i+1}] {name}")

    # ── 3. tile all triplets -> flat patch list ───────────────────────────
    print(f"\n[3/5] Tiling ({tile}x{tile}, stride={stride}) + quality filter ...")
    all_patches = []   # list of (img0, img1, gt)

    for tri_name, img0, img1, gt in tqdm(triplets, desc="  Triplets"):
        patches = tile_triplet(img0, img1, gt, tile, stride,
                               args.min_std, args.augment)
        all_patches.extend(patches)
        tqdm.write(f"    {tri_name}: {len(patches)} patches kept")

    total = len(all_patches)
    print(f"\n  Total patches : {total}")
    aug_factor = 2 if args.augment else 1
    print(f"  (includes {aug_factor}x augmentation)")

    if total == 0:
        sys.exit("[ERROR] No patches generated. Try --min-std 0 or check frames.")

    # ── 4. split ──────────────────────────────────────────────────────────
    print(f"\n[4/5] Splitting into train/val/test ...")
    tr_idx, va_idx, te_idx = split_indices(total, ratios, args.seed)
    print(f"  Train : {len(tr_idx)}")
    print(f"  Val   : {len(va_idx)}")
    print(f"  Test  : {len(te_idx)}")

    # ── 5. save ───────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving triplets ...")

    split_map = [
        ("train",      TRAIN_DIR, tr_idx),
        ("validation", VAL_DIR,   va_idx),
        ("test",       TEST_DIR,  te_idx),
    ]

    global_idx = 0
    for split_name, split_dir, indices in split_map:
        print(f"\n  -> {split_name}/  ({len(indices)} triplets)")
        for local_i, patch_idx in enumerate(
            tqdm(indices, desc=f"  {split_name}", leave=False)
        ):
            t0, t1, tg = all_patches[patch_idx]
            save_triplet(split_dir, local_i, t0, t1, tg)
            global_idx += 1

        # also mirror into processed/ with split prefix for browsing
        # (lightweight: just store a text manifest, not duplicate files)
        manifest = PROC_DIR / f"{split_name}_manifest.txt"
        with open(manifest, "w") as mf:
            for local_i in range(len(indices)):
                mf.write(f"triplet_{local_i:05d}\n")

    # ── summary ───────────────────────────────────────────────────────────
    print(f"""
{'='*62}
  Preprocessing complete!

  Tile size      : {tile}x{tile} px
  Total patches  : {total}
  Train          : {len(tr_idx)}  ->  {TRAIN_DIR}
  Validation     : {len(va_idx)}  ->  {VAL_DIR}
  Test           : {len(te_idx)}  ->  {TEST_DIR}

  Each triplet folder contains:
    img0.png  (input frame A)
    img1.png  (input frame B)
    gt.png    (ground-truth middle frame)

  Next: Phase 4 -- run baseline RIFE inference.
{'='*62}
""")


if __name__ == "__main__":
    main()
