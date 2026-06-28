# -*- coding: utf-8 -*-
"""
download_data.py
================
Downloads 6 consecutive GOES-19 ABI frames directly from the public
AWS S3 bucket (noaa-goes19). No API key or account required.

Pipeline
--------
  NOAA AWS S3  →  list files  →  download NetCDF  →  extract C13
  →  normalize (K → uint8)  →  save PNG  →  data/raw/frame0N.png

Product  : ABI-L2-MCMIPC  (Multi-Channel CMI, CONUS, 5-min cadence)
Channel  : C13  —  10.3 µm clean longwave IR
                   Works day AND night. Cold cloud tops → bright white.
                   Convection, fronts, and MCS boundaries are vivid.
Interval : 10-minute steps  (every 2nd 5-min scan → 6 frames = 1 hour)

Default window : 2025-06-10  20:00 – 21:00 UTC
  • Afternoon over CONUS (peak convection hour)
  • GOES-19 fully operational since April 7 2025
  • MCS activity common over Great Plains in June

Override via CLI: python download_data.py --start "2025-06-10 19:00" --hours 1
"""

import argparse
import io
import shutil
import sys

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# ── project paths ─────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
BUCKET       = "noaa-goes19"
PRODUCT      = "ABI-L2-MCMIPC"          # CONUS, 5-min scan
CHANNEL      = "CMI_C13"                # 10.3 µm brightness temperature (K)
N_FRAMES     = 6
STEP_MINUTES = 10                       # one frame every 10 minutes

# Brightness-temperature bounds for IR normalisation
T_MIN = 190.0   # cold anvil tops (dark storms → white in inverted palette)
T_MAX = 310.0   # warm land surface

DEFAULT_START = "2025-06-10 20:00"      # UTC


# ── normalisation ─────────────────────────────────────────────────────────────
def normalise_ir(data: np.ndarray) -> np.ndarray:
    """
    Clip BT to [T_MIN, T_MAX] then invert so cold tops = 255 (white).
    NaN fill values (GOES no-data pixels) are set to 0 (black border).
    Returns uint8 array suitable for PIL 'L' mode.
    """
    arr = data.astype(np.float32)
    arr = np.nan_to_num(arr, nan=T_MAX)          # NaN -> warm (dark after invert)
    arr = np.clip(arr, T_MIN, T_MAX)
    arr = (T_MAX - arr) / (T_MAX - T_MIN)        # invert: cold tops -> bright
    arr = np.clip(arr * 255, 0, 255)             # guard against float edge cases
    return arr.astype(np.uint8)


# ── S3 helpers ────────────────────────────────────────────────────────────────
def get_s3():
    """Return an anonymous S3FileSystem (public bucket, no credentials)."""
    try:
        import s3fs
    except ImportError:
        sys.exit("[ERROR] s3fs not installed.  Run: pip install s3fs")
    return s3fs.S3FileSystem(anon=True)


def list_product_files(fs, dt: datetime) -> list[str]:
    """
    List all MCMIPC NetCDF files for a given UTC hour from the S3 bucket.
    S3 path: noaa-goes19/ABI-L2-MCMIPC/YYYY/DDD/HH/
    """
    doy  = dt.strftime("%j")      # day-of-year, zero-padded to 3 digits
    hour = dt.strftime("%H")
    year = dt.strftime("%Y")
    prefix = f"{BUCKET}/{PRODUCT}/{year}/{doy}/{hour}/"
    try:
        files = fs.ls(prefix)
        # keep only .nc files
        return sorted(f for f in files if f.endswith(".nc"))
    except FileNotFoundError:
        return []


def pick_frames(all_files: list[str], n: int, step_minutes: int,
                files_per_hour: int = 12) -> list[str]:
    """
    Choose n files spaced ~step_minutes apart from the available list.
    CONUS scans every 5 min → 12 files/hour.
    step_minutes=10 → every 2nd file.
    """
    step = max(1, step_minutes // (60 // files_per_hour))
    candidates = all_files[::step]
    if len(candidates) < n:
        candidates = all_files                      # fall back to all
    return candidates[:n]


def download_nc(fs, s3_path: str, dest: Path) -> Path:
    """Stream a NetCDF file from S3 to local dest, show progress."""
    size_bytes = fs.info(s3_path)["size"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    chunk = 1 << 20   # 1 MB
    downloaded = 0
    with fs.open(s3_path, "rb") as src, open(dest, "wb") as out:
        while True:
            block = src.read(chunk)
            if not block:
                break
            out.write(block)
            downloaded += len(block)
            pct = downloaded / size_bytes * 100
            print(f"\r    {pct:5.1f}%  [{downloaded//1024:,} / {size_bytes//1024:,} KB]",
                  end="", flush=True)
    print()
    return dest


# ── PNG export ────────────────────────────────────────────────────────────────
def nc_to_png(nc_path: Path, png_path: Path) -> tuple[np.ndarray, dict]:
    """Open NetCDF, extract C13, normalise, save PNG. Returns (array, meta)."""
    try:
        import xarray as xr
    except ImportError:
        sys.exit("[ERROR] xarray not installed.  Run: pip install xarray netCDF4 h5netcdf")

    ds = xr.open_dataset(str(nc_path), engine="netcdf4")

    if CHANNEL not in ds:
        available = list(ds.data_vars)
        ds.close()
        raise KeyError(f"'{CHANNEL}' not in dataset. Available: {available}")

    raw = ds[CHANNEL].values.astype(np.float32)

    # pull timestamp from the dataset if available
    meta = {}
    for attr in ("time_coverage_start", "date_created"):
        if attr in ds.attrs:
            meta["timestamp"] = ds.attrs[attr]
            break
    meta["t_min_k"]   = float(np.nanmin(raw))
    meta["t_max_k"]   = float(np.nanmax(raw))
    meta["shape"]     = raw.shape
    meta["nc_file"]   = nc_path.name

    ds.close()

    arr = normalise_ir(raw)
    img = Image.fromarray(arr, mode="L")
    img.save(png_path)
    return arr, meta


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Download 6 GOES-19 CONUS ABI-C13 frames from AWS S3."
    )
    p.add_argument("--start",  default=DEFAULT_START,
                   help="UTC start time  'YYYY-MM-DD HH:MM'  (default: %(default)s)")
    p.add_argument("--hours",  type=float, default=1.0,
                   help="Search window in hours (default: 1)")
    p.add_argument("--frames", type=int,   default=N_FRAMES,
                   help="Number of frames to download (default: %(default)s)")
    p.add_argument("--step",   type=int,   default=STEP_MINUTES,
                   help="Minutes between frames (default: %(default)s)")
    p.add_argument("--keep-nc", action="store_true",
                   help="Keep the original NetCDF files in data/raw/")
    return p.parse_args()


def main():
    args = parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M").replace(
        tzinfo=timezone.utc
    )
    end_dt   = start_dt + timedelta(hours=args.hours)

    total_min = args.frames * args.step
    print("\n" + "="*62)
    print("  GOES-19  |  ABI-L2-MCMIPC  |  Channel C13 (10.3 µm IR)")
    print(f"  Window   : {start_dt.strftime('%Y-%m-%d %H:%M')} - {end_dt.strftime('%H:%M')} UTC")
    print(f"  Frames   : {args.frames}  every {args.step} min  (~{total_min} min total)")
    print(f"  Output   : {RAW_DIR}")
    print("="*62 + "\n")

    fs = get_s3()

    # ── 1. collect file listing across all hours in the window ────────────
    print("[1/4] Querying S3 bucket  noaa-goes19  …")
    all_files = []
    cur = start_dt.replace(minute=0, second=0, microsecond=0)
    while cur <= end_dt:
        hour_files = list_product_files(fs, cur)
        all_files.extend(hour_files)
        cur += timedelta(hours=1)

    # filter to window
    def file_in_window(f: str) -> bool:
        """Parse start-time token from filename and compare to window."""
        name = Path(f).stem          # e.g. OR_ABI-L2-MCMIPC-M6_G19_s20251611900...
        try:
            # token after '_s': sYYYYDDDHHMMSSf
            s_token = [t for t in name.split("_") if t.startswith("s")][0][1:]
            # format: YYYYDDDHHMMSSf  (DDD = day-of-year)
            year = int(s_token[0:4])
            doy  = int(s_token[4:7])
            hh   = int(s_token[7:9])
            mm   = int(s_token[9:11])
            dt   = datetime(year, 1, 1, hh, mm, tzinfo=timezone.utc) + \
                   timedelta(days=doy - 1)
            return start_dt <= dt <= end_dt
        except Exception:
            return True   # if parsing fails, keep the file

    all_files = [f for f in all_files if file_in_window(f)]

    if not all_files:
        print(f"\n[ERROR] No files found for window  {args.start}.")
        print("  Suggestions:")
        print("  • Check internet connection")
        print("  * Check internet connection")
        print("  * Try a later date (GOES-19 operational from 2025-04-07)")
        print("  * Try: python download_data.py --start '2025-06-15 18:00'")
        sys.exit(1)

    frames_s3 = pick_frames(all_files, args.frames, args.step)
    print(f"  Found {len(all_files)} file(s) in window -> selected {len(frames_s3)}\n")

    # -- 2. download NetCDF files ───────────────────────────────────────────
    print(f"[2/4] Downloading NetCDF files from S3 ...\n")
    nc_paths = []
    for i, s3_path in enumerate(frames_s3, start=1):
        fname = Path(s3_path).name
        nc_dest = RAW_DIR / f"frame{i:02d}.nc"
        print(f"  [{i}/{len(frames_s3)}]  {fname}")
        try:
            download_nc(fs, s3_path, nc_dest)
            nc_paths.append((i, nc_dest))
        except Exception as exc:
            print(f"  [WARN] Failed to download frame {i}: {exc}")

    if not nc_paths:
        sys.exit("[ERROR] All downloads failed.")

    # -- 3. convert NetCDF -> PNG ───────────────────────────────────────────
    print(f"\n[3/4] Converting to PNG  (channel {CHANNEL}) ...\n")
    png_paths = []
    for i, nc_path in nc_paths:
        png_path = RAW_DIR / f"frame{i:02d}.png"
        try:
            arr, meta = nc_to_png(nc_path, png_path)
            h, w = meta["shape"]
            kb   = png_path.stat().st_size // 1024
            ts   = meta.get("timestamp", "n/a")
            print(f"  frame{i:02d}.png  |  {w}x{h} px  |  {kb} KB  |  {ts}")
            png_paths.append(png_path)
        except Exception as exc:
            print(f"  [WARN] frame{i:02d}: conversion failed - {exc}")

        # optionally remove NetCDF to save space
        if not args.keep_nc and nc_path.exists():
            nc_path.unlink()

    # -- 4. summary ────────────────────────────────────────────────────────
    print(f"\n[4/4] Done!")
    print(f"  {len(png_paths)} PNG frame(s) saved to: {RAW_DIR}")
    print(f"  {'  '.join(p.name for p in sorted(png_paths))}")
    print("\nNext step: run  python scripts/visualize_sequence.py  to inspect frames.")
    print("If cloud motion is NOT visible, re-run with a different --start time.\n")


if __name__ == "__main__":
    main()
