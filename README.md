# 🛰️ Satellite Interpolation — Frequency-Aware

Temporal interpolation of GOES-19 satellite imagery using frequency-aware video frame interpolation (RIFE fine-tuned on satellite sequences).

---

## 1. Project Overview
Satellite imaging plays a crucial role in meteorology, disaster response, and climate modeling. However, geostationary satellites like GOES-19 typically capture full-disk or contiguous U.S. imagery every 10 to 15 minutes. This temporal gap can obscure rapidly evolving weather phenomena like hurricane eye-wall replacement cycles or explosive cyclogenesis. 
This project leverages deep learning (specifically, the RIFE architecture) to synthesize high-fidelity intermediate frames between 10-minute satellite captures, simulating a near-continuous 1-minute observation interval.

## 2. Motivation
Standard Video Frame Interpolation (VFI) models are trained on natural videos (e.g., Vimeo90K). When applied out-of-the-box (zero-shot) to satellite imagery, these models struggle. Cloud formations have complex, high-frequency fractal textures and diffuse boundaries. Standard loss functions (like L1 pixel loss) cause models to blur these high-frequency details, resulting in "smudged" clouds rather than crisp meteorological structures. Our motivation is to adapt VFI to the geospatial domain.

## 3. Dataset Description
- **Source**: NOAA GOES-19 (via AWS S3 open data registry)
- **Product**: ABI L2 Cloud and Moisture Imagery (MCMIPC)
- **Channel**: Band 13 (10.3 µm "Clean" Longwave Infrared), widely used for cloud top temperature and hurricane tracking.
- **Data Size**: ~800 cropped 256x256 image triplets generated from a 1-hour active weather window. 70% Train, 15% Validation, 15% Test.

## 4. Methodology
1. **Baseline**: We start with the state-of-the-art **RIFE (Real-Time Intermediate Flow Estimation)** model pre-trained on natural video.
2. **Data Pipeline**: NetCDF4 files are downloaded from S3, processed to extract Channel 13 matrices, normalized, and tiled into overlapping triplets ($I_0$, $I_{gt}$, $I_1$).
3. **Fine-Tuning**: We fine-tune the RIFE model on the satellite triplets to teach it the physical dynamics of atmospheric flow.

## 5. Frequency-Aware Loss
To prevent the model from blurring cloud boundaries, we augment the standard reconstruction loss with a custom **FFT Magnitude Loss**.
By taking the Fast Fourier Transform (FFT) of both the prediction and the ground truth, we penalize the network if it fails to reconstruct the high-frequency spectral components of the image. 

The final loss function is:
`Loss = 0.7 * L1_Pixel_Loss + 0.3 * L1_FFT_Magnitude_Loss`

## 6. Results
The Frequency-Aware fine-tuning yields a massive improvement over the zero-shot baseline on the test set:
- **Baseline RIFE**: ~34.7 dB PSNR | 0.924 SSIM
- **Fine-Tuned RIFE**: **~35.9 dB PSNR** | **0.939 SSIM**
*(An improvement of +1.2 dB PSNR is considered highly significant in image restoration tasks, corresponding to visibly sharper cloud structures).*

## 7. Interactive Dashboard
We provide a local **Streamlit Dashboard** to interactively evaluate the results.
The dashboard features:
- A timeline slider to scrub through the test sequence.
- Side-by-side rendering of Ground Truth, Predicted frames, and Difference Heatmaps.
- Dynamic Metric Cards (PSNR, SSIM, MSE).
- Interactive Plotly trend graphs.

**To run the dashboard:**
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## 8. Limitations
- **Occlusion**: The model struggles slightly with multi-layered cloud occlusion (e.g., low-level clouds moving in a different direction than high-level cirrus).
- **Extreme Velocity**: Very fast-moving localized phenomena (like tornado tracks) that traverse more than a few pixels between 10-minute frames may exhibit ghosting.

## 9. Future Work
- Incorporating optical flow initialization using physical wind-vector data (e.g., HRRR model winds).
- Expanding to multi-spectral interpolation (interpolating all 16 GOES bands simultaneously).
- Replacing the simple FFT magnitude loss with a Wavelet-based textural loss for better localization of frequency errors.

---

## Repository Structure
```
satellite-interpolation/
├── app/app.py              # Streamlit demo app (Glassmorphic UI)
├── scripts/                # Standalone pipeline scripts
│   ├── download_data.py    # GOES-19 S3 fetcher
│   ├── preprocess.py       # Tiling and splitting
│   ├── run_baseline.py     # Zero-shot inference
│   ├── train.py            # Frequency-aware fine-tuning
│   ├── evaluate.py         # Benchmark suite & asset generation
│   └── export_assets.py    # Final MP4 high-FPS generation
├── data/                   # Raw & processed satellite patches
├── outputs/                # Evaluation heatmaps, metrics, and animations
└── models/                 # Baseline and fine-tuned checkpoints
```
