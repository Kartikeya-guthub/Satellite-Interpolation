import streamlit as st
import pandas as pd
import json
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="GOES-19 RIFE Interpolation",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "evaluation"
METRICS_CSV = EVAL_DIR / "metrics" / "evaluation_metrics.csv"
METRICS_JSON = EVAL_DIR / "metrics" / "evaluation_summary.json"


@st.cache_data
def load_data():
    if not METRICS_CSV.exists():
        return None, None
    df = pd.read_csv(METRICS_CSV)
    with open(METRICS_JSON, "r") as f:
        summary = json.load(f)
    return df, summary


def main():
    st.title("🛰️ GOES-19 Satellite Frame Interpolation")
    st.markdown("Evaluating RIFE (Real-Time Intermediate Flow Estimation) for tracking cloud dynamics via zero-shot vs fine-tuned architectures.")

    df, summary = load_data()
    if df is None:
        st.error(f"❌ Evaluation data not found at `{EVAL_DIR}`. Please run Phase 6 (`evaluate.py`) on Colab and download the `outputs/` folder to your local machine.")
        st.stop()

    frames = df["frame"].tolist()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_choice = st.radio(
            "Select Model:",
            options=["Baseline RIFE", "Frequency Fine-Tuned RIFE"],
            index=1
        )
        is_finetuned = (model_choice == "Frequency Fine-Tuned RIFE")
        prefix = "finetuned" if is_finetuned else "baseline"

        st.divider()
        st.header("🎞️ Timeline")
        frame_idx = st.slider("Select Frame:", min_value=0, max_value=len(frames)-1, value=0)
        selected_frame = frames[frame_idx]
        
        st.divider()
        st.markdown("""
        **About the Models**
        - **Baseline**: Zero-shot HDv4.25 pre-trained on Vimeo90K.
        - **Fine-Tuned**: Adapted to GOES-19 imagery using custom FFT Magnitude Loss to preserve high-frequency cloud boundary structures.
        """)

    # ── Main Content ──────────────────────────────────────────────────────────
    
    # 1. Metrics Cards
    st.subheader(f"Metrics for Frame: `{selected_frame}`")
    
    col1, col2, col3 = st.columns(3)
    
    row = df.iloc[frame_idx]
    
    if is_finetuned:
        psnr_val = row["finetuned_psnr"]
        psnr_delta = psnr_val - row["baseline_psnr"]
        ssim_val = row["finetuned_ssim"]
        ssim_delta = ssim_val - row["baseline_ssim"]
        mse_val = row["finetuned_mse"]
        mse_delta = mse_val - row["baseline_mse"]
    else:
        psnr_val = row["baseline_psnr"]
        psnr_delta = None
        ssim_val = row["baseline_ssim"]
        ssim_delta = None
        mse_val = row["baseline_mse"]
        mse_delta = None

    col1.metric("PSNR (dB)", f"{psnr_val:.3f}", delta=f"{psnr_delta:+.3f}" if psnr_delta else None, help="Higher is better")
    col2.metric("SSIM", f"{ssim_val:.4f}", delta=f"{ssim_delta:+.4f}" if ssim_delta else None, help="Higher is better")
    col3.metric("MSE", f"{mse_val:.2f}", delta=f"{mse_delta:+.2f}" if mse_delta else None, delta_color="inverse", help="Lower is better")


    # 2. Image Comparison Panel
    st.subheader("Visual Inspection")
    
    c1, c2, c3 = st.columns(3)
    
    gt_path = EVAL_DIR / "comparisons" / f"{selected_frame}_gt.png"
    pred_path = EVAL_DIR / "comparisons" / f"{selected_frame}_pred_{prefix}.png"
    heat_path = EVAL_DIR / "heatmaps" / f"{selected_frame}_heat_{prefix}.png"

    with c1:
        st.markdown("**Ground Truth (Middle Frame)**")
        if gt_path.exists():
            st.image(Image.open(gt_path), use_column_width=True)
        else:
            st.warning("Missing image")

    with c2:
        st.markdown(f"**Predicted ({model_choice})**")
        if pred_path.exists():
            st.image(Image.open(pred_path), use_column_width=True)
        else:
            st.warning("Missing image")
            
    with c3:
        st.markdown("**Difference Heatmap**")
        if heat_path.exists():
            st.image(Image.open(heat_path), use_column_width=True)
        else:
            st.warning("Missing image")


    # 3. Trend Graph
    st.divider()
    st.subheader("📊 Performance Trends")
    
    metric_choice = st.selectbox("Select Metric to Plot:", ["PSNR", "SSIM", "MSE"])
    
    fig = go.Figure()
    
    if metric_choice == "PSNR":
        fig.add_trace(go.Scatter(x=df.index, y=df["baseline_psnr"], name="Baseline", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=df.index, y=df["finetuned_psnr"], name="Fine-Tuned", line=dict(color="#ff7f0e")))
        fig.update_layout(yaxis_title="PSNR (dB)")
    elif metric_choice == "SSIM":
        fig.add_trace(go.Scatter(x=df.index, y=df["baseline_ssim"], name="Baseline", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=df.index, y=df["finetuned_ssim"], name="Fine-Tuned", line=dict(color="#ff7f0e")))
        fig.update_layout(yaxis_title="SSIM")
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["baseline_mse"], name="Baseline", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=df.index, y=df["finetuned_mse"], name="Fine-Tuned", line=dict(color="#ff7f0e")))
        fig.update_layout(yaxis_title="MSE")
        
    # Highlight current frame
    fig.add_vline(x=frame_idx, line_dash="dash", line_color="red", annotation_text="Current Frame")
    
    fig.update_layout(
        xaxis_title="Sequence Frame Index",
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)


    # 4. Animations
    st.divider()
    st.subheader("🎥 Sequence Animations")
    
    ac1, ac2 = st.columns(2)
    
    anim_base = EVAL_DIR / "animations" / "baseline.gif"
    anim_ft = EVAL_DIR / "animations" / "finetuned.gif"
    
    with ac1:
        st.markdown("**Baseline Animation**")
        if anim_base.exists():
            st.image(str(anim_base), use_column_width=True)
            
    with ac2:
        st.markdown("**Fine-Tuned Animation**")
        if anim_ft.exists():
            st.image(str(anim_ft), use_column_width=True)


if __name__ == "__main__":
    main()
