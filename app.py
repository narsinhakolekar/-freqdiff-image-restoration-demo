from __future__ import annotations

import io
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# -----------------------------------------------------------------------------
# Project paths and verified V2 benchmark values
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints" / "v2_corrected"
EVALUATION_DIR = ROOT / "results" / "v2_corrected" / "evaluation"
HISTORY_DIR = ROOT / "results" / "v2_corrected"

TILE_SIZE = 128
DIFFUSION_STEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
BASE_CHANNELS = 32
TIME_DIM = 128

# Fallback values are the V2 held-out results supplied from the completed run.
# If the local evaluation CSVs exist, those files take precedence.
BENCHMARK_SUMMARY_FALLBACK = pd.DataFrame(
    [
        {
            "method": "Degraded input",
            "n_source_images": 80,
            "mean_psnr_db": 23.9335,
            "std_psnr_db": 0.9510,
            "mean_ssim": 0.4159,
            "std_ssim": 0.0891,
        },
        {
            "method": "Frequency-aware V2",
            "n_source_images": 80,
            "mean_psnr_db": 28.2926,
            "std_psnr_db": 2.0510,
            "mean_ssim": 0.7561,
            "std_ssim": 0.0985,
        },
        {
            "method": "Spatial-only V2",
            "n_source_images": 80,
            "mean_psnr_db": 29.3272,
            "std_psnr_db": 2.3960,
            "mean_ssim": 0.7649,
            "std_ssim": 0.0949,
        },
    ]
)

BOOTSTRAP_SUMMARY_FALLBACK = pd.DataFrame(
    [
        {
            "comparison": "Frequency-aware V2 minus Spatial-only V2",
            "psnr_mean_difference_db": -1.0347,
            "psnr_ci_low_95": -1.1825,
            "psnr_ci_high_95": -0.8908,
            "psnr_images_a_better": 3,
            "psnr_images_compared": 80,
            "ssim_mean_difference": -0.0087,
            "ssim_ci_low_95": -0.0116,
            "ssim_ci_high_95": -0.0059,
            "ssim_images_a_better": 20,
            "ssim_images_compared": 80,
        },
        {
            "comparison": "Frequency-aware V2 minus Degraded input",
            "psnr_mean_difference_db": 4.3590,
            "psnr_ci_low_95": 4.0212,
            "psnr_ci_high_95": 4.6899,
            "psnr_images_a_better": 79,
            "psnr_images_compared": 80,
            "ssim_mean_difference": 0.3402,
            "ssim_ci_low_95": 0.3180,
            "ssim_ci_high_95": 0.3619,
            "ssim_images_a_better": 79,
            "ssim_images_compared": 80,
        },
        {
            "comparison": "Spatial-only V2 minus Degraded input",
            "psnr_mean_difference_db": 5.3937,
            "psnr_ci_low_95": 4.9957,
            "psnr_ci_high_95": 5.7933,
            "psnr_images_a_better": 80,
            "psnr_images_compared": 80,
            "ssim_mean_difference": 0.3489,
            "ssim_ci_low_95": 0.3267,
            "ssim_ci_high_95": 0.3705,
            "ssim_images_a_better": 80,
            "ssim_images_compared": 80,
        },
    ]
)


# -----------------------------------------------------------------------------
# Streamlit page setup and visual theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FreqDiff | Image Restoration Research Demo",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}
      .hero {padding: 1.35rem 1.5rem; border-radius: 18px;
             background: linear-gradient(115deg, rgba(23,48,82,.96), rgba(31,91,119,.92));
             color: #fff; margin-bottom: 1.0rem;}
      .hero h1 {font-size: 2.35rem; margin: 0 0 .25rem 0; color: #fff;}
      .hero p {font-size: 1.02rem; margin: 0; color: #e6f0f7;}
      .small-muted {color: #778397; font-size: .88rem;}
      .section-note {padding: .8rem 1rem; border-left: 4px solid #4e91b8;
                     background: rgba(100,140,170,.08); border-radius: 6px;}
      div[data-testid="stMetric"] {border: 1px solid rgba(120,140,160,.20);
          padding: .7rem .85rem; border-radius: 12px;}
      .status-pill {display:inline-block; padding:.25rem .65rem; border-radius:999px;
          font-size:.78rem; font-weight:650; margin-right:.4rem;
          background:rgba(54,160,110,.15); color:#2b9b68;}
      .warning-pill {display:inline-block; padding:.25rem .65rem; border-radius:999px;
          font-size:.78rem; font-weight:650; margin-right:.4rem;
          background:rgba(221,159,48,.17); color:#c58a24;}
    </style>
    <div class="hero">
      <h1>🌀 FreqDiff</h1>
      <p>Frequency-aware conditional diffusion for image restoration · Interactive research demonstration</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Device, precision, and cached model loading
# -----------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_amp_dtype(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "BF16"
    return None, "FP32"


@st.cache_resource(show_spinner=False)
def load_diffusion_model(model_name: str, checkpoint_path: str, device_name: str):
    """Load a trained V2 model once per Streamlit server session/cache key."""
    device = torch.device(device_name)
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    # Import from the user's existing FreqDiff project source tree.
    from src.models.diffusion_model import FrequencyAwareConditionalUNet
    from src.models.spatial_conditional_unet import SpatialConditionalUNet
    from src.models.diffusion_process import DiffusionProcess

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=True,
    )
    config = checkpoint.get(
        "model_config",
        {"base_channels": BASE_CHANNELS, "time_dim": TIME_DIM},
    )

    if model_name == "frequency":
        model = FrequencyAwareConditionalUNet(**config)
    elif model_name == "spatial":
        model = SpatialConditionalUNet(**config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    diffusion = DiffusionProcess(
        timesteps=DIFFUSION_STEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        device=device,
    )

    metadata = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "val_loss": checkpoint.get("val_loss", float("nan")),
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(checkpoint_file),
    }
    return model, diffusion, metadata


# -----------------------------------------------------------------------------
# Image loading, resizing, degradation, and metrics
# -----------------------------------------------------------------------------
def uploaded_to_rgb(uploaded_file, max_side: int) -> np.ndarray:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if max(width, height) > max_side:
        scale = max_side / float(max(width, height))
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def apply_synthetic_degradation(
    clean_rgb: np.ndarray,
    blur_sigma: float,
    noise_sigma_255: float,
    seed: int,
) -> np.ndarray:
    """Apply the V2-style Gaussian blur + additive Gaussian noise."""
    degraded = clean_rgb.astype(np.float32)
    if blur_sigma > 0:
        degraded = cv2.GaussianBlur(
            degraded,
            (7, 7),
            sigmaX=float(blur_sigma),
        )
    if noise_sigma_255 > 0:
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(
            0.0,
            float(noise_sigma_255),
            size=degraded.shape,
        ).astype(np.float32)
        degraded = degraded + noise
    return np.clip(degraded, 0, 255).astype(np.uint8)


def calculate_metrics(reference: np.ndarray, restored: np.ndarray):
    if reference.shape != restored.shape:
        raise ValueError("Reference and restored image dimensions must match.")
    ref = reference.astype(np.float32) / 255.0
    pred = restored.astype(np.float32) / 255.0
    psnr = peak_signal_noise_ratio(ref, pred, data_range=1.0)
    ssim = structural_similarity(
        ref,
        pred,
        data_range=1.0,
        channel_axis=2,
    )
    return float(psnr), float(ssim)


def png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def make_fft_visualization(rgb: np.ndarray) -> np.ndarray:
    image = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    spectrum = torch.fft.fft2(tensor, dim=(-2, -1), norm="ortho")
    spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
    magnitude = torch.log1p(torch.abs(spectrum)).mean(dim=0).numpy()
    low, high = np.percentile(magnitude, [2, 99.5])
    scaled = np.clip((magnitude - low) / max(high - low, 1e-8), 0, 1)
    gray = (scaled * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[:, :, ::-1]


# -----------------------------------------------------------------------------
# Matched DDIM inference: direct clean-image x0 prediction
# -----------------------------------------------------------------------------
def ddim_sample_x0(
    model: torch.nn.Module,
    diffusion,
    degraded_tile: torch.Tensor,
    initial_noise: torch.Tensor,
    sampling_steps: int,
    device: torch.device,
    amp_dtype,
) -> torch.Tensor:
    image = initial_noise.float().clone()
    batch_size = degraded_tile.shape[0]

    step_indices = sorted(
        torch.linspace(
            0,
            diffusion.timesteps - 1,
            sampling_steps,
            dtype=torch.float64,
        )
        .round()
        .long()
        .unique()
        .tolist(),
        reverse=True,
    )

    for index, step in enumerate(step_indices):
        timesteps = torch.full(
            (batch_size,),
            step,
            device=device,
            dtype=torch.long,
        )

        if amp_dtype is None:
            predicted_clean = model(image, degraded_tile, timesteps)
        else:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                predicted_clean = model(image, degraded_tile, timesteps)

        predicted_clean = predicted_clean.float().clamp(-1.0, 1.0)
        alpha_bar = diffusion.alpha_bars[step].float()
        predicted_noise = (
            image - torch.sqrt(alpha_bar) * predicted_clean
        ) / torch.sqrt((1.0 - alpha_bar).clamp_min(1e-8))

        if index == len(step_indices) - 1:
            image = predicted_clean
        else:
            next_step = step_indices[index + 1]
            next_alpha_bar = diffusion.alpha_bars[next_step].float()
            image = (
                torch.sqrt(next_alpha_bar) * predicted_clean
                + torch.sqrt((1.0 - next_alpha_bar).clamp_min(0.0))
                * predicted_noise
            )

    return image.clamp(-1.0, 1.0)


def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    count = int(math.ceil((length - tile_size) / stride)) + 1
    return [i * stride for i in range(count)]


def restore_tiled(
    model: torch.nn.Module,
    diffusion,
    degraded_rgb: np.ndarray,
    sampling_steps: int,
    seed: int,
    device: torch.device,
    amp_dtype,
    overlap: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Restore any image size using overlapping 128x128 patches with blending."""
    if not 0 <= overlap < TILE_SIZE:
        raise ValueError("Tile overlap must be between 0 and 127 pixels.")

    original_height, original_width = degraded_rgb.shape[:2]
    stride = TILE_SIZE - overlap

    # Build dimensions that are exactly covered by the tile grid.
    rows_for_grid = tile_positions(original_height, TILE_SIZE, stride)
    cols_for_grid = tile_positions(original_width, TILE_SIZE, stride)
    padded_height = rows_for_grid[-1] + TILE_SIZE
    padded_width = cols_for_grid[-1] + TILE_SIZE

    image_tensor = torch.from_numpy(
        np.ascontiguousarray(degraded_rgb.transpose(2, 0, 1))
    ).float().unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)

    pad_bottom = padded_height - original_height
    pad_right = padded_width - original_width
    if pad_bottom or pad_right:
        image_tensor = F.pad(
            image_tensor,
            (0, pad_right, 0, pad_bottom),
            mode="replicate",
        )

    # Smooth overlap weights reduce visible seams between restored tiles.
    if overlap > 0:
        one_d = torch.hann_window(TILE_SIZE, periodic=False, device=device)
        one_d = one_d.clamp_min(0.05)
        weight = torch.outer(one_d, one_d)[None, None, :, :]
    else:
        weight = torch.ones((1, 1, TILE_SIZE, TILE_SIZE), device=device)

    accumulation = torch.zeros(
        (1, 3, padded_height, padded_width),
        device=device,
        dtype=torch.float32,
    )
    weight_sum = torch.zeros(
        (1, 1, padded_height, padded_width),
        device=device,
        dtype=torch.float32,
    )

    coordinates = [(top, left) for top in rows_for_grid for left in cols_for_grid]
    total_tiles = len(coordinates)

    model.eval()
    with torch.inference_mode():
        for tile_index, (top, left) in enumerate(coordinates):
            degraded_tile = image_tensor[
                :, :, top:top + TILE_SIZE, left:left + TILE_SIZE
            ]
            degraded_model = degraded_tile * 2.0 - 1.0

            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) + tile_index)
            initial_noise = torch.randn(
                degraded_model.shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )

            restored = ddim_sample_x0(
                model=model,
                diffusion=diffusion,
                degraded_tile=degraded_model,
                initial_noise=initial_noise,
                sampling_steps=sampling_steps,
                device=device,
                amp_dtype=amp_dtype,
            )
            restored = ((restored + 1.0) / 2.0).clamp(0.0, 1.0)

            accumulation[
                :, :, top:top + TILE_SIZE, left:left + TILE_SIZE
            ] += restored * weight
            weight_sum[
                :, :, top:top + TILE_SIZE, left:left + TILE_SIZE
            ] += weight

            if progress_callback is not None:
                progress_callback(tile_index + 1, total_tiles)

    restored_full = accumulation / weight_sum.clamp_min(1e-8)
    restored_full = restored_full[:, :, :original_height, :original_width]
    restored_np = (
        restored_full[0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    return np.clip(restored_np * 255.0, 0, 255).astype(np.uint8)


def classical_filter(rgb: np.ndarray, method: str) -> np.ndarray:
    if method == "Median filter (5x5)":
        return cv2.medianBlur(rgb, 5)
    if method == "Gaussian smoothing (sigma=1.0)":
        return cv2.GaussianBlur(rgb, (7, 7), sigmaX=1.0)
    raise ValueError(f"Unknown classical filter: {method}")


# -----------------------------------------------------------------------------
# Research results and interpretation
# -----------------------------------------------------------------------------
def load_benchmark_data():
    summary_path = EVALUATION_DIR / "summary.csv"
    bootstrap_path = EVALUATION_DIR / "paired_bootstrap.csv"
    per_image_path = EVALUATION_DIR / "per_image_metrics.csv"

    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        summary_source = "Local V2 evaluation CSV"
    else:
        summary = BENCHMARK_SUMMARY_FALLBACK.copy()
        summary_source = "Verified V2 values embedded in dashboard"

    if bootstrap_path.is_file():
        bootstrap = pd.read_csv(bootstrap_path)
    else:
        bootstrap = BOOTSTRAP_SUMMARY_FALLBACK.copy()

    per_image = pd.read_csv(per_image_path) if per_image_path.is_file() else None
    return summary, bootstrap, per_image, summary_source


def load_training_history(model_name: str):
    path = HISTORY_DIR / f"{model_name}_training_history.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        return pd.DataFrame(json.load(file))


def row_for_method(summary: pd.DataFrame, method: str):
    matched = summary[summary["method"].astype(str) == method]
    if matched.empty:
        return None
    return matched.iloc[0]


def render_research_interpretation(summary: pd.DataFrame, bootstrap: pd.DataFrame):
    degraded = row_for_method(summary, "Degraded input")
    frequency = row_for_method(summary, "Frequency-aware V2")
    spatial = row_for_method(summary, "Spatial-only V2")

    if degraded is None or frequency is None or spatial is None:
        st.warning("The benchmark table is missing one or more expected methods.")
        return

    freq_spatial = bootstrap[
        bootstrap["comparison"].astype(str)
        == "Frequency-aware V2 minus Spatial-only V2"
    ]
    if freq_spatial.empty:
        st.info("Paired-bootstrap comparison row is unavailable.")
        return
    paired = freq_spatial.iloc[0]

    psnr_gap = float(spatial["mean_psnr_db"] - frequency["mean_psnr_db"])
    ssim_gap = float(spatial["mean_ssim"] - frequency["mean_ssim"])

    st.markdown("### What the results mean")
    st.write(
        f"Both diffusion models improved the degraded-input baseline. "
        f"The frequency-aware model gained **{frequency['mean_psnr_db'] - degraded['mean_psnr_db']:.3f} dB PSNR** "
        f"and **{frequency['mean_ssim'] - degraded['mean_ssim']:.4f} SSIM** on average. "
        f"The spatial-only model gained **{spatial['mean_psnr_db'] - degraded['mean_psnr_db']:.3f} dB PSNR** "
        f"and **{spatial['mean_ssim'] - degraded['mean_ssim']:.4f} SSIM**."
    )

    ci_psnr_low = float(paired["psnr_ci_low_95"])
    ci_psnr_high = float(paired["psnr_ci_high_95"])
    ci_ssim_low = float(paired["ssim_ci_low_95"])
    ci_ssim_high = float(paired["ssim_ci_high_95"])

    if ci_psnr_high < 0 and ci_ssim_high < 0:
        st.warning(
            f"On this held-out synthetic-degradation benchmark, the spatial-only model "
            f"outperformed the frequency-aware model on both metrics: **+{psnr_gap:.4f} dB PSNR** "
            f"and **+{ssim_gap:.4f} SSIM** (spatial minus frequency). The paired 95% bootstrap "
            f"intervals for frequency minus spatial were [{ci_psnr_low:.4f}, {ci_psnr_high:.4f}] dB "
            f"for PSNR and [{ci_ssim_low:.4f}, {ci_ssim_high:.4f}] for SSIM; both intervals are below zero."
        )
    else:
        st.info(
            "The paired intervals do not both lie entirely below zero; interpret the relative ranking cautiously."
        )

    st.markdown(
        "**Research interpretation:** the experiments show that conditional diffusion improves "
        "restoration under the tested synthetic blur-and-noise protocol, but the current V2 results "
        "do not demonstrate an advantage from adding the frequency-conditioning branch. This is a "
        "useful ablation finding, not evidence that frequency information is never useful."
    )


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Demo controls")
    st.caption("Choose an inference mode and image-processing settings.")

    model_choice = st.selectbox(
        "Model selection",
        [
            "Compare both V2 models",
            "Frequency-aware V2 diffusion",
            "Spatial-only V2 diffusion",
            "Classical median filter",
            "Classical Gaussian smoothing",
        ],
        index=0,
    )

    input_mode = st.radio(
        "Input image type",
        [
            "Clean reference — generate synthetic degradation",
            "Already-degraded image",
        ],
        index=0,
    )

    st.divider()
    st.markdown("**Synthetic degradation**")
    blur_sigma = st.slider(
        "Gaussian blur sigma",
        min_value=0.0,
        max_value=2.5,
        value=1.0,
        step=0.1,
        help="V2 benchmark used sigma=1.0 with a 7x7 Gaussian kernel.",
    )
    noise_sigma = st.slider(
        "Gaussian noise standard deviation (pixel levels)",
        min_value=0.0,
        max_value=40.0,
        value=15.0,
        step=1.0,
        help="V2 benchmark used Gaussian noise with standard deviation 15/255.",
    )
    degradation_seed = st.number_input(
        "Degradation random seed",
        min_value=0,
        max_value=2_000_000_000,
        value=2026,
        step=1,
    )

    st.divider()
    st.markdown("**Inference settings**")
    sampling_steps = st.select_slider(
        "DDIM sampling steps",
        options=[10, 15, 25, 35, 50],
        value=25,
        help="The official V2 benchmark used 25 DDIM steps. Fewer steps are faster but are not benchmark-identical.",
    )
    max_side = st.select_slider(
        "Maximum image side (pixels)",
        options=[384, 512, 768, 1024],
        value=512,
        help="Large images are downscaled proportionally before tiled inference.",
    )
    tile_overlap = st.select_slider(
        "Tile overlap (pixels)",
        options=[0, 16, 32, 48, 64],
        value=32,
        help="Overlapping tiles are blended to reduce visible seams.",
    )
    inference_seed = st.number_input(
        "Initial diffusion noise seed",
        min_value=0,
        max_value=2_000_000_000,
        value=30000,
        step=1,
    )

    st.divider()
    device = get_device()
    amp_dtype, precision_label = get_amp_dtype(device)
    st.caption(f"Runtime device: **{device}**")
    if device.type == "cuda":
        st.caption(f"GPU: **{torch.cuda.get_device_name(0)}**")
    st.caption(f"Inference precision: **{precision_label}**")

    freq_exists = (CHECKPOINT_DIR / "frequency_best.pth").is_file()
    spatial_exists = (CHECKPOINT_DIR / "spatial_best.pth").is_file()
    st.caption(
        f"Frequency checkpoint: {'✅ found' if freq_exists else '⚠️ missing'}"
    )
    st.caption(
        f"Spatial checkpoint: {'✅ found' if spatial_exists else '⚠️ missing'}"
    )


# -----------------------------------------------------------------------------
# Main page and tabs
# -----------------------------------------------------------------------------
summary_df, bootstrap_df, per_image_df, summary_source = load_benchmark_data()

st.markdown(
    "<span class='status-pill'>V2 experiments completed</span>"
    "<span class='status-pill'>80 held-out images</span>"
    "<span class='status-pill'>720 evaluation patches</span>"
    "<span class='warning-pill'>Synthetic degradation benchmark</span>",
    unsafe_allow_html=True,
)

live_tab, results_tab, architecture_tab, methodology_tab = st.tabs(
    [
        "🖼️ Live restoration",
        "📊 Research results",
        "🧠 Architecture",
        "📘 Methodology & limitations",
    ]
)


# -----------------------------------------------------------------------------
# Live restoration tab
# -----------------------------------------------------------------------------
with live_tab:
    st.subheader("Restore an image")
    st.markdown(
        "Upload a clean reference image to generate controlled blur/noise, or upload an already-degraded image. "
        "The diffusion model operates on overlapping 128×128 patches and blends them back together."
    )

    input_col, ref_col = st.columns([1.25, 1])
    with input_col:
        if input_mode.startswith("Clean reference"):
            uploaded_image = st.file_uploader(
                "Upload a clean/reference image",
                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                key="clean_upload",
            )
        else:
            uploaded_image = st.file_uploader(
                "Upload an already-degraded image",
                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                key="degraded_upload",
            )

    with ref_col:
        if input_mode.startswith("Already-degraded"):
            reference_upload = st.file_uploader(
                "Optional clean ground-truth reference (same scene and dimensions)",
                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                key="reference_upload",
            )
        else:
            reference_upload = None
            st.info(
                "In clean-reference mode, the uploaded image itself is the ground truth. "
                "The app will generate its degraded input using the controls in the sidebar."
            )

    if uploaded_image is not None:
        try:
            source_rgb = uploaded_to_rgb(uploaded_image, int(max_side))
        except Exception as exc:
            st.error(f"Could not read the uploaded image: {exc}")
            source_rgb = None

        if source_rgb is not None:
            if input_mode.startswith("Clean reference"):
                clean_reference = source_rgb.copy()
                degraded_rgb = apply_synthetic_degradation(
                    clean_reference,
                    blur_sigma=float(blur_sigma),
                    noise_sigma_255=float(noise_sigma),
                    seed=int(degradation_seed),
                )
                st.caption(
                    f"Synthetic degradation: Gaussian blur σ={blur_sigma:.1f}; "
                    f"Gaussian noise σ={noise_sigma:.0f}/255; seed={degradation_seed}."
                )
            else:
                degraded_rgb = source_rgb.copy()
                clean_reference = None
                if reference_upload is not None:
                    try:
                        candidate_reference = uploaded_to_rgb(
                            reference_upload,
                            int(max_side),
                        )
                        if candidate_reference.shape != degraded_rgb.shape:
                            st.error(
                                "The reference image dimensions do not match the degraded input "
                                "after resizing. Upload a matching image pair; otherwise PSNR/SSIM "
                                "will be omitted."
                            )
                        else:
                            clean_reference = candidate_reference
                    except Exception as exc:
                        st.error(f"Could not read reference image: {exc}")

            show_cols = st.columns(3)
            with show_cols[0]:
                st.markdown("**Input / clean reference**" if input_mode.startswith("Clean") else "**Uploaded degraded input**")
                st.image(source_rgb, use_container_width=True)
            with show_cols[1]:
                st.markdown("**Model conditioning image**")
                st.image(degraded_rgb, use_container_width=True)
            with show_cols[2]:
                st.markdown("**Fourier magnitude preview**")
                st.image(make_fft_visualization(degraded_rgb), use_container_width=True)

            if st.button("🚀 Run restoration", type="primary", use_container_width=True):
                requested_models = []
                if model_choice in (
                    "Compare both V2 models",
                    "Frequency-aware V2 diffusion",
                ):
                    requested_models.append("frequency")
                if model_choice in (
                    "Compare both V2 models",
                    "Spatial-only V2 diffusion",
                ):
                    requested_models.append("spatial")

                if "frequency" in requested_models and not freq_exists:
                    st.error(
                        "Missing frequency checkpoint. Expected: "
                        "checkpoints/v2_corrected/frequency_best.pth"
                    )
                elif "spatial" in requested_models and not spatial_exists:
                    st.error(
                        "Missing spatial checkpoint. Expected: "
                        "checkpoints/v2_corrected/spatial_best.pth"
                    )
                else:
                    run_outputs = {}
                    total_steps = len(requested_models) if requested_models else 1
                    overall_progress = st.progress(0.0)
                    progress_text = st.empty()
                    run_start = time.perf_counter()

                    try:
                        if requested_models:
                            for model_number, model_name in enumerate(requested_models):
                                model_label = (
                                    "Frequency-aware V2"
                                    if model_name == "frequency"
                                    else "Spatial-only V2"
                                )
                                progress_text.info(
                                    f"Loading {model_label} checkpoint and preparing tiled inference..."
                                )
                                model, diffusion, metadata = load_diffusion_model(
                                    model_name,
                                    str(CHECKPOINT_DIR / f"{model_name}_best.pth"),
                                    str(device),
                                )
                                tile_progress = st.progress(0.0)
                                tile_status = st.empty()

                                def tile_callback(done, total, base=model_number):
                                    local_fraction = done / max(total, 1)
                                    overall_progress.progress(
                                        min(1.0, (base + local_fraction) / total_steps)
                                    )
                                    tile_progress.progress(local_fraction)
                                    tile_status.caption(
                                        f"{model_label}: tile {done}/{total}"
                                    )

                                restored = restore_tiled(
                                    model=model,
                                    diffusion=diffusion,
                                    degraded_rgb=degraded_rgb,
                                    sampling_steps=int(sampling_steps),
                                    seed=int(inference_seed),
                                    device=device,
                                    amp_dtype=amp_dtype,
                                    overlap=int(tile_overlap),
                                    progress_callback=tile_callback,
                                )
                                run_outputs[model_name] = restored
                                tile_progress.empty()
                                tile_status.empty()
                                del model, diffusion

                        elif model_choice == "Classical median filter":
                            progress_text.info("Applying the median-filter baseline...")
                            run_outputs["classical"] = classical_filter(
                                degraded_rgb,
                                "Median filter (5x5)",
                            )
                            overall_progress.progress(1.0)
                        else:
                            progress_text.info("Applying Gaussian smoothing...")
                            run_outputs["classical"] = classical_filter(
                                degraded_rgb,
                                "Gaussian smoothing (sigma=1.0)",
                            )
                            overall_progress.progress(1.0)

                        st.session_state["freqdiff_last_run"] = {
                            "input": source_rgb,
                            "degraded": degraded_rgb,
                            "reference": clean_reference,
                            "outputs": run_outputs,
                            "model_choice": model_choice,
                            "sampling_steps": int(sampling_steps),
                            "seconds": time.perf_counter() - run_start,
                            "seed": int(inference_seed),
                        }
                        progress_text.success(
                            f"Restoration completed in {time.perf_counter() - run_start:.1f} seconds."
                        )
                    except Exception as exc:
                        overall_progress.empty()
                        st.exception(exc)

    # Display persisted result after Streamlit reruns.
    last_run = st.session_state.get("freqdiff_last_run")
    if last_run is not None:
        st.divider()
        st.subheader("Restoration output")
        st.caption(
            f"Last run: {last_run['model_choice']} · "
            f"{last_run['sampling_steps']} DDIM steps · "
            f"{last_run['seconds']:.1f} seconds"
        )

        outputs = last_run["outputs"]
        degraded_saved = last_run["degraded"]
        reference_saved = last_run["reference"]

        if "frequency" in outputs and "spatial" in outputs:
            result_cols = st.columns(3)
            with result_cols[0]:
                st.markdown("**Degraded input**")
                st.image(degraded_saved, use_container_width=True)
            with result_cols[1]:
                st.markdown("**Frequency-aware V2**")
                st.image(outputs["frequency"], use_container_width=True)
            with result_cols[2]:
                st.markdown("**Spatial-only V2**")
                st.image(outputs["spatial"], use_container_width=True)
        else:
            key = "frequency" if "frequency" in outputs else "spatial" if "spatial" in outputs else "classical"
            label = {
                "frequency": "Frequency-aware V2 output",
                "spatial": "Spatial-only V2 output",
                "classical": "Classical filter output (illustrative comparator)",
            }[key]
            result_cols = st.columns(2)
            with result_cols[0]:
                st.markdown("**Degraded input**")
                st.image(degraded_saved, use_container_width=True)
            with result_cols[1]:
                st.markdown(f"**{label}**")
                st.image(outputs[key], use_container_width=True)

        # Live whole-image metrics are intentionally labeled separately from
        # the 80-image / 9-patch official benchmark metrics.
        if reference_saved is not None:
            st.markdown("### Live image metrics (uploaded image only)")
            st.caption(
                "These metrics are computed over this uploaded image after any proportional resize. "
                "They are not the official held-out benchmark averages."
            )
            metric_rows = []
            degraded_psnr, degraded_ssim = calculate_metrics(
                reference_saved,
                degraded_saved,
            )
            metric_rows.append(
                {
                    "Output": "Degraded input",
                    "PSNR (dB)": degraded_psnr,
                    "SSIM": degraded_ssim,
                }
            )
            for key, output in outputs.items():
                psnr_value, ssim_value = calculate_metrics(
                    reference_saved,
                    output,
                )
                label = {
                    "frequency": "Frequency-aware V2",
                    "spatial": "Spatial-only V2",
                    "classical": "Classical filter",
                }[key]
                metric_rows.append(
                    {
                        "Output": label,
                        "PSNR (dB)": psnr_value,
                        "SSIM": ssim_value,
                    }
                )
            live_metrics = pd.DataFrame(metric_rows)
            metric_cols = st.columns(len(metric_rows))
            for idx, row in live_metrics.iterrows():
                with metric_cols[idx]:
                    st.metric(
                        f"{row['Output']} · PSNR",
                        f"{row['PSNR (dB)']:.2f} dB",
                    )
                    st.metric(
                        f"{row['Output']} · SSIM",
                        f"{row['SSIM']:.4f}",
                    )
            st.dataframe(live_metrics, use_container_width=True, hide_index=True)

            # Difference image for the first available model output.
            first_output_key = next(iter(outputs))
            difference = np.abs(
                reference_saved.astype(np.float32)
                - outputs[first_output_key].astype(np.float32)
            ).mean(axis=2)
            difference = np.clip(difference / 80.0, 0, 1)
            diff_rgb = cv2.applyColorMap(
                (difference * 255).astype(np.uint8),
                cv2.COLORMAP_INFERNO,
            )[:, :, ::-1]
            with st.expander("View pixel-wise error map"):
                st.image(
                    diff_rgb,
                    caption="Absolute RGB error relative to the uploaded reference (bright = larger error)",
                    use_container_width=True,
                )
        else:
            st.info(
                "No clean reference was supplied. The output is shown visually, but PSNR and SSIM "
                "cannot be calculated without ground truth."
            )

        st.markdown("### Download restored image")
        download_cols = st.columns(max(1, len(outputs)))
        for idx, (key, output) in enumerate(outputs.items()):
            label = {
                "frequency": "Download frequency-aware PNG",
                "spatial": "Download spatial-only PNG",
                "classical": "Download classical-filter PNG",
            }[key]
            filename = f"freqdiff_{key}_restored.png"
            with download_cols[idx]:
                st.download_button(
                    label,
                    data=png_bytes(output),
                    file_name=filename,
                    mime="image/png",
                    use_container_width=True,
                    key=f"download_{key}",
                )


# -----------------------------------------------------------------------------
# Research results tab: benchmark, charts, confidence intervals, interpretations
# -----------------------------------------------------------------------------
with results_tab:
    st.subheader("V2 held-out benchmark")
    st.caption(f"Data source: {summary_source}. Official evaluation files are read from {EVALUATION_DIR} when present.")

    degraded_row = row_for_method(summary_df, "Degraded input")
    frequency_row = row_for_method(summary_df, "Frequency-aware V2")
    spatial_row = row_for_method(summary_df, "Spatial-only V2")

    if all(row is not None for row in [degraded_row, frequency_row, spatial_row]):
        kpis = st.columns(4)
        kpis[0].metric("Test source images", "80")
        kpis[1].metric("Test patches", "720")
        kpis[2].metric(
            "Spatial PSNR gain vs input",
            f"+{spatial_row['mean_psnr_db'] - degraded_row['mean_psnr_db']:.2f} dB",
        )
        kpis[3].metric(
            "Spatial SSIM gain vs input",
            f"+{spatial_row['mean_ssim'] - degraded_row['mean_ssim']:.3f}",
        )

    st.markdown("### Benchmark summary")
    st.dataframe(
        summary_df.style.format(
            {
                "mean_psnr_db": "{:.4f}",
                "std_psnr_db": "{:.4f}",
                "mean_ssim": "{:.4f}",
                "std_ssim": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig_psnr = px.bar(
            summary_df,
            x="method",
            y="mean_psnr_db",
            error_y="std_psnr_db",
            title="Mean PSNR by method (error bars: image-level SD)",
            labels={
                "method": "Method",
                "mean_psnr_db": "PSNR (dB)",
            },
            color="method",
        )
        fig_psnr.update_layout(showlegend=False, xaxis_tickangle=-15)
        st.plotly_chart(fig_psnr, use_container_width=True)

    with chart_right:
        fig_ssim = px.bar(
            summary_df,
            x="method",
            y="mean_ssim",
            error_y="std_ssim",
            title="Mean SSIM by method (error bars: image-level SD)",
            labels={
                "method": "Method",
                "mean_ssim": "SSIM",
            },
            color="method",
        )
        fig_ssim.update_layout(showlegend=False, xaxis_tickangle=-15)
        st.plotly_chart(fig_ssim, use_container_width=True)

    st.markdown("### Paired bootstrap comparisons")
    st.caption("Differences are method A minus method B. The resampling unit is the source image (n=80), not each individual patch.")
    st.dataframe(bootstrap_df, use_container_width=True, hide_index=True)

    freq_spatial = bootstrap_df[
        bootstrap_df["comparison"].astype(str)
        == "Frequency-aware V2 minus Spatial-only V2"
    ]
    if not freq_spatial.empty:
        paired = freq_spatial.iloc[0]
        ci_chart_cols = st.columns(2)
        with ci_chart_cols[0]:
            fig_ci_psnr = go.Figure()
            fig_ci_psnr.add_trace(
                go.Scatter(
                    x=[paired["psnr_mean_difference_db"]],
                    y=["Frequency − Spatial"],
                    mode="markers",
                    marker={"size": 13},
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": [paired["psnr_ci_high_95"] - paired["psnr_mean_difference_db"]],
                        "arrayminus": [paired["psnr_mean_difference_db"] - paired["psnr_ci_low_95"]],
                        "thickness": 2,
                        "width": 8,
                    },
                    name="Mean difference and 95% CI",
                )
            )
            fig_ci_psnr.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_ci_psnr.update_layout(
                title="Paired PSNR difference (95% bootstrap CI)",
                xaxis_title="Frequency − Spatial (dB); positive favors frequency",
                yaxis_title="",
                showlegend=False,
            )
            st.plotly_chart(fig_ci_psnr, use_container_width=True)

        with ci_chart_cols[1]:
            fig_ci_ssim = go.Figure()
            fig_ci_ssim.add_trace(
                go.Scatter(
                    x=[paired["ssim_mean_difference"]],
                    y=["Frequency − Spatial"],
                    mode="markers",
                    marker={"size": 13},
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": [paired["ssim_ci_high_95"] - paired["ssim_mean_difference"]],
                        "arrayminus": [paired["ssim_mean_difference"] - paired["ssim_ci_low_95"]],
                        "thickness": 2,
                        "width": 8,
                    },
                    name="Mean difference and 95% CI",
                )
            )
            fig_ci_ssim.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_ci_ssim.update_layout(
                title="Paired SSIM difference (95% bootstrap CI)",
                xaxis_title="Frequency − Spatial; positive favors frequency",
                yaxis_title="",
                showlegend=False,
            )
            st.plotly_chart(fig_ci_ssim, use_container_width=True)

    if per_image_df is not None and not per_image_df.empty:
        st.markdown("### Per-image metric distributions")
        if all(
            col in per_image_df.columns
            for col in ["frequency_psnr", "spatial_psnr", "degraded_psnr"]
        ):
            psnr_long = per_image_df[
                ["degraded_psnr", "frequency_psnr", "spatial_psnr"]
            ].melt(var_name="method", value_name="PSNR (dB)")
            psnr_long["method"] = psnr_long["method"].map(
                {
                    "degraded_psnr": "Degraded input",
                    "frequency_psnr": "Frequency-aware V2",
                    "spatial_psnr": "Spatial-only V2",
                }
            )
            st.plotly_chart(
                px.box(
                    psnr_long,
                    x="method",
                    y="PSNR (dB)",
                    color="method",
                    points="all",
                    title="Per-source-image PSNR distribution",
                ).update_layout(showlegend=False, xaxis_tickangle=-15),
                use_container_width=True,
            )
        if all(
            col in per_image_df.columns
            for col in ["frequency_ssim", "spatial_ssim", "degraded_ssim"]
        ):
            ssim_long = per_image_df[
                ["degraded_ssim", "frequency_ssim", "spatial_ssim"]
            ].melt(var_name="method", value_name="SSIM")
            ssim_long["method"] = ssim_long["method"].map(
                {
                    "degraded_ssim": "Degraded input",
                    "frequency_ssim": "Frequency-aware V2",
                    "spatial_ssim": "Spatial-only V2",
                }
            )
            st.plotly_chart(
                px.box(
                    ssim_long,
                    x="method",
                    y="SSIM",
                    color="method",
                    points="all",
                    title="Per-source-image SSIM distribution",
                ).update_layout(showlegend=False, xaxis_tickangle=-15),
                use_container_width=True,
            )
    else:
        st.info(
            "Per-image metric CSV not found. Copy results/v2_corrected/evaluation/per_image_metrics.csv "
            "from your project to enable per-image boxplots and scatter plots."
        )

    st.markdown("### Training curves")
    training_cols = st.columns(2)
    for col, model_name, title in [
        (training_cols[0], "frequency", "Frequency-aware V2 training"),
        (training_cols[1], "spatial", "Spatial-only V2 training"),
    ]:
        history = load_training_history(model_name)
        with col:
            st.markdown(f"**{title}**")
            if history is not None and not history.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=history["epoch"],
                        y=history["train_min_snr_x0_loss"],
                        mode="lines+markers",
                        name="Training loss",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=history["epoch"],
                        y=history["val_min_snr_x0_loss"],
                        mode="lines+markers",
                        name="Validation loss",
                    )
                )
                fig.update_layout(
                    xaxis_title="Epoch",
                    yaxis_title="Min-SNR x0 loss",
                    legend_title="",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(
                    f"Training history not found: {HISTORY_DIR / (model_name + '_training_history.json')}"
                )

    with st.expander("Automatic result interpretation"):
        render_research_interpretation(summary_df, bootstrap_df)


# -----------------------------------------------------------------------------
# Architecture tab
# -----------------------------------------------------------------------------
with architecture_tab:
    st.subheader("Model architecture and design")
    st.write(
        "Both V2 models are conditional encoder–decoder U-Nets. They receive the noisy target "
        "image, the degraded observation, and a diffusion timestep, and predict the clean image x₀. "
        "The frequency-aware model additionally processes Fourier-magnitude features from the degraded input."
    )

    arch_left, arch_right = st.columns(2)
    with arch_left:
        st.markdown("#### Frequency-aware conditional U-Net")
        st.graphviz_chart(
            """
            digraph G {
              rankdir=TB; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#e8f1fa", color="#53799c", fontname="Arial"];
              degraded [label="Degraded RGB input"];
              fft [label="2D FFT → shift → log magnitude"];
              cond [label="Frequency conditioner CNN"];
              noisy [label="Noisy image x_t + timestep t"];
              unet [label="Conditional U-Net encoder / bottleneck / decoder"];
              fuse [label="Frequency feature injection"];
              out [label="Predicted clean image x₀"];
              degraded -> fft -> cond -> fuse;
              noisy -> unet -> fuse;
              fuse -> out;
            }
            """,
            use_container_width=True,
        )

    with arch_right:
        st.markdown("#### Spatial-only conditional U-Net")
        st.graphviz_chart(
            """
            digraph G {
              rankdir=TB; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#edf3ed", color="#64856b", fontname="Arial"];
              degraded [label="Degraded RGB input"];
              noisy [label="Noisy image x_t + timestep t"];
              unet [label="Conditional U-Net encoder / bottleneck / decoder"];
              out [label="Predicted clean image x₀"];
              degraded -> unet;
              noisy -> unet;
              unet -> out;
            }
            """,
            use_container_width=True,
        )

    st.markdown("### Model complexity")
    parameter_df = pd.DataFrame(
        [
            {"Model": "Frequency-aware V2", "Parameters": 1_293_443},
            {"Model": "Spatial-only V2", "Parameters": 1_287_811},
        ]
    )
    pcols = st.columns(3)
    pcols[0].metric("Frequency-aware parameters", "1.293 M")
    pcols[1].metric("Spatial-only parameters", "1.288 M")
    pcols[2].metric("Parameter difference", "+5,632 (+0.44%)")
    st.plotly_chart(
        px.bar(
            parameter_df,
            x="Model",
            y="Parameters",
            color="Model",
            title="Trainable parameter count",
        ).update_layout(showlegend=False),
        use_container_width=True,
    )

    st.markdown("### Training objective")
    st.latex(r"w_t = \min(\mathrm{SNR}_t, \gamma), \qquad \gamma=5")
    st.write(
        "The models are trained for direct clean-image x₀ prediction. The V2 trainer uses the standard "
        "Min-SNR weighting for x₀ prediction, with γ=5. This differs from the custom SNR-normalized "
        "weighting used in the earlier V1 training scripts."
    )


# -----------------------------------------------------------------------------
# Methodology and limitations tab
# -----------------------------------------------------------------------------
with methodology_tab:
    st.subheader("Experimental protocol")

    details = [
        ("Dataset", "DIV2K training-image collection, split at the source-image level."),
        ("Source-image split", "640 training / 80 validation / 80 held-out test images."),
        ("Training patches", "5,120 patches per epoch: 8 patches per each of 640 training images."),
        ("Validation/test patches", "3×3 deterministic native-resolution grid: 9 patches per image."),
        ("Patch size", "128×128 RGB."),
        ("Evaluation degradation", "7×7 Gaussian blur with σ=1.0 and additive Gaussian noise σ=15/255."),
        ("Diffusion process", "1,000 linear beta steps, β from 1e-4 to 0.02."),
        ("Sampler", "Deterministic DDIM with 25 steps for the reported benchmark."),
        ("Training duration", "5 epochs per V2 model; validation loss was still improving at epoch 5."),
        ("Evaluation units", "9 patch metrics averaged within each of 80 source images."),
        ("Uncertainty analysis", "10,000 paired bootstrap resamples of the 80 source-image differences."),
    ]
    st.table(pd.DataFrame(details, columns=["Item", "Protocol"]))

    st.markdown("### Interpretation and limits")
    st.markdown(
        "- The reported benchmark evaluates **synthetic blur and Gaussian noise**, not real-world camera degradations.\n"
        "- The 80 held-out images are from the DIV2K training-image collection split, not the official DIV2K validation set.\n"
        "- Five epochs is a compute-constrained training schedule; the decreasing validation loss does not establish convergence.\n"
        "- The V2 comparison includes the two diffusion architectures and degraded input. V1 CNN/classical results were generated under a different evaluation protocol and must not be merged into the V2 table.\n"
        "- PSNR/SSIM shown in the live upload demo are whole-image metrics for that user-provided pair; they are not directly interchangeable with the official patch-aggregated benchmark.\n"
        "- Without a clean reference image, objective PSNR and SSIM cannot be calculated. Visual output alone is not a quantitative quality claim."
    )

    st.markdown("### Reproducibility files expected in the project")
    expected_files = [
        "checkpoints/v2_corrected/frequency_best.pth",
        "checkpoints/v2_corrected/spatial_best.pth",
        "results/v2_corrected/frequency_training_history.json",
        "results/v2_corrected/spatial_training_history.json",
        "results/v2_corrected/evaluation/summary.csv",
        "results/v2_corrected/evaluation/per_image_metrics.csv",
        "results/v2_corrected/evaluation/paired_bootstrap.csv",
        "results/v2_corrected/evaluation/evaluation_protocol.json",
    ]
    status_rows = []
    for relative in expected_files:
        file_path = ROOT / relative
        status_rows.append(
            {
                "File": relative,
                "Status": "Found" if file_path.is_file() else "Not found",
            }
        )
    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "FreqDiff research demonstration · V2 benchmark results are stored experiment results; "
    "live restoration is an interactive inference demonstration. The app does not claim real-world validation."
)
