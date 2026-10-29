import os
import io
import json
import re
import sys
from pathlib import Path

# Resolve the project root so the app can import project modules regardless of
# where Streamlit is started from. This is important in a capstone project where
# the working directory may not be the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# Import the shared model-build and explainability functions from src/gradcam.
# Keeping the model logic in one place avoids duplication and makes the frontend
# consistent with the actual training implementation.
from src.gradcam import (
    ARCHS,
    SHARED_CONFIG,
    load_model,
    grad_cam_pp,
    layer_cam,
    overlay_heatmap,
    get_transform,
    preprocess_fundus,
    _corn_conditional_to_cumulative,
    _corn_cumulative_to_class_probs,
)

# --------------------------------------------------------------------------
# PROJECT-SPECIFIC CONFIG
# --------------------------------------------------------------------------
# These paths tell the app which saved checkpoints to use for each model. The
# app expects the checkpoints to already exist before it can visualize results.

CHECKPOINT_PATHS = {
    "efficientnet": str(PROJECT_ROOT / "models" / "efficientnet_b0_qwk_best.pt"),
    "resnet": str(PROJECT_ROOT / "models" / "resnet18_corn_best.pt"),
    "convnext": str(PROJECT_ROOT / "models" / "convnext_tiny_corn_best.pt"),
}
ARCH_ORDER = ["efficientnet", "resnet", "convnext"]
ARCH_DISPLAY = {
    "efficientnet": "EfficientNet-B0 + QWK",
    "resnet": "ResNet-18 + CORN",
    "convnext": "ConvNeXt-Tiny + CORN",
}
ARCH_SHORT = {"efficientnet": "EffNet-B0", "resnet": "ResNet-18", "convnext": "ConvNeXt-Tiny"}
ARCH_ACCENT = {"efficientnet": "#8FB8FF", "resnet": "#C8A6FF", "convnext": "#6FE3D9"}

# Optional sample-image picker for local project images.
SAMPLE_IMAGE_DIR = str(PROJECT_ROOT / "data" / "train_images")

# --------------------------------------------------------------------------
# Visual design system
# --------------------------------------------------------------------------
# Severity scale colors — deliberately not a generic red/green traffic light,
# since 0 and 4 aren't "good/bad" in a binary sense, this is a clinical
# ordinal scale. Uses a sequential amber->crimson ramp, common in
# ophthalmology grading UIs, with class 0 in a calm teal to set it apart
# as "no disease" rather than just "least severe".
SEVERITY_COLORS = {
    0: "#2A9D8F",  # No DR — teal
    1: "#8AB17D",  # Mild — soft green
    2: "#E9C46A",  # Moderate — amber
    3: "#F4A261",  # Severe — orange
    4: "#E76F51",  # Proliferative — deep red
}
SEVERITY_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}
SEVERITY_SHORT_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Prolif",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

/* App background */
.stApp {
    background: radial-gradient(circle at 15% 0%, #101820 0%, #0B0F14 55%, #0A0D11 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D1117;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label {
    font-family: 'Space Grotesk', sans-serif;
}

/* Headline */
.app-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    background: linear-gradient(90deg, #6FE3D9 0%, #8FB8FF 50%, #C8A6FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
    letter-spacing: -0.02em;
}
.app-subtitle {
    color: #8C97A8;
    font-size: 0.98rem;
    margin-bottom: 1.6rem;
}

/* Section labels */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6FE3D9;
    margin: 1.6rem 0 0.6rem 0;
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    backdrop-filter: blur(6px);
    margin-bottom: 1rem;
}

/* Model column header */
.model-col-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.9rem;
}
.model-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.model-col-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: #F2F5F8;
}

/* Prediction hero card */
.pred-hero {
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
    min-width: 0;
}
.pred-dot {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 20px 2px var(--dot-color, #fff);
}
.pred-text-main {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: #F2F5F8;
    line-height: 1.2;
    word-break: break-word;
    overflow-wrap: anywhere;
}
.pred-text-sub {
    color: #9AA5B4;
    font-size: 0.86rem;
    margin-top: 0.15rem;
}
.pred-confidence-badge {
    margin-left: auto;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 0.35rem 0.75rem;
    border-radius: 12px;
    background: rgba(255,255,255,0.06);
    color: #F2F5F8;
    white-space: nowrap;
    flex-shrink: 0;
}

/* Agreement banner */
.agree-banner {
    border-radius: 14px;
    padding: 0.85rem 1.2rem;
    margin-bottom: 1.3rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.92rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Image captions */
.img-caption {
    text-align: center;
    color: #9AA5B4;
    font-size: 0.78rem;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.03);
    border: 1.5px dashed rgba(111,227,217,0.35);
    border-radius: 14px;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}

/* Radio pills in sidebar */
div[role="radiogroup"] label {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.35rem 0.7rem;
    margin-bottom: 0.3rem;
}
</style>
"""


# --------------------------------------------------------------------------
# Cached model loading — st.cache_resource keeps this from reloading on
# every interaction, only re-running when arch/checkpoint path changes.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Cached model loading
# --------------------------------------------------------------------------
# Streamlit re-renders the page often, so this function is cached to avoid
# repeatedly loading full model weights and recreating the target layer hook.
@st.cache_resource(show_spinner=False)
def get_model_and_layer(arch: str, checkpoint_path: str):
    device = SHARED_CONFIG["device"]
    model = load_model(arch, checkpoint_path, SHARED_CONFIG["num_classes"], device)
    target_layer = ARCHS[arch]["target_layer"](model)
    return model, target_layer


# --------------------------------------------------------------------------
# Single-model inference pipeline
# --------------------------------------------------------------------------
# This is the core runtime path for one model. It preprocesses the image, runs a
# forward pass, converts model logits to class probabilities, and computes the
# Grad-CAM++ and Layer-CAM overlays used in the demo.
def run_inference(model, arch, target_layer, pil_img: Image.Image):
    device = SHARED_CONFIG["device"]
    cfg = SHARED_CONFIG
    decode_fn = ARCHS[arch]["decode_fn"]
    score_fn = ARCHS[arch]["score_fn"]

    preprocessed = preprocess_fundus(pil_img, cfg["img_size"])
    original_rgb = np.array(preprocessed)

    tfm = get_transform(cfg["img_size"], cfg["imagenet_mean"], cfg["imagenet_std"])
    input_tensor = tfm(preprocessed).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    with torch.no_grad():
        raw_output = model(input_tensor)
        pred_class = decode_fn(raw_output)

        if arch == "efficientnet":
            class_probs = F.softmax(raw_output, dim=1)[0].cpu().numpy()
        else:  # resnet + CORN + convnext
            cumulative_probs = _corn_conditional_to_cumulative(raw_output)
            class_probs = _corn_cumulative_to_class_probs(cumulative_probs)[0].cpu().numpy()

    # Convert CORN cumulative thresholds to a stable display distribution.
    class_probs = np.clip(np.asarray(class_probs, dtype=np.float64), 0.0, 1.0)
    probability_sum = class_probs.sum()
    if probability_sum > 0:
        class_probs /= probability_sum

    cam_pp, _ = grad_cam_pp(model, target_layer, input_tensor, pred_class, score_fn)
    cam_layer, _ = layer_cam(model, target_layer, input_tensor, pred_class, score_fn)

    overlay_pp = overlay_heatmap(cam_pp, original_rgb)
    overlay_layer = overlay_heatmap(cam_layer, original_rgb)

    return {
        "pred_class": pred_class,
        "class_probs": class_probs,
        "original_rgb": original_rgb,
        "overlay_gradcam_pp": overlay_pp,
        "overlay_layercam": overlay_layer,
    }


# --------------------------------------------------------------------------
# Confidence visualization
# --------------------------------------------------------------------------
# The app renders the per-class probability distribution as a visual bar chart.
# The predicted class is highlighted with a stronger border and glow so the user
# can immediately see the model's strongest decision.
def render_confidence_chart(class_probs, class_names, pred_class):
    """Compact horizontal bar chart with severity colors, built with plain
    HTML/CSS so no extra plotting dependency is required."""
    rows_html = []
    short_labels = [SEVERITY_SHORT_LABELS.get(i, name) for i, name in enumerate(class_names)]
    for i, (name, p) in enumerate(zip(short_labels, class_probs)):
        pct = p * 100
        width_pct = max(pct, 1.5)
        color = SEVERITY_COLORS.get(i, "#8FB8FF")
        is_pred = i == pred_class
        border = f"1px solid {color}" if is_pred else "1px solid rgba(255,255,255,0.06)"
        glow = f"box-shadow: 0 0 12px 0 {color}55;" if is_pred else ""
        rows_html.append(f"""
        <div style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0.45rem; min-width:0;">
            <div style="width:76px; flex-shrink:0; font-family:'Space Grotesk',sans-serif; font-size:0.72rem;
                        color:{'#F2F5F8' if is_pred else '#9AA5B4'}; font-weight:{600 if is_pred else 400};
                        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                {name}
            </div>
            <div style="flex:1; min-width:0; background:rgba(255,255,255,0.04); border-radius:7px; height:18px;
                        border:{border}; {glow} overflow:hidden;">
                <div style="width:{width_pct:.1f}%; height:100%; background:{color};
                            border-radius:6px; transition:width 0.4s ease;"></div>
            </div>
            <div style="width:50px; flex-shrink:0; text-align:right; font-family:'Space Grotesk',sans-serif;
                        font-size:0.78rem; color:{'#F2F5F8' if is_pred else '#8C97A8'};
                        font-weight:{700 if is_pred else 500};">
                {pct:.1f}%
            </div>
        </div>
        """)
    chart_html = f'<div class="glass-card">{"".join(rows_html)}</div>'
    if hasattr(st, "html"):
        st.html(chart_html)
    else:
        st.markdown(chart_html, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Evaluation metrics loader
# --------------------------------------------------------------------------
# This section reads the saved metrics files created during training and packages
# them into a single row per model. The frontend renders the final comparison in a
# clean table with best-value highlighting for each metric column.
def _load_model_metrics(arch):
    """Load the public validation metrics for one architecture."""
    if arch == "efficientnet":
        report_path = PROJECT_ROOT / "outputs" / "metrics" / "classification_report_efficientnet_b0.txt"
        if not report_path.exists():
            return None
        report_text = report_path.read_text()
        acc = float(re.search(r"Accuracy:\s*([0-9.]+)", report_text).group(1))
        qwk = float(re.search(r"Quadratic QWK:\s*([0-9.]+)", report_text).group(1))
        macro_f1 = float(re.search(r"Macro-F1:\s*([0-9.]+)", report_text).group(1))
        f1_by_class = {}
        for line in report_text.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                class_id = int(parts[0])
                f1_by_class[class_id] = float(parts[3])
    elif arch == "resnet":
        report_path = PROJECT_ROOT / "outputs" / "metrics" / "classification_report_resnet18_corn.txt"
        if not report_path.exists():
            return None
        report_text = report_path.read_text()
        acc = float(re.search(r"Accuracy:\s*([0-9.]+)", report_text).group(1))
        qwk = float(re.search(r"Quadratic QWK:\s*([0-9.]+)", report_text).group(1))
        macro_f1 = float(re.search(r"Macro-F1:\s*([0-9.]+)", report_text).group(1))
        f1_by_class = {}
        for line in report_text.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                class_id = int(parts[0])
                f1_by_class[class_id] = float(parts[3])
    else:
        metrics_path = PROJECT_ROOT / "outputs" / "metrics" / "convnext_tiny_corn_metrics.json"
        if not metrics_path.exists():
            return None
        payload = json.loads(metrics_path.read_text())
        acc = float(payload["accuracy"])
        qwk = float(payload["qwk"])
        macro_f1 = float(payload["macro_f1"])
        f1_by_class = {idx: float(value) for idx, value in enumerate(payload["f1_by_class"])}

    row = {
        "Model": ARCH_DISPLAY[arch],
        "Accuracy": acc,
        "Macro F1": macro_f1,
        "QWK": qwk,
    }
    for grade in range(5):
        row[f"F1 Grade {grade}"] = f1_by_class.get(grade, float("nan"))
    return row


# --------------------------------------------------------------------------
# Metrics page
# --------------------------------------------------------------------------
# The metrics view is intentionally separated from the analysis view so the user
# can inspect performance without cluttering the image-understanding screen.
def render_metrics_page():
    """Render a dedicated metrics view in the sidebar navigation."""
    st.markdown('<div class="app-title">Model Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Validation metrics for the trained project models.</div>',
        unsafe_allow_html=True,
    )

    metrics_rows = []
    for arch in ARCH_ORDER:
        row = _load_model_metrics(arch)
        if row is not None:
            metrics_rows.append(row)

    if not metrics_rows:
        st.info("No saved model metrics were found for this project yet.")
        return

    df = pd.DataFrame(metrics_rows)
    metric_cols = [
        "Accuracy",
        "Macro F1",
        "F1 Grade 0",
        "F1 Grade 1",
        "F1 Grade 2",
        "F1 Grade 3",
        "F1 Grade 4",
        "QWK",
    ]

    def highlight_best(s):
        if s.empty:
            return [""] * len(s)
        best = s.max()
        return ["background-color: rgba(38, 166, 154, 0.28); color: #F2F5F8; font-weight: 700;" if v == best else "" for v in s]

    styler = df.style.format({col: "{:.4f}" for col in metric_cols})
    styler = styler.apply(highlight_best, subset=metric_cols, axis=0)
    st.dataframe(styler, hide_index=True, use_container_width=True)
    st.caption("These are on the 550-image held-out validation set.")


# --------------------------------------------------------------------------
# Full per-model card renderer
# --------------------------------------------------------------------------
# Each architecture gets one column in the dashboard. This function creates the
# prediction hero, confidence chart, and input/CAM image layout for that model.
def render_model_column(arch, checkpoint_path, input_image, method, class_names):
    """Renders one full model's results (prediction hero and confidence bars)
    inside whatever Streamlit container is currently active."""
    accent = ARCH_ACCENT[arch]

    st.markdown(
        f"""
        <div class="model-col-header">
            <div class="model-dot" style="background:{accent}; box-shadow:0 0 10px 1px {accent}88;"></div>
            <div class="model-col-title">{ARCH_DISPLAY[arch]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        st.markdown(
            f'<div class="glass-card" style="color:#E76F51;">'
            f'Checkpoint not found:<br><code style="font-size:0.75rem;">{checkpoint_path}</code>'
            f'<br>Train this model first.</div>',
            unsafe_allow_html=True,
        )
        return None

    with st.spinner(f"Running {ARCH_SHORT[arch]}..."):
        model, target_layer = get_model_and_layer(arch, checkpoint_path)
        result = run_inference(model, arch, target_layer, input_image)

    pred_class = result["pred_class"]
    color = SEVERITY_COLORS.get(pred_class, "#8FB8FF")
    confidence = result["class_probs"][pred_class] * 100

    st.markdown(
        f"""
        <div class="pred-hero" style="background: linear-gradient(120deg, {color}22 0%, rgba(255,255,255,0.02) 60%);">
            <div class="pred-dot" style="background:{color}; --dot-color:{color};"></div>
            <div style="min-width:0;">
                <div class="pred-text-main">{SEVERITY_SHORT_LABELS.get(pred_class, class_names[pred_class])}</div>
                <div class="pred-text-sub">Grade {pred_class} of 4</div>
            </div>
            <div class="pred-confidence-badge">{confidence:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_confidence_chart(result["class_probs"], class_names, pred_class)

    img_cols = st.columns(3 if method == "both" else 2)
    with img_cols[0]:
        st.image(result["original_rgb"], caption="Input")

    slot = 1
    if method in ("gradcam++", "both"):
        with img_cols[slot]:
            st.image(result["overlay_gradcam_pp"], caption="Grad-CAM++")
        slot += 1
    if method in ("layercam", "both"):
        with img_cols[slot]:
            st.image(result["overlay_layercam"], caption="Layer-CAM")

    return result


# --------------------------------------------------------------------------
# UI entry point
# --------------------------------------------------------------------------
# This is the actual dashboard startup section. Streamlit executes this file top
# to bottom whenever the app runs. The sidebar configuration and upload widgets
# here define the main user experience.
st.set_page_config(page_title="DR Severity Grading", layout="wide", page_icon="🩺")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar configuration controls the app mode and user settings for model
# explanation, while keeping the primary image-analysis layout simple.
with st.sidebar:
    checkpoint_paths = {arch: CHECKPOINT_PATHS[arch] for arch in ARCH_ORDER}
    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Analysis", "Metrics"], index=0, label_visibility="collapsed")

    st.markdown('<div class="section-label">Models</div>', unsafe_allow_html=True)
    st.caption("Using the project default checkpoints for EfficientNet, ResNet, and ConvNeXt.")

    st.markdown('<div class="section-label">Explanation</div>', unsafe_allow_html=True)
    method = st.radio(
        "Explanation method",
        options=["gradcam++", "layercam", "both"],
        index=2,
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">Severity Scale</div>', unsafe_allow_html=True)
    legend_html = "".join(
        f'<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">'
        f'<div style="width:10px; height:10px; border-radius:50%; background:{SEVERITY_COLORS[i]};"></div>'
        f'<span style="font-size:0.8rem; color:#9AA5B4;">{i} — {SEVERITY_LABELS[i]}</span></div>'
        for i in range(5)
    )
    st.markdown(f'<div class="glass-card">{legend_html}</div>', unsafe_allow_html=True)

if page == "Metrics":
    render_metrics_page()
    st.stop()

st.markdown('<div class="app-title">Diabetic Retinopathy Severity Grading</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload a fundus image to compare EfficientNet-B0+QWK, '
    'ResNet-18+CORN, and ConvNeXt-Tiny+CORN side by side — ordinal severity grade (0–4), '
    'confidence, and Grad-CAM++/Layer-CAM explainability.</div>',
    unsafe_allow_html=True,
)

st.divider()

# This block handles the image upload workflow. The user can either load a local
# image or pick a project sample, and the app converts the input into a clean RGB
# image before running any model inference.
upload_col, sample_col = st.columns([2, 1])
with upload_col:
    uploaded_file = st.file_uploader(
        "Upload a fundus image",
        type=["png", "jpg", "jpeg"],
        key="fundus_upload",
        help="Choose a PNG or JPEG fundus image from your computer.",
    )

sample_choice = None
if SAMPLE_IMAGE_DIR:
    with sample_col:
        sample_paths = sorted(Path(SAMPLE_IMAGE_DIR).glob("*.png"))
        sample_choice = st.selectbox(
            "...or pick a sample image", options=["(none)"] + [p.name for p in sample_paths]
        )

# Normalize the input source into one consistent image object so all models are
# evaluated on the same data representation.
input_image = None
input_source = None
if uploaded_file is not None:
    try:
        # Read and copy the upload immediately so inference never depends on
        # Streamlit's temporary upload handle remaining open.
        input_image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB").copy()
        input_source = uploaded_file.name
        st.success(f"Loaded {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")
elif sample_choice and sample_choice != "(none)":
    input_image = Image.open(Path(SAMPLE_IMAGE_DIR) / sample_choice).convert("RGB").copy()
    input_source = sample_choice

if input_image is not None:
    # Once an image is loaded, the app runs all three models in parallel columns
    # and populates the per-model cards with prediction, confidence, and heatmap
    # explainability information.
    class_names = SHARED_CONFIG["class_names"]
    st.caption(f"Analyzing: {input_source}")
    try:
        model_cols = st.columns(len(ARCH_ORDER), gap="large")
        results = {}
        for idx, arch in enumerate(ARCH_ORDER):
            with model_cols[idx]:
                results[arch] = render_model_column(
                    arch, checkpoint_paths[arch], input_image, method, class_names
                )

        st.markdown(
            """
            <div class="glass-card" style="padding:0.9rem 1rem; margin-top: 0.75rem;">
                <div style="color:#F2F5F8; font-family:'Space Grotesk',sans-serif;
                            font-weight:600; margin-bottom:0.35rem;">
                    Explainability note
                </div>
                <div style="color:#AAB4C2; font-size:0.82rem; line-height:1.5;">
                    Warm colors show regions that contributed more strongly to the model's prediction.
                    Grad-CAM++ emphasizes fine-grained evidence, while Layer-CAM shows the spatial
                    importance from the final convolutional layer. These maps are model attention,
                    not a clinical diagnosis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.error(f"The image loaded, but model inference failed: {type(exc).__name__}: {exc}")
        st.info("Try a PNG/JPEG fundus image or refresh the page to reset the model session.")
        st.stop()

else:
    st.markdown(
        '<div class="glass-card" style="text-align:center; padding:3rem 1.5rem; color:#8C97A8;">'
        'Upload a fundus image above to compare all three models\' predictions and explainability heatmaps.'
        '</div>',
        unsafe_allow_html=True,
    )