"""
Grad-CAM++ and Layer-CAM for the Diabetic Retinopathy graders.

Supports BOTH models from the same script:
    --arch efficientnet   EfficientNet-B0 + soft-QWK loss   (train_efficientnet_b0.py)
    --arch resnet         ResNet-18 + CORN ordinal loss      (train_resnet18_corn.py)

The hook mechanics and the two CAM algorithms (Grad-CAM++, Layer-CAM) are
identical either way — only three things differ per architecture, and each
is isolated behind the ARCHS registry below:
    1. how the model is built / its head shape
    2. which layer to hook (last conv block)
    3. how to turn raw model output into (a) a scalar "class score" to
       backprop through and (b) a decoded class prediction 0-4
       (plain softmax for EfficientNet vs CORN's cumulative-threshold
       decoding for ResNet)

Usage
-----
Single image:
    python src/gradcam.py --arch efficientnet --checkpoint models/efficientnet_b0_qwk_best.pt --image img.png --method both
    python src/gradcam.py --arch resnet       --checkpoint models/resnet18_corn_best.pt --image img.png --method both

Batch folder, for the Streamlit dashboard to consume later:
    python src/gradcam.py --arch efficientnet --checkpoint models/efficientnet_b0_qwk_best.pt \
        --image_dir data/train_images --out_dir cam_outputs --method both

Both methods run on the SAME class by default (the model's own top-1
prediction, decoded per-architecture). Pass --target_class to force a
specific class (0-4) instead, e.g. for a "why not class 3" error-analysis
figure in the report.
"""

# --------------------------------------------------------------------------
# Grad-CAM++ and Layer-CAM for the diabetic-retinopathy grading models.
#
# This file is the explanation engine behind the dashboard. It handles the model
# architecture differences, chooses the correct target layer for each backbone,
# converts CORN outputs into probabilities, and overlays heatmaps back onto the
# original retinal image so the user can see what the model is focusing on.
# --------------------------------------------------------------------------

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import models, transforms

# --------------------------------------------------------------------------
# Shared config — adjust if your preprocessing differs
# --------------------------------------------------------------------------
SHARED_CONFIG = {
    "num_classes": 5,
    "img_size": 224,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
    "device": torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    ),
    "class_names": ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative"],
}


# --------------------------------------------------------------------------
# Image preprocessing for CAM generation
# --------------------------------------------------------------------------
# The training and validation pipelines resize the retinal image to 224x224 and
# normalize with ImageNet statistics. The explainability code must match this to
# keep heatmaps comparable to the original model behavior.
def preprocess_fundus(image: Image.Image, target_size: int = 224) -> Image.Image:
    """Prepare an image exactly as the training datasets do at inference.

    The PNGs are loaded directly by both training scripts. Their validation
    transform is resize to 224, ImageNet normalization, and tensor conversion;
    no additional crop, padding, or Ben Graham pass is applied here.
    """
    image_rgb = np.array(image.convert("RGB"))
    resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(resized)


# --------------------------------------------------------------------------
# Per-architecture logic — this is the ONLY section that differs by model
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Architecture-specific model builders
# --------------------------------------------------------------------------
# Each model head is different: EfficientNet uses a normal softmax classifier,
# while ResNet and ConvNeXt use CORN ordinal heads. The class count is always 5,
# but the output dimension differs because ordinal models predict thresholds.
def build_efficientnet(num_classes: int) -> nn.Module:
    """EfficientNet-B0 with a plain num_classes-way softmax head (Model 1)."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet(num_classes: int) -> nn.Module:
    """ResNet-18 with a CORN head (Model 2).

    CORN needs (num_classes - 1) output units — one binary "rank exceeds
    threshold k" logit per ordinal cutoff, not a normal num_classes-way
    softmax. If train_resnet18_corn.py built the head differently (extra
    hidden layer, dropout, etc.), mirror that here — everything downstream
    only needs `model.layer4[-1]` to still exist as the last conv block,
    which holds for any torchvision resnet regardless of head changes.
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes - 1)
    return model


def build_convnext(num_classes: int) -> nn.Module:
    """ConvNeXt-Tiny with a CORN head (Model 3)."""
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes - 1)
    return model


# Explainability score functions are the glue between the model output and the
# heatmap. A heatmap is only meaningful if it is produced from the same scoring
# convention that the model uses during training and inference.
def score_efficientnet(output: torch.Tensor, target_class: int):
    """Plain softmax logit for the target class — a normal CAM score."""
    return output[:, target_class]


def score_resnet_corn(output: torch.Tensor, target_class: int):
    """CORN score to backprop through for class `target_class`.

    CORN's raw output is (num_classes - 1) logits, each the CONDITIONAL
    probability P(y>k | y>k-1) — not the cumulative P(y>k) directly. They
    must be combined via cumulative product to get true rank probabilities
    (see _corn_conditional_to_cumulative below); thresholding the raw
    sigmoids independently breaks CORN's rank-consistency guarantee and
    was the root cause of a systematic off-by-one-class bias found when
    validating Model 2 (fixed here to match the corrected decoder in
    train_resnet18_corn.py). We then convert cumulative probs to per-class
    probabilities and backprop through the target class's probability, so
    the heatmap explains the actual predicted/target class the same way
    the softmax score does for the EfficientNet model.
    """
    cumulative_probs = _corn_conditional_to_cumulative(output)
    class_probs = _corn_cumulative_to_class_probs(cumulative_probs)
    return class_probs[:, target_class]


# CORN conversion utilities
# --------------------------------------------------------------------------
# CORN models predict ordinal cut-off logits. To explain the model correctly, we
# must convert those conditional probabilities into cumulative probabilities and
# then to class probabilities. This is a key part of explaining why the ResNet and
# ConvNeXt heatmaps behave differently from EfficientNet.
def _corn_conditional_to_cumulative(output: torch.Tensor) -> torch.Tensor:
    """[B, K] conditional logits -> [B, K] true cumulative P(rank > k).

    P(y>k) = prod_{j<=k} P(y>j | y>j-1), i.e. the cumulative product of the
    conditional sigmoids in order. This is the step CORN's decode requires
    and that a naive independent-threshold decode skips.
    """
    conditional_probs = torch.sigmoid(output)  # [B, K], P(y>k | y>k-1)
    cumulative_probs = torch.cumprod(conditional_probs, dim=1)  # [B, K], P(y>k)
    return cumulative_probs


def _corn_cumulative_to_class_probs(cumulative_probs: torch.Tensor) -> torch.Tensor:
    """[B, K] cumulative P(rank > k) -> [B, K+1] per-class probabilities.

    P(class=0)  = 1 - P(rank>0)
    P(class=k)  = P(rank>k-1) - P(rank>k)      for 0 < k < K
    P(class=K)  = P(rank>K-1)
    """
    batch, k = cumulative_probs.shape
    ones = torch.ones(batch, 1, device=cumulative_probs.device)
    zeros = torch.zeros(batch, 1, device=cumulative_probs.device)
    padded = torch.cat([ones, cumulative_probs, zeros], dim=1)  # [B, K+2]
    class_probs = padded[:, :-1] - padded[:, 1:]  # [B, K+1]
    return class_probs


def decode_efficientnet(output: torch.Tensor) -> int:
    # EfficientNet uses a standard argmax over the 5-class logits.
    return int(output.argmax(dim=1).item())


def decode_resnet_corn(output: torch.Tensor) -> int:
    """Corrected CORN decode: cumulative-product the conditional sigmoids
    first, THEN count how many cumulative thresholds exceed 0.5 — matches
    the fixed corn_predict in train_resnet18_corn.py. Counting raw
    conditional sigmoids independently (the old, buggy version) breaks
    rank-consistency and causes systematic class-shift errors."""
    cumulative_probs = _corn_conditional_to_cumulative(output)
    predicted_rank = (cumulative_probs > 0.5).sum(dim=1)
    return int(predicted_rank.item())


# Registry mapping each model name to the right architecture-specific logic.
# The frontend and CLI both use this single registry to stay consistent with the
# training scripts and prevent architecture mismatches.
ARCHS = {
    "efficientnet": {
        "build": build_efficientnet,
        "target_layer": lambda model: model.features[-1],
        "score_fn": score_efficientnet,
        "decode_fn": decode_efficientnet,
    },
    "resnet": {
        "build": build_resnet,
        "target_layer": lambda model: model.layer4[-1],
        "score_fn": score_resnet_corn,
        "decode_fn": decode_resnet_corn,
    },
    "convnext": {
        "build": build_convnext,
        "target_layer": lambda model: model.features[-1][-1],
        "score_fn": score_resnet_corn,
        "decode_fn": decode_resnet_corn,
    },
}


# --------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------
# The app and CLI both rely on this helper to rebuild the exact architecture and
# load the saved weights. This prevents mismatches between training heads and CAM
# scoring logic.
def load_model(arch: str, checkpoint_path: str, num_classes: int, device: str) -> nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the selected model first."
        )
    model = ARCHS[arch]["build"](num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] non-strict load — missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Shared hook-based CAM machinery (architecture-agnostic)
# --------------------------------------------------------------------------
class _ActivationsAndGradients:
    """Grabs forward activations and backward gradients at one target layer."""
    # This utility is the core of Grad-CAM and Layer-CAM. It stores the output of
    # a chosen feature map and the gradient of the target score with respect to
    # that activation so the heatmap can be weighted by saliency.

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        return self.model(x)

    def release(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def _normalize_cam(cam: torch.Tensor) -> np.ndarray:
    cam = F.relu(cam)
    cam = cam - cam.min()
    denom = cam.max()
    if denom > 0:
        cam = cam / denom
    return cam.cpu().numpy()


def grad_cam_pp(model, target_layer, input_tensor, target_class, score_fn):
    """Grad-CAM++ (Chattopadhay et al., 2018).

    Weights each pixel's gradient contribution rather than using a single
    channel-wide weight — matters here since DR lesions (microaneurysms,
    hemorrhages, exudates) are scattered across the retina, not one blob.
    """
    extractor = _ActivationsAndGradients(model, target_layer)
    try:
        model.zero_grad()
        output = extractor(input_tensor)
        score = score_fn(output, target_class)
        score.backward(retain_graph=False)

        activations = extractor.activations[0]  # [C, H, W]
        gradients = extractor.gradients[0]       # [C, H, W]

        grads_2 = gradients ** 2
        grads_3 = gradients ** 3
        sum_acts = activations.sum(dim=(1, 2), keepdim=True)

        eps = 1e-8
        alpha_denom = 2 * grads_2 + sum_acts * grads_3
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.full_like(alpha_denom, eps))
        alpha = grads_2 / alpha_denom

        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))  # [C]
        cam = (weights[:, None, None] * activations).sum(dim=0)  # [H, W]
        return _normalize_cam(cam), output.detach()
    finally:
        extractor.release()


def layer_cam(model, target_layer, input_tensor, target_class, score_fn):
    """Layer-CAM (Jiang et al., 2021).

    Weights activations pixel-by-pixel with the (positive part of the)
    gradient at that exact spatial location — sharper, more fine-grained
    maps than a single channel-wide weight, useful for pinpointing small
    lesions rather than a coarse retinal region.
    """
    extractor = _ActivationsAndGradients(model, target_layer)
    try:
        model.zero_grad()
        output = extractor(input_tensor)
        score = score_fn(output, target_class)
        score.backward(retain_graph=False)

        activations = extractor.activations[0]  # [C, H, W]
        gradients = extractor.gradients[0]       # [C, H, W]

        weighted_acts = F.relu(gradients) * activations
        cam = weighted_acts.sum(dim=0)  # [H, W]
        return _normalize_cam(cam), output.detach()
    finally:
        extractor.release()


# --------------------------------------------------------------------------
# Preprocessing / visualization (architecture-agnostic)
# --------------------------------------------------------------------------
def get_transform(img_size, mean, std):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def overlay_heatmap(cam_2d: np.ndarray, original_rgb: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """cam_2d in [0,1] at feature-map resolution; original_rgb is HxWx3 uint8."""
    h, w = original_rgb.shape[:2]
    cam_resized = np.array(Image.fromarray((cam_2d * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))
    cam_resized = cam_resized.astype(np.float32) / 255.0

    heatmap = _colorize(cam_resized)  # HxWx3 uint8, jet-like colormap
    overlay = (alpha * heatmap + (1 - alpha) * original_rgb).astype(np.uint8)
    return overlay


def _colorize(cam_norm: np.ndarray) -> np.ndarray:
    """Minimal dependency-free jet-like colormap (avoids requiring matplotlib/opencv)."""
    r = np.clip(1.5 - np.abs(4 * cam_norm - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam_norm - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam_norm - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Per-image pipeline
# --------------------------------------------------------------------------
def explain_image(model, arch, target_layer, image_path, device, cfg, method, target_class=None):
    score_fn = ARCHS[arch]["score_fn"]
    decode_fn = ARCHS[arch]["decode_fn"]

    pil_img = preprocess_fundus(Image.open(image_path), cfg["img_size"])
    original_rgb = np.array(pil_img)

    tfm = get_transform(cfg["img_size"], cfg["imagenet_mean"], cfg["imagenet_std"])
    input_tensor = tfm(pil_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    results = {}

    with torch.no_grad():
        raw_output = model(input_tensor)
        pred_class = decode_fn(raw_output)
    used_class = target_class if target_class is not None else pred_class

    if method in ("gradcam++", "both"):
        cam, output = grad_cam_pp(model, target_layer, input_tensor, used_class, score_fn)
        results["gradcam++"] = overlay_heatmap(cam, original_rgb)

    if method in ("layercam", "both"):
        cam, output = layer_cam(model, target_layer, input_tensor, used_class, score_fn)
        results["layercam"] = overlay_heatmap(cam, original_rgb)

    meta = {
        "predicted_class": pred_class,
        "explained_class": used_class,
        "original_rgb": original_rgb,
    }
    return results, meta


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Grad-CAM++ / Layer-CAM for DR grading models")
    parser.add_argument("--arch", choices=list(ARCHS.keys()), required=True,
                         help="efficientnet = Model 1 (EfficientNet-B0+QWK), resnet = Model 2 (ResNet-18+CORN)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, help="Directory of images to process in batch")
    parser.add_argument("--out_dir", type=str, default="cam_outputs")
    parser.add_argument("--method", choices=["gradcam++", "layercam", "both"], default="both")
    parser.add_argument("--target_class", type=int, default=None, help="Force explanation for this class (0-4)")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Pass either --image or --image_dir")

    cfg = SHARED_CONFIG
    device = cfg["device"]
    model = load_model(args.arch, args.checkpoint, cfg["num_classes"], device)
    target_layer = ARCHS[args.arch]["target_layer"](model)

    os.makedirs(args.out_dir, exist_ok=True)

    image_paths = [Path(args.image)] if args.image else sorted(Path(args.image_dir).glob("*.png"))

    for img_path in image_paths:
        results, meta = explain_image(
            model, args.arch, target_layer, img_path, device, cfg, args.method, args.target_class
        )
        stem = img_path.stem
        pred_name = cfg["class_names"][meta["predicted_class"]]
        explained_name = cfg["class_names"][meta["explained_class"]]
        print(f"[{args.arch}] {img_path.name}: predicted={pred_name} (explained class={explained_name})")

        for method_name, overlay in results.items():
            out_path = Path(args.out_dir) / f"{stem}_{args.arch}_{method_name.replace('++', 'pp')}.png"
            Image.fromarray(overlay).save(out_path)
            print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
