"""Model 3: ConvNeXt-Tiny + CORN ordinal loss for DR severity grading.

This is the third and final architecture in the project. It mirrors the same
train/validation split and output conventions as the other models so that the
comparison remains fair. The major difference is that ConvNeXt is a modern CNN
backbone with a CORN ordinal head, which is designed for ordered labels such as
DR severity grades 0 to 4.

Run from the project root with:
    python3 src/train_convnext_corn.py
"""
import os
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

# ---------------------------------------------------------------- config
# This configuration block keeps the script portable. It resolves the project
# root automatically, supports environment overrides, and defines the paths for
# the training CSV, images, checkpoints, and saved metrics.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
CSV_PATH = DATA_DIR / "train.csv"            # columns: id_code, diagnosis
IMG_DIR = Path(os.environ.get("IMG_DIR", DATA_DIR / "train_images"))
IMG_EXT = os.environ.get("IMG_EXT", ".png")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
LR = float(os.environ.get("LR", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.05"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
SEED = 42
NUM_CLASSES = 5

OUT = PROJECT_ROOT / "outputs"
CKPT_PATH = PROJECT_ROOT / "models" / "convnext_tiny_corn_best.pt"
SUB_PATH = OUT / "submissions" / "submission_convnext_tiny_corn.csv"
METRICS_PATH = OUT / "metrics" / "convnext_tiny_corn_metrics.json"
for p in (CKPT_PATH, SUB_PATH, METRICS_PATH):
    p.parent.mkdir(parents=True, exist_ok=True)

# Prefer Apple MPS for Mac, then CUDA, then CPU. This matters because training on
# local machines can be inconsistent between GPU and CPU backends.
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------------------------------
# Reproducibility helpers
# --------------------------------------------------------------------------
# Every training run should be deterministic when possible so the metrics and
# checkpoint behavior remain stable across repeated runs.
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
# Eval transform = resize -> ImageNet normalize. Nothing else. Whatever the
# app / gradcam code does at inference must match this exactly.
NORM = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
    NORM, ToTensorV2(),
])
eval_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), NORM, ToTensorV2()])


# A custom dataset class keeps the image loading and label extraction logic in
# one place, which makes the training loop easier to reason about and explain.
class DRDataset(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image_path = IMG_DIR / f"{row['id_code']}{IMG_EXT}"
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read training image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(image=img)["image"], torch.tensor(
            int(row["diagnosis"]), dtype=torch.long
        )


# ---------------------------------------------------------------- CORN
# --------------------------------------------------------------------------
# CORN ordinal loss
# --------------------------------------------------------------------------
# This is the key idea for the ordinal model. Instead of predicting a single
# 5-class softmax, the network predicts whether the true grade is above each
# ordinal threshold. This matches the ordinal nature of DR severity.
def corn_loss(logits, y, num_classes=NUM_CLASSES):
    """Conditional ordinal regression: task k predicts P(y > k | y >= k),
    trained only on samples with y >= k."""
    total, count = 0.0, 0
    for k in range(num_classes - 1):
        mask = y >= k
        if mask.sum() == 0:
            continue
        target = (y[mask] > k).float()
        total = total + F.binary_cross_entropy_with_logits(
            logits[mask, k], target, reduction="sum")
        count += int(mask.sum())
    return total / count


def corn_predict(logits):
    """Rank-consistent decode: cumulative product FIRST, then threshold."""
    # At inference time, the model predicts one score per ordinal threshold.
    # The cumulative product turns these conditional probabilities into a valid
    # ranking probability, then thresholding gives the final grade 0-4.
    cumulative = torch.cumprod(torch.sigmoid(logits), dim=1)
    return (cumulative > 0.5).sum(dim=1)


# ---------------------------------------------------------------- model
# --------------------------------------------------------------------------
# Backbone and output head
# --------------------------------------------------------------------------
# ConvNeXt-Tiny is initialized with pretrained ImageNet weights, then the final
# classifier is replaced with a CORN head sized for 4 ordinal cut-offs.
def build_model():
    m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # classifier = [LayerNorm2d, Flatten, Linear(768, 1000)] -> swap the Linear for 4 CORN logits
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES - 1)
    return m


# ---------------------------------------------------------------- eval
# --------------------------------------------------------------------------
# Validation evaluation
# --------------------------------------------------------------------------
# The validation loop scores the model on held-out data without updating weights.
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        out = model(x.to(DEVICE))
        preds.append(corn_predict(out).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def compute_metrics(y_true, y_pred):
    # This function summarizes performance with the metrics that matter in an
    # ordinal grading task: general accuracy, weighted agreement, macro F1, and
    # per-class F1 for each severity grade.
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_by_class": [float(v) for v in
                        f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)))],
    }


# ---------------------------------------------------------------- main
# --------------------------------------------------------------------------
# Training entry point
# --------------------------------------------------------------------------
# This is the main execution block. It validates the dataset, builds a class-
# balanced sampler, trains the model, saves the best checkpoint, and exports the
# final validation metrics and predictions.
def main():
    seed_everything(SEED)
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Training CSV not found: {CSV_PATH}")
    if not IMG_DIR.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {IMG_DIR}")

    df = pd.read_csv(CSV_PATH)
    required_columns = {"id_code", "diagnosis"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{CSV_PATH} is missing columns: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError(f"{CSV_PATH} contains no training rows")
    if not df["diagnosis"].between(0, NUM_CLASSES - 1).all():
        raise ValueError(f"diagnosis values must be integers from 0 to {NUM_CLASSES - 1}")

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=SEED, stratify=df["diagnosis"])

    if len(df) != 3662 or len(val_df) != 550:
        print(
            "Warning: this dataset does not match the project's standard "
            f"3,662-row/550-row split ({len(df)} rows/{len(val_df)} val rows)."
        )
    assert not set(train_df.id_code) & set(val_df.id_code), "train/val leakage"
    print(f"device={DEVICE}  train={len(train_df)}  val={len(val_df)}")
    print(f"csv={CSV_PATH}  images={IMG_DIR}")

    # Class-balanced sampling, same idea as Models 1 and 2
    counts = (
        train_df["diagnosis"]
        .value_counts()
        .reindex(range(NUM_CLASSES), fill_value=0)
        .to_numpy()
    )
    if (counts == 0).any():
        missing_classes = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training split is missing diagnosis classes: {missing_classes}")
    class_w = 1.0 / counts
    sample_w = train_df["diagnosis"].map(lambda c: class_w[c]).to_numpy()
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(train_df),
        replacement=True,
    )

    train_loader = DataLoader(DRDataset(train_df, train_tf), batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(DRDataset(val_df, eval_tf), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = build_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_qwk, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = corn_loss(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        sched.step()

        preds, labels = evaluate(model, val_loader)
        m = compute_metrics(labels, preds)
        print(f"epoch {epoch:02d}  loss={running / len(train_df):.4f}  "
              f"acc={m['accuracy']:.4f}  qwk={m['qwk']:.4f}  f1={m['macro_f1']:.4f}  "
              f"f1/class={[round(v, 2) for v in m['f1_by_class']]}")

        if m["qwk"] > best_qwk:
            best_qwk = m["qwk"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state": best_state,
                    "val_qwk": best_qwk,
                    "epoch": epoch,
                    "arch": "convnext_tiny_corn",
                    "img_size": IMG_SIZE,
                    "num_classes": NUM_CLASSES,
                },
                CKPT_PATH,
            )

    # Final report from the best checkpoint
    if best_state is None:
        raise RuntimeError("No checkpoint was produced; set EPOCHS to a positive value")
    model.load_state_dict(best_state)
    preds, labels = evaluate(model, val_loader)
    final = compute_metrics(labels, preds)
    final["confusion_matrix"] = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))).tolist()
    METRICS_PATH.write_text(json.dumps(final, indent=2))
    pd.DataFrame({"id_code": val_df["id_code"].values, "diagnosis": preds}).to_csv(SUB_PATH, index=False)

    print("\n=== FINAL (best val QWK checkpoint) ===")
    print(f"Accuracy {final['accuracy']:.4f} | QWK {final['qwk']:.4f} | Macro-F1 {final['macro_f1']:.4f}")
    print("F1 by class:", [round(v, 4) for v in final["f1_by_class"]])
    print("Confusion matrix (rows = true):")
    print(np.array(final["confusion_matrix"]))


if __name__ == "__main__":
    main()
