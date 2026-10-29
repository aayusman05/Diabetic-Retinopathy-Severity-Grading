"""
DR Severity Grading — Model 2: ResNet-18 + CORN ordinal loss
Mirrors train_efficientnet_b0.py: same preprocessed images, same deterministic
stratified split (random_state=42, test_size=0.15), same WeightedRandomSampler
strategy, same MPS training loop / checkpointing / submission export format.

CORN (Conditional Ordinal Regression for Neural networks — Shi et al. 2021,
arXiv:2111.08851) replaces CORAL. CORAL trains K-1 independent binary
threshold classifiers, which can produce rank-inconsistent probabilities in
practice despite the theoretical guarantee. CORN conditions each threshold
classifier's training set on the previous threshold, which guarantees
monotonic (rank-consistent) predictions by construction — this is why CORN is
used here instead of plain CORAL.

Usage:
    python train_resnet18_corn.py
Environment overrides (mirrors the EfficientNet-B0 script; fast mode is OFF
by default and must be explicitly enabled — see the FAST_TRAIN incident notes
from the model-1 run):
    FAST_TRAIN=1 MAX_TRAIN_SAMPLES=200 MAX_VAL_SAMPLES=50 python train_resnet18_corn.py
    TRAIN_SUBSAMPLE=1500 IMG_SIZE=224 BATCH_SIZE=8 EPOCHS=20 NUM_WORKERS=0 python train_resnet18_corn.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore", message=".*var_limit.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = (
    SCRIPT_DIR
    if os.path.isdir(os.path.join(SCRIPT_DIR, "data"))
    else os.path.dirname(SCRIPT_DIR)
)

# ---------------------------------------------------------------------------
# Config (env-overridable, mirrors model-1 script)
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMG_DIR = os.environ.get("IMG_DIR", os.path.join(DATA_DIR, "train_images"))

IMG_SIZE = int(os.environ.get("IMG_SIZE", 224))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
EPOCHS = int(os.environ.get("EPOCHS", 20))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))
LR = float(os.environ.get("LR", 3e-4))
NUM_CLASSES = 5
SEED = 42

# Explicit opt-in fast mode ONLY (see model-1 incident: silently-added env
# overrides produced an invalid 50-row / 2-epoch "completed" run). Off by default.
FAST_TRAIN = os.environ.get("FAST_TRAIN", "0") == "1"
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", 0))
MAX_VAL_SAMPLES = int(os.environ.get("MAX_VAL_SAMPLES", 0))
TRAIN_SUBSAMPLE = int(os.environ.get("TRAIN_SUBSAMPLE", 0))  # 0 = full dataset

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "resnet18_corn_best.pt")
SUBMISSION_PATH = os.path.join(PROJECT_ROOT, "outputs", "submissions", "submission_resnet18_corn.csv")
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

if FAST_TRAIN:
    print("=" * 70)
    print("WARNING: FAST_TRAIN is ON. Results from this run are NOT valid")
    print("for reporting. This mode exists for debugging only.")
    print(f"  MAX_TRAIN_SAMPLES={MAX_TRAIN_SAMPLES or 'unset'}  MAX_VAL_SAMPLES={MAX_VAL_SAMPLES or 'unset'}")
    print("=" * 70)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"Tensor device: {torch.zeros(1).to(device).device}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = pd.read_csv(TRAIN_CSV)

train_df, val_df = train_test_split(
    df, test_size=0.15, stratify=df["diagnosis"], random_state=SEED
)

if TRAIN_SUBSAMPLE and TRAIN_SUBSAMPLE < len(train_df):
    print(f"Compute-constrained run: stratified training subsample of {TRAIN_SUBSAMPLE} images.")
    train_df, _ = train_test_split(
        train_df, train_size=TRAIN_SUBSAMPLE, stratify=train_df["diagnosis"], random_state=SEED
    )
else:
    print("Using full training split (no subsampling).")

if FAST_TRAIN:
    if MAX_TRAIN_SAMPLES:
        train_df = train_df.iloc[:MAX_TRAIN_SAMPLES]
    if MAX_VAL_SAMPLES:
        val_df = val_df.iloc[:MAX_VAL_SAMPLES]

print(f"Training images: {len(train_df)} | Validation images: {len(val_df)}")
print(f"Train classes: {sorted(train_df['diagnosis'].unique().tolist())}")
print(f"Validation classes: {sorted(val_df['diagnosis'].unique().tolist())}")
print(f"Validation class counts: {val_df['diagnosis'].value_counts().sort_index().to_dict()}")

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
    A.GaussNoise(std_range=(0.02, 0.06), p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class DRDataset(Dataset):
    """Assumes Ben Graham preprocessing already applied to images on disk
    (crop black borders -> pad square -> resize -> Ben Graham normalization),
    matching the preprocessing pipeline built earlier for model 1."""

    def __init__(self, dataframe, img_dir, transform):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        image = np.array(Image.open(img_path).convert("RGB"))
        label = int(row["diagnosis"])
        augmented = self.transform(image=image)
        return augmented["image"], label, row["id_code"]


train_dataset = DRDataset(train_df, IMG_DIR, train_transform)
val_dataset = DRDataset(val_df, IMG_DIR, val_transform)

class_counts = train_df["diagnosis"].value_counts().sort_index()
class_weights = 1.0 / class_counts
sample_weights = train_df["diagnosis"].map(class_weights).values
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
)

# ---------------------------------------------------------------------------
# CORN loss + prediction (Shi et al. 2021)
# ---------------------------------------------------------------------------
def corn_loss(logits, y_true, num_classes):
    """
    logits: (batch, num_classes-1) raw scores, one per ordinal threshold
    y_true: (batch,) integer labels in [0, num_classes-1]
    Conditional training: threshold task k only trains on examples whose true
    rank is >= k, which is what guarantees rank-monotonic predictions at
    inference (the CORN property CORAL lacks).
    """
    total_loss = 0.0
    total_examples = 0
    for k in range(num_classes - 1):
        mask = y_true > (k - 1)  # examples eligible for this conditional task
        if mask.sum() == 0:
            continue
        target_k = (y_true[mask] > k).float()
        pred_k = logits[mask, k]
        # numerically stable binary cross-entropy on logits
        loss_k = -torch.sum(
            F.logsigmoid(pred_k) * target_k
            + (F.logsigmoid(pred_k) - pred_k) * (1 - target_k)
        )
        total_loss += loss_k
        total_examples += mask.sum().item()
    return total_loss / max(total_examples, 1)


def corn_predict(logits):
    """Decode CORN logits using cumulative conditional probabilities.

    Each CORN output is P(y > k | y > k - 1). Multiplying the conditional
    probabilities gives P(y > k), which must be used for rank-consistent
    ordinal decoding.
    """
    conditional_probas = torch.sigmoid(logits)
    cumulative_probas = torch.cumprod(conditional_probas, dim=1)
    preds = (cumulative_probas > 0.5).sum(dim=1)
    return preds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Linear(backbone.fc.in_features, NUM_CLASSES - 1)
    return backbone


model = build_model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(loader, train_mode):
    model.train() if train_mode else model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            if train_mode:
                optimizer.zero_grad()

            logits = model(images)
            loss = corn_loss(logits, labels, NUM_CLASSES)

            if train_mode:
                loss.backward()
                optimizer.step()

            preds = corn_predict(logits)
            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    qwk = cohen_kappa_score(all_true, all_preds, weights="quadratic")
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_true, all_preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
    )
    return avg_loss, qwk, macro_f1, per_class_f1, all_preds, all_true


best_val_qwk = -1.0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_qwk, train_f1, _, _, _ = run_epoch(train_loader, train_mode=True)
    val_loss, val_qwk, val_f1, val_f1_by_class, val_preds, val_true = run_epoch(
        val_loader, train_mode=False
    )
    scheduler.step()
    elapsed = time.time() - t0

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} train_qwk={train_qwk:.4f} train_macro_f1={train_f1:.4f} | "
        f"val_loss={val_loss:.4f} val_qwk={val_qwk:.4f} val_macro_f1={val_f1:.4f} | "
        f"{elapsed:.1f}s"
    )
    print(f"  val_f1_by_class={[round(float(x), 4) for x in val_f1_by_class]}")

    if val_qwk > best_val_qwk:
        best_val_qwk = val_qwk
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"  -> saved new best checkpoint (val_qwk={val_qwk:.4f})")

print(f"\nBest val QWK: {best_val_qwk:.4f}")

# ---------------------------------------------------------------------------
# Final submission export (reload best checkpoint, predict on val split)
# ---------------------------------------------------------------------------
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

ids, preds = [], []
with torch.no_grad():
    for images, labels, id_codes in val_loader:
        images = images.to(device)
        logits = model(images)
        batch_preds = corn_predict(logits).cpu().numpy().tolist()
        preds.extend(batch_preds)
        ids.extend(id_codes)

submission = pd.DataFrame({"id_code": ids, "diagnosis": preds})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
