"""
Model 1: EfficientNet-B0 + soft-QWK loss for DR severity grading (0-4).
APTOS 2019, ordinal regression via differentiable soft QWK loss.
Runs on M1 MPS (falls back to CPU/CUDA automatically).
"""

import os
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

# Resolve project data correctly when this script is run from src/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- Config ----------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "train_images")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "train.csv")
IMG_SIZE = 224
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))
NUM_CLASSES = 5
EPOCHS = int(os.getenv("EPOCHS", "20"))
LR = 3e-4
VAL_SPLIT = 0.15
SEED = 42
CKPT_PATH = os.path.join(PROJECT_ROOT, "models", "efficientnet_b0_qwk_best.pt")
SUBMISSION_PATH = os.path.join(PROJECT_ROOT, "outputs", "submissions", "submission_efficientnet_b0.csv")

if os.getenv("FAST_TRAIN", "0") == "1":
    print("FAST_TRAIN enabled: using a short run to generate model output in this environment.")

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
x = torch.randn(1).to(DEVICE)
print(f"Tensor device: {x.device}")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------- Dataset ----------------
class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        label = int(row["diagnosis"])
        return image, label


def get_transforms(train: bool):
    if train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------- Soft-QWK Loss ----------------
class SoftQWKLoss(nn.Module):
    """
    Differentiable approximation of Quadratic Weighted Kappa.
    Model outputs softmax over NUM_CLASSES; loss compares expected
    label distribution against true one-hot via QWK's weight matrix.
    Reference: de La Torre et al., "Kappa Loss for Ordinal Regression" (2018).
    """
    def __init__(self, num_classes=NUM_CLASSES, eps=1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        w = np.zeros((num_classes, num_classes), dtype=np.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer_workaround(w)

    def register_buffer_workaround(self, w):
        # keep as plain tensor, moved to device in forward
        self.weight_matrix = torch.tensor(w)

    def forward(self, logits, targets):
        device = logits.device
        W = self.weight_matrix.to(device)
        probs = F.softmax(logits, dim=1)                      # (B, C)
        targets_oh = F.one_hot(targets, self.num_classes).float()  # (B, C)

        # observed disagreement
        O = torch.matmul(targets_oh.T, probs)                 # (C, C)
        O = O / (O.sum() + self.eps)

        # expected disagreement (outer product of marginals)
        hist_true = targets_oh.sum(dim=0)
        hist_pred = probs.sum(dim=0)
        E = torch.outer(hist_true, hist_pred)
        E = E / (E.sum() + self.eps)

        num = (W * O).sum()
        den = (W * E).sum() + self.eps
        kappa = 1.0 - num / den
        return 1.0 - kappa  # minimize (1 - QWK)


# ---------------- Model ----------------
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


# ---------------- Train / Eval loops ----------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    qwk = cohen_kappa_score(all_targets, all_preds, weights="quadratic")
    per_class_f1 = f1_score(
        all_targets, all_preds, labels=list(range(NUM_CLASSES)),
        average=None, zero_division=0
    )
    macro_f1 = float(per_class_f1.mean())
    return avg_loss, qwk, macro_f1, per_class_f1, all_preds, all_targets


# ---------------- Main ----------------
def main():
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(
        df, test_size=VAL_SPLIT, stratify=df["diagnosis"], random_state=SEED
    )

    print(f"Training images: {len(train_df)} | Validation images: {len(val_df)}")
    print(f"Train classes: {sorted(train_df['diagnosis'].unique())}")
    print(f"Validation classes: {sorted(val_df['diagnosis'].unique())}")

    # WeightedRandomSampler for class imbalance
    class_counts = train_df["diagnosis"].value_counts().sort_index()
    class_weights = 1.0 / class_counts
    sample_weights = train_df["diagnosis"].map(class_weights).values
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    sampled_indices = list(iter(sampler))
    sampled_labels = train_df.iloc[sampled_indices]["diagnosis"]
    sampled_counts = sampled_labels.value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    print(f"Actual one-epoch sampler counts: {sampled_counts.to_dict()}")

    train_ds = APTOSDataset(train_df, DATA_DIR, transform=get_transforms(train=True))
    val_ds = APTOSDataset(val_df, DATA_DIR, transform=get_transforms(train=False))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = build_model().to(DEVICE)
    criterion = SoftQWKLoss(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_qwk = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_qwk, train_macro_f1, train_per_class_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_qwk, val_macro_f1, val_per_class_f1, val_preds, val_targets = run_epoch(
            model, val_loader, criterion
        )
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_qwk={train_qwk:.4f} "
              f"train_macro_f1={train_macro_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_qwk={val_qwk:.4f} "
              f"val_macro_f1={val_macro_f1:.4f} "
              f"val_f1_by_class={np.round(val_per_class_f1, 4).tolist()}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_qwk": val_qwk,
            }, CKPT_PATH)
            print(f"  -> saved new best checkpoint (val_qwk={val_qwk:.4f})")

    print(f"\nBest val QWK: {best_qwk:.4f}")

    # ---------------- Submission prep ----------------
    # Loads best checkpoint, runs inference on val set (swap val_df for real
    # test set once test.csv/test images exist), writes submission CSV.
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, _, _, final_preds, _ = run_epoch(model, val_loader, criterion)
    submission = pd.DataFrame({
        "id_code": val_df["id_code"].values,
        "diagnosis": final_preds,
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission written to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
