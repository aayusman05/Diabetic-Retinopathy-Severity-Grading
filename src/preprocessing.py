import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------- 1. Crop black borders (circular fundus mask) ----------
def crop_black_border(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:  # fully black edge case
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

# ---------- 2. Ben Graham preprocessing (illumination normalization) ----------
def ben_graham(img, sigmaX=10):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX)
    return cv2.addWeighted(img, 4, blurred, -4, 128)

# ---------- 3. Pad to square (preserve aspect ratio, no stretching) ----------
def pad_to_square(img):
    h, w = img.shape[:2]
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))

IMG_SIZE = 224

def preprocess_image(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_black_border(img)
    img = pad_to_square(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = ben_graham(img, sigmaX=10)
    return img

# ---------- 4. Transforms: train vs val ----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),          # fundus images have no fixed "up" — safe
    A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=0,
             border_mode=cv2.BORDER_CONSTANT, p=0.3),
    A.GaussNoise(std_range=(0.01, 0.03), p=0.2),  # mild sensor-noise sim
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ---------- 5. Dataset ----------
class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row.id_code}.png"
        img = preprocess_image(img_path)  # crop, pad, resize, ben-graham
        if self.transform:
            img = self.transform(image=img)["image"]
        label = torch.tensor(row.diagnosis, dtype=torch.long)  # keep as ordinal int
        return img, label

# ---------- 6. Stratified split (no leakage) ----------
if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    image_dir = data_dir / "train_images"

    df = pd.read_csv(data_dir / "train.csv")
    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df.diagnosis, random_state=42
    )

    train_ds = DRDataset(train_df, image_dir, transform=train_transform)
    val_ds = DRDataset(val_df, image_dir, transform=val_transform)

    class_counts = train_df.diagnosis.value_counts().sort_index()
    class_weights = 1.0 / class_counts
    sample_weights = train_df.diagnosis.map(class_weights).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=8, sampler=sampler, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=0
    )
    print(f"Training images: {len(train_ds)} | Validation images: {len(val_ds)}")