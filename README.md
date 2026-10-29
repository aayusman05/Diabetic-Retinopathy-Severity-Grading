# Diabetic Retinopathy Severity Grading

Deep-learning experiments for five-class diabetic-retinopathy severity grading on the APTOS-style dataset. The project includes an EfficientNet-B0 soft-QWK model, a ResNet-18 CORN ordinal model, and a Streamlit dashboard with Grad-CAM++ and Layer-CAM explanations.

## Project layout

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   ├── train.csv
│   └── train_images/          # local dataset; ignored by Git
├── models/                    # local .pt checkpoints; ignored by Git
├── outputs/                   # predictions and evaluation figures; ignored by Git
├── src/
│   ├── gradcam.py
│   ├── preprocessing.py
│   ├── train_convnext_corn.py
│   ├── train_efficientnet_b0.py
│   └── train_resnet18_corn.py
├── .gitignore
├── README.md
└── requirements.txt
```

The deterministic split uses 3,112 training images and 550 validation images from 3,662 labeled images. Both models use 224px inputs, batch size 8, MPS when available, and zero DataLoader workers for M1 stability.

## Setup

```bash
cd "/Users/aayush/Documents/vscode/DL Project"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Place the dataset at `data/train.csv` and `data/train_images/`.

## Train EfficientNet-B0

```bash
unset FAST_TRAIN MAX_TRAIN_SAMPLES MAX_VAL_SAMPLES TRAIN_SUBSAMPLE
unset IMG_SIZE BATCH_SIZE EPOCHS NUM_WORKERS LR DATA_DIR IMG_DIR
python3 src/train_efficientnet_b0.py
```

## Train ResNet-18 + CORN

```bash
unset FAST_TRAIN MAX_TRAIN_SAMPLES MAX_VAL_SAMPLES TRAIN_SUBSAMPLE
unset IMG_SIZE BATCH_SIZE EPOCHS NUM_WORKERS LR DATA_DIR IMG_DIR
python3 src/train_resnet18_corn.py
```

Training creates checkpoints in `models/` and validation submissions in `outputs/submissions/`. Fast or subsampled runs must be explicitly enabled and should not be reported as full-data experiments.

## Train ConvNeXt-Tiny + CORN

This experiment uses the same deterministic split and CORN ordinal loss as
the ResNet-18 model, with a ConvNeXt-Tiny backbone:

```bash
python3 src/train_convnext_corn.py
```

The checkpoint is written to `models/convnext_tiny_corn_best.pt`, with
validation metrics in `outputs/metrics/convnext_tiny_corn_metrics.json` and
predictions in `outputs/submissions/submission_convnext_tiny_corn.csv`.
The script can be run from any directory because it resolves paths from the
project root. For a short debugging run, use environment overrides such as:

```bash
EPOCHS=1 BATCH_SIZE=2 python3 src/train_convnext_corn.py
```

## Run the dashboard

Train the desired checkpoint first, then run:

```bash
streamlit run app/streamlit_app.py
```

The dashboard supports both checkpoint paths:

- `models/efficientnet_b0_qwk_best.pt`
- `models/resnet18_corn_best.pt`

## Generate CAM explanations

```bash
python3 src/gradcam.py \
  --arch efficientnet \
  --checkpoint models/efficientnet_b0_qwk_best.pt \
  --image data/train_images/46923eea9a4e.png \
  --out_dir cam_outputs \
  --method both
```

Use `--arch resnet` with `models/resnet18_corn_best.pt` after the ResNet model has been trained.

## GitHub note

The dataset images, checkpoints, generated outputs, and CAM images are intentionally ignored because they are large or generated artifacts. Commit the source code, README, dependency file, and a small example/configuration only; do not commit patient/image data unless distribution rights and repository policy allow it.
