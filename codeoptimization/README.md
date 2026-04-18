# Multimodal Deepfake Detection — How to Run

## What This Project Does

Detects whether a video is **real or fake (deepfake)** by analysing both the **face** and the **audio** together using 10 different model architectures split into 5 research pairs.

---

## Current Status (as of 2026-04-06)

### M1 Training Results (only model trained so far)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | Notes |
|-------|-----------|-----------|----------|---------|--------|-------|
| 1 | 0.1248 | 95.63% | 0.1641 | 97.70% | 0.9846 | |
| 2 | 0.0480 | 98.86% | 0.0233 | 99.25% | 0.9949 | Best saved |
| 3 | 0.0342 | 99.28% | 0.0613 | 99.14% | 0.9942 | |
| 4 | 0.0278 | 99.42% | 0.2151 | 94.23% | 0.9593 | |
| 5 | 0.0210 | 99.56% | 0.2033 | 94.16% | 0.9587 | |
| 6 | 0.0175 | 99.62% | 0.0409 | **99.35%** | **0.9956** | **Best saved** |
| 7+ | — | — | — | — | — | Power cut — training stopped |

**Best M1 checkpoint**: `results/M1.pth` → **99.35% Val Acc, F1 0.9956**

Models M2–M10: **not yet trained.**

---

## Issues Found & Fixed (2026-04-06)

### Issue 1 — Training took 2.5 hours per epoch
**Root cause**: `load_frames()` opened **16 separate JPEG files** per video sample from disk.
With 78k training samples that is 1.25 million file reads per epoch.
The dataset lives on a **spinning HDD (D: drive)** which can only do ~100–150 random
reads per second — so the GPU sat idle waiting for data.

**Fix applied** (`preprocess_data.py` + `multimodal_dataset.py`):
All 16 frames are now saved as **one `.npy` file** per video `(16, 224, 224, 3) uint8`.
Data loading drops from 16 disk reads → 1 disk read per sample.

**Further fix recommended** — move processed data to the NVMe SSD (C:, 808 GB free):
```bash
robocopy "D:\Multimodal Deepfake Detection\Data\processed" "C:\MLData\processed" /E /MOVE
mklink /D "D:\Multimodal Deepfake Detection\Data\processed" "C:\MLData\processed"
```

| Situation | Estimated epoch time |
|-----------|---------------------|
| HDD + 16 JPEGs (before fix) | ~2.5 h |
| HDD + single .npy (fixed) | ~25–40 min |
| SSD + single .npy (ideal) | ~5–10 min |

---

### Issue 2 — MTCNN face extraction ran on CPU
**Root cause**: During the April 2 preprocessing run, CUDA was not detected
(driver/PyTorch version mismatch) so MTCNN fell back to CPU — 10–15× slower.

**Fix applied** (`preprocess_data.py`):
Added an explicit warning that fires if `DEVICE == "cpu"` before MTCNN is created,
so the problem is visible immediately instead of silently running for hours.

**Root fix recommended** — upgrade PyTorch to match the Ada GPU:
```bash
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Current: `torch 2.0.1+cu117` — Ada Lovelace requires CUDA 11.8+.
After upgrade `torch.cuda.is_available()` will be reliable every run.

---

### Issue 3 — Test frame extraction was cut off by power outage
**Status**: The April 2 preprocessing log ends mid-way through the test split.
Test frames are partially extracted — some videos have no `.npy` yet.

**Action needed**: Re-run frame extraction after upgrading PyTorch:
```bash
venv\Scripts\python.exe codeoptimization/preprocess_data.py --steps frames
```
The script auto-skips already-extracted files so it only processes what's missing.

---

### Issue 4 — ~67 videos had no detectable face (zero-frame samples)
**Root cause**: MTCNN found no face in ~47 train / ~10 dev / ~9 test videos.
These were silently included in training with all-zero visual tensors,
polluting the visual backbone's gradient signal.

**Fix applied** (`multimodal_dataset.py`):
`__init__` now filters out any sample whose `.npy` file does not exist.
A warning is printed at dataset creation showing how many were dropped.

---

## What To Do Next (in order)

```
1. Upgrade PyTorch to CUDA 12.1
   venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

2. Re-run frame extraction (MTCNN on GPU this time — fast)
   venv\Scripts\python.exe codeoptimization/preprocess_data.py --steps frames

3. Move processed data to SSD for maximum training speed
   robocopy "D:\Multimodal Deepfake Detection\Data\processed" "C:\MLData\processed" /E /MOVE
   mklink /D "D:\Multimodal Deepfake Detection\Data\processed" "C:\MLData\processed"

4. Find optimal batch size for your 16 GB VRAM
   venv\Scripts\python.exe codeoptimization/find_batch_size.py
   → Update BATCH_SIZE in train_pair.py with the recommended value

5. Resume M1 (already at epoch 6, checkpoint saved in results/M1_resume.pth)
   venv\Scripts\python.exe codeoptimization/train_pair.py --model M1

6. Train remaining models M2–M10
   venv\Scripts\python.exe codeoptimization/train_pair.py --model M2
   ... (or use run_all_models.py to run all at once)

7. Compare all results
   venv\Scripts\python.exe codeoptimization/compare_results.py
```

---

---

## Dataset Status

The project uses the **LAV-DF** dataset. Currently only the balanced subset is preprocessed and ready to train on:

| Split | Subset (ready) | Full dataset |
|-------|---------------|--------------|
| Train | 10,000 (5k real / 5k fake) | 78,703 (73% fake) |
| Dev   | 2,000  (1k real / 1k fake) | 31,501 (74% fake) |
| Test  | 2,000  (1k real / 1k fake) | 26,100 (74% fake) |

**Use the subset for training now.** Run full preprocessing on the college PC when you have time (see Step 2B).

---

## Step 1 — Setup (Do this once)

Double-click `codeoptimization/setup_env.bat` or run from terminal:

```bash
cd "f:/Multimodal Deepfake Detection"
codeoptimization\setup_env.bat
```

Verify GPU is detected:
```bash
venv\Scripts\activate
python check_gpu.py
```

Expected:
```
CUDA Available: True
GPU Name: NVIDIA RTX 4000 Ada Generation
```

---

## Step 2A — Prepare Subset Data (Already Done)

The subset (10k/2k/2k) is already preprocessed and saved in `Data/processed/`. You can skip straight to Step 3.

Verify:
```
Data/processed/
  train_subset.csv    ← 10,000 rows (balanced)
  dev_subset.csv      ← 2,000 rows  (balanced)
  test_subset.csv     ← 2,000 rows  (balanced)
  frames/train/       ← 10,000 face image folders (16 frames each)
  frames/dev/         ← 2,000 folders
  frames/test/        ← 2,000 folders
  mels/train/         ← 10,000 .npy mel spectrograms
  mels/dev/           ← 2,000 .npy files
  mels/test/          ← 2,000 .npy files
  audio/train/        ← 10,000 .wav files
```

---

## Step 2B — Full Dataset Preprocessing (College PC, optional)

Only needed if you want to train on the full 78k dataset. This will take several hours.

```bash
venv\Scripts\activate
python codeoptimization/preprocess_data.py
```

Or double-click `codeoptimization/preprocess_data.bat`

Run specific steps only if some are already done:
```bash
python codeoptimization/preprocess_data.py --steps audio
python codeoptimization/preprocess_data.py --steps mels
python codeoptimization/preprocess_data.py --steps frames
```

After full preprocessing completes, change `train_subset.csv` to `train.csv` in `train_pair.py` line 97 and `train.py` line 29.

---

## Step 3 — Train the Pairs

Run from the project root with the virtual environment active:

```bash
cd "f:/Multimodal Deepfake Detection"
venv\Scripts\activate
```

Each model auto-saves to:
- `codeoptimization/results/M1.pth` — best model weights
- `codeoptimization/results/M1.json` — accuracy, F1, full training history

---

### Pair 1 — Does a deeper visual backbone help?

| Model | Visual | Audio | Difference |
|-------|--------|-------|-----------|
| M1 | ResNet18 | Custom CNN | Baseline — lightweight, 512-dim features |
| M2 | ResNet50 | Custom CNN | Deeper — 2048-dim features, more capacity |

```bash
python codeoptimization/train_pair.py --model M1
python codeoptimization/train_pair.py --model M2
```

---

### Pair 2 — CNN vs LSTM for audio?

| Model | Visual | Audio | Difference |
|-------|--------|-------|-----------|
| M3 | ResNet18 | CNN Audio | Treats mel as 2D image (frequency patterns) |
| M4 | ResNet18 | LSTM Audio | Treats mel as time sequence (temporal patterns) |

```bash
python codeoptimization/train_pair.py --model M3
python codeoptimization/train_pair.py --model M4
```

---

### Pair 3 — Can lightweight models compete?

| Model | Visual | Audio | Difference |
|-------|--------|-------|-----------|
| M5 | MobileNetV3-Small | Lightweight CNN | Smallest and fastest model |
| M6 | EfficientNet-B0 | Custom CNN | Efficient with stronger features |

```bash
python codeoptimization/train_pair.py --model M5
python codeoptimization/train_pair.py --model M6
```

---

### Pair 4 — Does cross-attention beat simple concatenation?

| Model | Visual | Audio | Difference |
|-------|--------|-------|-----------|
| M7 | ResNet18 | Custom CNN | Simple concat — stacks features together |
| M8 | ResNet18 | Custom CNN | Cross-attention — each modality queries the other |

```bash
python codeoptimization/train_pair.py --model M7
python codeoptimization/train_pair.py --model M8
```

---

### Pair 5 — Is multimodal better than unimodal?

| Model | Visual | Audio | Difference |
|-------|--------|-------|-----------|
| M9 | ResNet18 only | — | Face signal only, ignores audio |
| M10 | — | Custom CNN only | Audio signal only, ignores video |

```bash
python codeoptimization/train_pair.py --model M9
python codeoptimization/train_pair.py --model M10
```

These are the control models. If M1–M8 don't beat M9 and M10, fusion isn't helping.

---

## Step 4 — Compare All Results

After training any models:

```bash
python codeoptimization/compare_results.py
```

### Console output shows:

```
Model  Pair     Visual           Audio            Fusion        Val Acc   Val F1      Params
M1     Pair 1   ResNet18         Custom CNN       Concat         87.50%   0.8812  11,182,210
M2     Pair 1   ResNet50         Custom CNN       Concat         89.10%   0.8945  24,557,698
...

Pair 1 — Does deeper visual backbone help?
  M1: 87.50% | M2: 89.10% — Winner: M2 (+1.60%)

Overall Ranking:
  #1  M8  91.20%  Cross-Attention Fusion
  #2  M6  90.50%  EfficientNet-B0
  ...
```

### Plots saved to `codeoptimization/results/plots/`:

| File | What it shows |
|------|--------------|
| `1_accuracy_f1.png` | All 10 models — accuracy and F1 side by side |
| `2_model_size.png` | Parameter count + accuracy vs model size scatter |
| `3_learning_curves.png` | Train/val loss and accuracy per epoch for each model |
| `4_pair_comparison.png` | Each pair side by side, winner highlighted in gold |
| `5_ranking.png` | Final overall ranking by accuracy |

---

## Step 5 — Train All 10 at Once (Optional)

```bash
# Train all 10 then auto-generate comparison and plots
python codeoptimization/run_all_models.py
```

Resume if interrupted (skips already-finished models):
```bash
python codeoptimization/run_all_models.py --skip_existing
```

Or double-click `codeoptimization/run_all_models.bat`

---

## Step 6 — Predict on a Video

```bash
python codeoptimization/predict_video.py "path\to\video.mp4"
```

Output:
```
===== FACE ONLY =====
Prediction : Fake
Confidence : 88.42 %

===== AUDIO ONLY =====
Prediction : Real
Confidence : 71.30 %

===== FUSION =====
Prediction : Fake
Confidence : 93.17 %
```

---

## Quick Reference — All Commands

```bash
# One-time setup
codeoptimization\setup_env.bat

# Subset preprocessing (already done — skip if Data/processed/ has files)
python scripts/parse_metadata.py
python scripts/create_subset.py
python scripts/extract_audio.py
python scripts/generate_mels.py
python scripts/extract_frames.py

# Full dataset preprocessing (college PC only, takes hours)
python codeoptimization/preprocess_data.py

# Train individual models
python codeoptimization/train_pair.py --model M1    # Pair 1 — ResNet18 baseline
python codeoptimization/train_pair.py --model M2    # Pair 1 — ResNet50 deeper
python codeoptimization/train_pair.py --model M3    # Pair 2 — CNN audio
python codeoptimization/train_pair.py --model M4    # Pair 2 — LSTM audio
python codeoptimization/train_pair.py --model M5    # Pair 3 — MobileNetV3
python codeoptimization/train_pair.py --model M6    # Pair 3 — EfficientNet-B0
python codeoptimization/train_pair.py --model M7    # Pair 4 — Concat fusion
python codeoptimization/train_pair.py --model M8    # Pair 4 — Cross-attention
python codeoptimization/train_pair.py --model M9    # Pair 5 — Visual only
python codeoptimization/train_pair.py --model M10   # Pair 5 — Audio only

# Train all 10 at once
python codeoptimization/run_all_models.py --skip_existing

# Compare trained models + generate plots
python codeoptimization/compare_results.py

# Predict on a video
python codeoptimization/predict_video.py "path\to\video.mp4"
```

---

## Known Dataset Issues

| Issue | Status | Detail |
|-------|--------|--------|
| ~67 videos with no face detected | **Fixed** — filtered out in dataset `__init__` | MTCNN found no face; these are now excluded rather than trained with zero tensors |
| Test frame extraction incomplete | **Action needed** — re-run `--steps frames` | Power outage cut off April 2 preprocessing run mid-test-split |
| Full dataset imbalance (73% fake) | Handled | Class-weighted CrossEntropyLoss in `train_pair.py` compensates |
| Frame storage was 16 JPEGs per video | **Fixed** — now single `.npy` per video | Was the cause of 2.5h/epoch; new format is ~16× fewer disk reads |

---

## Folder Structure

```
Multimodal Deepfake Detection/
│
├── Data/
│   ├── raw/LAV-DF/              ← original dataset videos + metadata.json
│   └── processed/
│       ├── train_subset.csv     ← 10k balanced (use this for training now)
│       ├── dev_subset.csv       ← 2k balanced
│       ├── test_subset.csv      ← 2k balanced
│       ├── train.csv            ← 78k full (use after full preprocessing)
│       ├── frames/train/        ← face image folders (16 .jpg each)
│       ├── mels/train/          ← mel spectrograms (.npy, shape 128x126)
│       └── audio/train/         ← audio files (.wav, 16kHz mono)
│
├── codeoptimization/
│   ├── README.md                ← this file
│   ├── multimodal_model.py      ← baseline merged model (M1/M7 base)
│   ├── multimodal_dataset.py    ← dataset loader with fixes
│   ├── pair1_models.py          ← M1 (ResNet18) and M2 (ResNet50)
│   ├── pair2_models.py          ← M3 (CNN audio) and M4 (LSTM audio)
│   ├── pair3_models.py          ← M5 (MobileNetV3) and M6 (EfficientNet)
│   ├── pair4_models.py          ← M7 (Concat) and M8 (Cross-Attention)
│   ├── pair5_models.py          ← M9 (Visual only) and M10 (Audio only)
│   ├── train_pair.py            ← unified training script for all models
│   ├── train.py                 ← baseline-only training script
│   ├── evaluate.py              ← test set evaluation with metrics
│   ├── predict_video.py         ← run inference on a video file
│   ├── compare_results.py       ← comparison table + 5 plots
│   ├── run_all_models.py        ← trains all 10 models sequentially
│   ├── run_all_models.bat       ← one-click train all
│   ├── preprocess_data.py       ← full dataset preprocessing pipeline
│   ├── preprocess_data.bat      ← one-click full preprocessing
│   ├── setup_env.bat            ← creates venv + installs packages
│   ├── requirements.txt         ← Python dependencies
│   └── results/                 ← auto-created after first training run
│       ├── M1.json              ← accuracy, F1, training history
│       ├── M1.pth               ← saved model weights
│       └── plots/               ← generated comparison charts
│
├── scripts/                     ← original subset preprocessing scripts
├── check_gpu.py                 ← verify CUDA is available
└── multimodal_model.pth         ← best model checkpoint from train.py
```
