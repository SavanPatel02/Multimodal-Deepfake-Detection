# Multimodal Deepfake Detection

A research system that detects deepfake videos by jointly analysing **face (visual)** and **audio** signals. The project benchmarks 10 model architectures across 5 research pairs on the [LAV-DF](https://github.com/ControlNet/LAV-DF) dataset to determine which architectural choices best improve detection accuracy.

---

## Overview

| Component | Details |
|-----------|---------|
| Dataset | LAV-DF (Large-scale Audio-Visual Deepfake) |
| Task | Binary classification — Real vs Fake |
| Modalities | Face frames (visual) + Mel spectrograms (audio) |
| Models | 10 architectures across 5 ablation pairs |
| Best result | 99.35% validation accuracy (M1 — ResNet18 + CNN) |
| GPU | NVIDIA RTX 4000 Ada Generation (16 GB VRAM) |

---

## Architecture

```
Video Input
    │
    ├── Face Extraction (MTCNN) ──► 16 frames per video
    │                                     │
    │                               Visual Encoder
    │                            (ResNet18 / ResNet50 /
    │                          MobileNetV3 / EfficientNet-B0)
    │                                     │
    │                               512-dim features
    │
    └── Audio Extraction (FFmpeg) ──► WAV (16kHz mono)
                                          │
                                    Mel Spectrogram
                                    (128×126, log scale)
                                          │
                                    Audio Encoder
                                  (CNN / LSTM / Lightweight)
                                          │
                                     128-dim features
                                          │
                              ┌───────────┴────────────┐
                              │        Fusion           │
                              │  (Concat / Cross-Attn)  │
                              └───────────┬────────────┘
                                          │
                                   Classifier (FC)
                                          │
                                   Real / Fake
```

---

## Research Pairs

| Pair | Models | Question |
|------|--------|----------|
| 1 | M1 (ResNet18) vs M2 (ResNet50) | Does a deeper visual backbone help? |
| 2 | M3 (CNN audio) vs M4 (LSTM audio) | Does temporal audio modelling beat 2D CNN? |
| 3 | M5 (MobileNetV3-Small) vs M6 (EfficientNet-B0) | Can lightweight models compete? |
| 4 | M7 (Concat fusion) vs M8 (Cross-attention fusion) | Does cross-attention beat simple concat? |
| 5 | M9 (Visual only) vs M10 (Audio only) | Is multimodal fusion better than unimodal? |

---

## Results (M1 — trained so far)

| Epoch | Train Acc | Val Acc | Val F1 |
|-------|-----------|---------|--------|
| 1 | 95.63% | 97.70% | 0.9846 |
| 2 | 98.86% | 99.25% | 0.9949 |
| 6 | 99.62% | **99.35%** | **0.9956** |

Models M2–M10 are defined and ready to train.

---

## Repository Structure

```
Multimodal Deepfake Detection/
│
├── scripts/                      ← Original baseline pipeline
│   ├── parse_metadata.py         ← Parse LAV-DF metadata.json
│   ├── create_subset.py          ← Build balanced train/dev/test subsets
│   ├── extract_audio.py          ← Extract WAV from MP4 (FFmpeg)
│   ├── generate_mels.py          ← Compute mel spectrograms (.npy)
│   ├── extract_frames.py         ← MTCNN face detection + frame export
│   ├── multimodal_dataset.py     ← PyTorch Dataset class
│   ├── multimodal_model.py       ← Baseline model (ResNet18 + CNN + Concat)
│   ├── fusion_model.py           ← Fusion layer variants
│   ├── train.py                  ← Baseline training script
│   ├── evaluate.py               ← Test-set evaluation
│   └── predict_video.py          ← Inference on a single video
│
├── codeoptimization/             ← 10-model research experiment
│   ├── pair1_models.py           ← M1 (ResNet18) and M2 (ResNet50)
│   ├── pair2_models.py           ← M3 (CNN audio) and M4 (LSTM audio)
│   ├── pair3_models.py           ← M5 (MobileNetV3) and M6 (EfficientNet-B0)
│   ├── pair4_models.py           ← M7 (Concat fusion) and M8 (Cross-attention)
│   ├── pair5_models.py           ← M9 (Visual only) and M10 (Audio only)
│   ├── train_pair.py             ← Unified training script for all models
│   ├── multimodal_dataset.py     ← Optimised dataset loader
│   ├── preprocess_data.py        ← Full preprocessing pipeline
│   ├── compare_results.py        ← Results table + 5 comparison plots
│   ├── run_all_training.py       ← Train all 10 models sequentially
│   ├── evaluate.py               ← Per-model test evaluation
│   ├── predict_video.py          ← Video inference (face + audio + fusion)
│   ├── dashboard.py              ← Interactive results dashboard
│   ├── results/
│   │   ├── M1.json               ← Training history and metrics
│   │   └── subset/               ← Subset experiment results (M1–M10)
│   └── requirements.txt
│
├── report_images/                ← Dashboard screenshots and pipeline diagrams
├── main.py                       ← Dependency version checker
├── check_gpu.py                  ← CUDA availability check
├── requirements.txt              ← Python dependencies
└── setup_env.bat                 ← One-click environment setup (Windows)
```

> **Dataset not included.** Download LAV-DF separately and place under `Data/raw/LAV-DF/`. See setup instructions below.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SavanPatel02/Multimodal-Deepfake-Detection.git
cd Multimodal-Deepfake-Detection
```

### 2. Create virtual environment and install dependencies

**Windows (one-click):**
```bat
setup_env.bat
```

**Manual:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 3. Verify GPU

```bash
python check_gpu.py
```

Expected output:
```
CUDA Available: True
GPU Name: NVIDIA RTX 4000 Ada Generation
```

### 4. Download and place the dataset

Download LAV-DF from the [official source](https://github.com/ControlNet/LAV-DF) and place it as:

```
Data/
└── raw/
    └── LAV-DF/
        ├── metadata.json
        └── test/          ← video files (.mp4)
```

---

## Data Preprocessing

Run the baseline preprocessing scripts in order:

```bash
python scripts/parse_metadata.py       # Parse metadata.json → CSV splits
python scripts/create_subset.py        # Build balanced 10k/2k/2k subset
python scripts/extract_audio.py        # Extract WAV files
python scripts/generate_mels.py        # Compute mel spectrograms
python scripts/extract_frames.py       # MTCNN face detection + frame export
```

Or run the optimised all-in-one pipeline:

```bash
python codeoptimization/preprocess_data.py
```

Run individual steps only:
```bash
python codeoptimization/preprocess_data.py --steps audio
python codeoptimization/preprocess_data.py --steps mels
python codeoptimization/preprocess_data.py --steps frames
```

---

## Training

### Baseline model

```bash
python scripts/train.py
```

### Research models (Pairs 1–5)

```bash
# Train a specific model
python codeoptimization/train_pair.py --model M1

# Train all 10 models sequentially
python codeoptimization/run_all_training.py --skip_existing
```

All models save to `codeoptimization/results/`:
- `M1.pth` — best model weights
- `M1.json` — accuracy, F1, full training history per epoch

---

## Model Comparison

After training one or more models:

```bash
python codeoptimization/compare_results.py
```

Generates a results table and 5 plots in `codeoptimization/results/plots/`:

| Plot | Description |
|------|-------------|
| `1_accuracy_f1.png` | All 10 models — accuracy and F1 side by side |
| `2_model_size.png` | Parameter count vs accuracy scatter |
| `3_learning_curves.png` | Train/val loss curves per model |
| `4_pair_comparison.png` | Each pair side by side, winner highlighted |
| `5_ranking.png` | Final ranking by accuracy |

---

## Inference

```bash
python codeoptimization/predict_video.py "path/to/video.mp4"
```

Output:
```
===== FACE ONLY =====
Prediction : Fake
Confidence : 88.42%

===== AUDIO ONLY =====
Prediction : Real
Confidence : 71.30%

===== FUSION =====
Prediction : Fake
Confidence : 93.17%
```

---

## Dashboard

```bash
python codeoptimization/dashboard.py
```

Interactive web dashboard showing training results, model comparisons, per-model details, and video upload for live prediction.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch + torchvision | Model training |
| facenet-pytorch (MTCNN) | Face detection |
| librosa | Mel spectrogram extraction |
| ffmpeg-python | Audio extraction from video |
| pandas / numpy | Data handling |
| scikit-learn | Metrics (F1, AUC) |
| matplotlib / plotly | Plots and dashboard |

Install all:
```bash
pip install -r requirements.txt
```

---

## Dataset

**LAV-DF** (Large-scale Audio-Visual Deepfake Dataset)

| Split | Subset used | Full dataset |
|-------|-------------|--------------|
| Train | 25,000  | 78,703 (73% fake) |
| Dev | 5,000  | 31,501 (74% fake) |
| Test | 2,000  | 26,100 (74% fake) |

Class imbalance in the full dataset is handled with weighted CrossEntropyLoss.

---

## Author

**Savan Patel** 
