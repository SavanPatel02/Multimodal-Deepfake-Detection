"""
Test all 10 trained models (M1-M10) on sample videos from the dataset.

Loads preprocessed frames (.npy) and mel spectrograms (.npy) directly,
runs each model, and prints a comparison table showing predictions vs ground truth.

Usage:
    python test_videos.py                    # default: 3 real + 3 fake from test subset
    python test_videos.py --num_samples 5    # 5 real + 5 fake
    python test_videos.py --split dev        # use dev set instead of test
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_pair import get_model

# ==========================
# CONFIG
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data", "Processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "subset")

MODELS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]

# ImageNet normalization (must match training pipeline)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
MEL_TIME_STEPS = 126

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# DATA LOADING
# ==========================

def load_frames(npy_path):
    """Load preprocessed face frames from .npy file."""
    frames = np.load(npy_path)  # (16, 224, 224, 3) uint8

    if len(frames) < 16:
        pad = np.repeat(frames[-1:], 16 - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    frames = frames[:16]

    # (16, 224, 224, 3) uint8 -> (16, 3, 224, 224) float32 normalized
    frames = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

    return torch.from_numpy(frames.copy()).unsqueeze(0).to(device)  # (1, 16, 3, 224, 224)


def load_mel(npy_path):
    """Load preprocessed mel spectrogram from .npy file."""
    mel = np.load(npy_path)  # (128, T)

    if mel.shape[1] < MEL_TIME_STEPS:
        mel = np.pad(mel, ((0, 0), (0, MEL_TIME_STEPS - mel.shape[1])))
    else:
        mel = mel[:, :MEL_TIME_STEPS]

    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 128, 126)
    return mel


# ==========================
# MODEL LOADING
# ==========================

def load_trained_model(model_name):
    """Load a trained model checkpoint."""
    checkpoint_path = os.path.join(RESULTS_DIR, f"{model_name}.pth")
    if not os.path.exists(checkpoint_path):
        return None

    model = get_model(model_name)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ==========================
# PREDICTION
# ==========================

def predict(model, frames, mel):
    """Run inference and return (predicted_label, confidence%)."""
    with torch.no_grad():
        logits = model(frames, mel)
        probs = F.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        return pred.item(), round(confidence.item() * 100, 2)


# ==========================
# MAIN
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Test all 10 models on sample videos")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of real + fake samples each (default: 3)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "dev", "train"], help="Dataset split to use")
    args = parser.parse_args()

    # Load CSV
    csv_path = os.path.join(DATA_DIR, f"{args.split}_subset.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(DATA_DIR, f"{args.split}.csv")
    df = pd.read_csv(csv_path)

    frames_dir = os.path.join(DATA_DIR, "frames", args.split)
    mels_dir = os.path.join(DATA_DIR, "mels", args.split)

    # Get video names and check availability
    df["name"] = df["video_path"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    df["has_frames"] = df["name"].apply(lambda n: os.path.exists(os.path.join(frames_dir, n + ".npy")))
    df["has_mels"] = df["name"].apply(lambda n: os.path.exists(os.path.join(mels_dir, n + ".npy")))
    df["has_both"] = df["has_frames"] & df["has_mels"]

    # Pick N real + N fake
    real_samples = df[(df["label"] == 0) & df["has_both"]].head(args.num_samples)
    fake_samples = df[(df["label"] == 1) & df["has_both"]].head(args.num_samples)
    samples = pd.concat([real_samples, fake_samples]).reset_index(drop=True)

    n = len(samples)
    print(f"\n{'='*80}")
    print(f"  TESTING {n} VIDEOS ({len(real_samples)} Real + {len(fake_samples)} Fake) from {args.split} set")
    print(f"  Device: {device}")
    print(f"{'='*80}\n")

    # Load all models
    print("Loading models...")
    loaded_models = {}
    for m in MODELS:
        model = load_trained_model(m)
        if model is not None:
            loaded_models[m] = model
            print(f"  {m}: loaded")
        else:
            print(f"  {m}: MISSING checkpoint")
    print()

    # Run predictions
    # results[video_idx][model_name] = (pred_label, confidence)
    results = []

    for idx, row in samples.iterrows():
        name = row["name"]
        true_label = row["label"]
        true_str = "REAL" if true_label == 0 else "FAKE"

        frames_path = os.path.join(frames_dir, name + ".npy")
        mel_path = os.path.join(mels_dir, name + ".npy")

        print(f"Processing {name}.mp4 (Ground Truth: {true_str})...")

        frames = load_frames(frames_path)
        mel = load_mel(mel_path)

        video_results = {"name": name, "true_label": true_label}
        for m_name, model in loaded_models.items():
            pred, conf = predict(model, frames, mel)
            video_results[m_name] = (pred, conf)

        results.append(video_results)

    # ==========================
    # PRINT RESULTS TABLE
    # ==========================
    print(f"\n{'='*120}")
    print(f"  PREDICTION RESULTS — ALL 10 MODELS")
    print(f"{'='*120}")

    # Header
    header = f"{'Video':<14} {'Truth':<7}"
    for m in loaded_models:
        header += f" | {m:^14}"
    print(header)
    print("-" * len(header))

    correct_counts = {m: 0 for m in loaded_models}
    total = len(results)

    for r in results:
        true_label = r["true_label"]
        true_str = "REAL" if true_label == 0 else "FAKE"
        line = f"{r['name']:<14} {true_str:<7}"

        for m in loaded_models:
            pred, conf = r[m]
            pred_str = "REAL" if pred == 0 else "FAKE"
            match = "OK" if pred == true_label else "XX"
            if pred == true_label:
                correct_counts[m] += 1
            line += f" | {pred_str} {conf:5.1f}% {match}"

        print(line)

    # Accuracy summary
    print("-" * len(header))
    acc_line = f"{'Accuracy':<14} {'':7}"
    for m in loaded_models:
        acc = correct_counts[m] / total * 100
        acc_line += f" | {acc:>6.1f}%       "
    print(acc_line)

    # Per-model summary
    print(f"\n{'='*60}")
    print(f"  MODEL ACCURACY SUMMARY ({total} videos)")
    print(f"{'='*60}")
    print(f"  {'Model':<8} {'Correct':>8} {'Accuracy':>10}  Status")
    print(f"  {'-'*45}")

    for m in loaded_models:
        c = correct_counts[m]
        acc = c / total * 100
        status = "PERFECT" if c == total else f"{total - c} wrong"
        print(f"  {m:<8} {c:>4}/{total:<4} {acc:>8.1f}%  {status}")

    print()


if __name__ == "__main__":
    main()
