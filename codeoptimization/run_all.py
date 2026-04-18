"""
Run all 10 models sequentially, time each one, and print a summary.

Usage:
    python codeoptimization/run_all.py

Logs: codeoptimization/logs/
Results: codeoptimization/results/
"""

import subprocess
import sys
import time
import os
from datetime import timedelta

MODELS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]

MODEL_NOTES = {
    "M1":  "ResNet18   + CNN audio     (baseline)",
    "M2":  "ResNet50   + CNN audio     (deeper visual)",
    "M3":  "ResNet18   + CNN audio v2  (pair 2a)",
    "M4":  "ResNet18   + LSTM audio    (temporal audio)",
    "M5":  "MobileNetV3 + Light CNN    (lightweight)",
    "M6":  "EfficientNet-B0 + CNN      (efficient)",
    "M7":  "ResNet18   + Concat fusion (pair 4a)",
    "M8":  "ResNet18   + Attention     (cross-attention)",
    "M9":  "ResNet18   visual only     (unimodal control)",
    "M10": "CNN audio  only            (unimodal control)",
}

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pair.py")
PYTHON = sys.executable


def fmt(seconds):
    return str(timedelta(seconds=int(seconds)))


def main():
    total_start = time.time()
    results = []

    print("=" * 60)
    print(f"  Running all {len(MODELS)} models")
    print(f"  Script : {SCRIPT}")
    print(f"  Python : {PYTHON}")
    print("=" * 60)

    for i, model in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Starting {model}  —  {MODEL_NOTES[model]}")
        print("-" * 60)

        t0 = time.time()
        proc = subprocess.run(
            [PYTHON, SCRIPT, "--model", model],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        elapsed = time.time() - t0

        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        results.append((model, elapsed, status))

        # Estimate remaining time based on average so far
        avg = sum(r[1] for r in results) / len(results)
        remaining = avg * (len(MODELS) - i)

        print(f"\n  {model} done in {fmt(elapsed)}  [{status}]")
        print(f"  Estimated time remaining: {fmt(remaining)}")

    total = time.time() - total_start

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for model, elapsed, status in results:
        print(f"  {model:<4}  {fmt(elapsed):>10}  {status}")
    print("-" * 60)
    print(f"  Total: {fmt(total)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
