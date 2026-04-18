"""
Create stratified subset CSVs for fast model comparison.

Scans the frames directory to only include videos that have extracted frames,
then does a stratified sample preserving the real/fake ratio.

Output:
    Data/processed/train_subset.csv   (default 5000 samples)
    Data/processed/dev_subset.csv     (default 2000 samples)

Usage:
    python codeoptimization/create_subset.py
    python codeoptimization/create_subset.py --train-n 3000 --dev-n 1000
"""

import os
import sys
import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_subset(csv_path, frames_root, n, split_name, seed=42):
    df = pd.read_csv(csv_path)
    print(f"\n[{split_name}] Full CSV: {len(df)} rows")

    # Scan frames dir once — same logic as MultimodalDataset
    if not os.path.isdir(frames_root):
        print(f"  ERROR: frames dir not found: {frames_root}")
        sys.exit(1)

    dir_contents = os.listdir(frames_root)
    npy_set    = {f[:-4] for f in dir_contents if f.endswith(".npy")}
    folder_set = {f     for f in dir_contents if not f.endswith(".npy")}

    names       = df["video_path"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    has_frames  = names.isin(npy_set) | names.isin(folder_set)
    df_valid    = df[has_frames].reset_index(drop=True)

    print(f"  Valid (have frames): {len(df_valid)}")

    if len(df_valid) < n:
        print(f"  WARNING: only {len(df_valid)} valid samples — using all of them")
        subset = df_valid
    else:
        # Stratified sample: preserve label distribution
        subset = (
            df_valid
            .groupby("label", group_keys=False)
            .apply(lambda g: g.sample(frac=n / len(df_valid), random_state=seed))
            .reset_index(drop=True)
        )
        # Trim to exactly n if rounding gave more
        subset = subset.sample(n=min(n, len(subset)), random_state=seed).reset_index(drop=True)

    counts = subset["label"].value_counts().sort_index()
    print(f"  Subset size: {len(subset)}")
    print(f"  Label distribution: { {int(k): int(v) for k, v in counts.items()} }")
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=5000,
                        help="Training subset size (default: 5000)")
    parser.add_argument("--dev-n",   type=int, default=2000,
                        help="Dev subset size (default: 2000)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    processed_dir = os.path.join(BASE_DIR, "Data", "processed")

    train_subset = build_subset(
        csv_path    = os.path.join(processed_dir, "train.csv"),
        frames_root = os.path.join(processed_dir, "frames", "train"),
        n           = args.train_n,
        split_name  = "train",
        seed        = args.seed,
    )

    dev_subset = build_subset(
        csv_path    = os.path.join(processed_dir, "dev.csv"),
        frames_root = os.path.join(processed_dir, "frames", "dev"),
        n           = args.dev_n,
        split_name  = "dev",
        seed        = args.seed,
    )

    train_out = os.path.join(processed_dir, "train_subset.csv")
    dev_out   = os.path.join(processed_dir, "dev_subset.csv")

    train_subset.to_csv(train_out, index=False)
    dev_subset.to_csv(dev_out,   index=False)

    print(f"\n  Saved → {train_out}")
    print(f"  Saved → {dev_out}")
    print(f"\n  Run subset training:")
    print(f"    python codeoptimization/run_all_training.py --subset")
    print(f"  Or single model:")
    print(f"    python codeoptimization/train_pair.py --model M2 --subset")


if __name__ == "__main__":
    main()
