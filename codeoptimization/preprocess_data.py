"""
Full dataset preprocessing pipeline — runs all steps in order.
Processes the ENTIRE dataset (no subsets).

Steps:
  1. Parse metadata  → train.csv, dev.csv, test.csv
  2. Extract audio   → .wav files (16kHz mono)
  3. Generate mels   → .npy mel spectrograms
  4. Extract frames  → 16 face crops per video (.jpg)

Usage:
    python codeoptimization/preprocess_data.py

    # Run only specific steps:
    python codeoptimization/preprocess_data.py --steps audio mels
    python codeoptimization/preprocess_data.py --steps frames

    # Resume (skip already processed files):
    All steps auto-skip already-processed files by default.

Available step names: metadata, audio, mels, frames
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import cv2
import torch
from tqdm import tqdm
from moviepy import VideoFileClip
from facenet_pytorch import MTCNN
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger_setup import setup_logger

# ==============================
# PATHS
# ==============================

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(BASE_DIR, "Data", "raw", "LAV-DF")
PROCESSED    = os.path.join(BASE_DIR, "Data", "processed")
AUDIO_DIR    = os.path.join(PROCESSED, "audio")
MELS_DIR     = os.path.join(PROCESSED, "mels")
FRAMES_DIR   = os.path.join(PROCESSED, "frames")

for d in [PROCESSED, AUDIO_DIR, MELS_DIR, FRAMES_DIR]:
    os.makedirs(d, exist_ok=True)

log = setup_logger("preprocess_data")

# ==============================
# SHARED SETTINGS
# ==============================

SAMPLE_RATE      = 16000
DURATION         = 4            # seconds
TARGET_SAMPLES   = SAMPLE_RATE * DURATION   # 64000

N_FFT        = 1024
HOP_LENGTH   = 512
N_MELS       = 128

FRAMES_PER_VIDEO = 16
IMG_SIZE         = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================
# STEP 1 — PARSE METADATA
# ==============================

def step_metadata():
    log.info("\n" + "="*60)
    log.info("  STEP 1 — Parsing Metadata")
    log.info("="*60)

    metadata_path = os.path.join(RAW_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        log.info(f"  ERROR: metadata.json not found at {metadata_path}")
        return False

    with open(metadata_path) as f:
        metadata = json.load(f)

    train_data, dev_data, test_data = [], [], []

    log.info(f"  Total entries: {len(metadata)}")

    for entry in tqdm(metadata, desc="  Parsing"):
        filename = entry["file"]
        split    = entry["split"]

        is_fake = (
            entry.get("modify_video", False) or
            entry.get("modify_audio", False) or
            entry.get("n_fakes", 0) > 0
        )
        label      = 1 if is_fake else 0
        video_path = os.path.join(RAW_DIR, filename)

        row = {"video_path": video_path, "label": label, "split": split}

        if split == "train":
            train_data.append(row)
        elif split == "dev":
            dev_data.append(row)
        elif split == "test":
            test_data.append(row)

    for data, name in [(train_data, "train"), (dev_data, "dev"), (test_data, "test")]:
        df   = pd.DataFrame(data)
        path = os.path.join(PROCESSED, f"{name}.csv")
        df.to_csv(path, index=False)

        real  = (df["label"] == 0).sum()
        fake  = (df["label"] == 1).sum()
        log.info(f"  {name}.csv → {len(df)} samples  (real: {real}, fake: {fake})")

    log.info("  Metadata parsing complete.")
    return True


# ==============================
# STEP 2 — EXTRACT AUDIO
# ==============================

def extract_audio_from_video(video_path, output_path):
    try:
        clip  = VideoFileClip(video_path)
        audio = clip.audio

        if audio is None:
            clip.close()
            return False

        temp_wav = output_path.replace(".wav", "_temp.wav")
        audio.write_audiofile(temp_wav, logger=None)

        y, _ = librosa.load(temp_wav, sr=SAMPLE_RATE, mono=True)
        sf.write(output_path, y, SAMPLE_RATE)

        os.remove(temp_wav)
        clip.close()
        return True

    except Exception as e:
        log.info(f"  Error (audio) {os.path.basename(video_path)}: {e}")
        return False


def step_audio():
    log.info("\n" + "="*60)
    log.info("  STEP 2 — Extracting Audio")
    log.info("="*60)

    errors = 0

    for split in ["train", "dev", "test"]:
        csv_path = os.path.join(PROCESSED, f"{split}.csv")
        if not os.path.exists(csv_path):
            log.info(f"  Skipping {split} — {split}.csv not found. Run metadata step first.")
            continue

        df         = pd.read_csv(csv_path)
        out_dir    = os.path.join(AUDIO_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        pending = []
        for _, row in df.iterrows():
            name   = os.path.splitext(os.path.basename(row["video_path"]))[0]
            out_path = os.path.join(out_dir, name + ".wav")
            if not os.path.exists(out_path):
                pending.append((row["video_path"], out_path))

        log.info(f"\n  {split}: {len(df)} total | {len(pending)} to process "
                 f"| {len(df) - len(pending)} already done")

        for video_path, out_path in tqdm(pending, desc=f"  {split}"):
            if not extract_audio_from_video(video_path, out_path):
                errors += 1

    log.info(f"\n  Audio extraction complete. Errors: {errors}")
    return errors == 0


# ==============================
# STEP 3 — GENERATE MEL SPECTROGRAMS
# ==============================

def generate_mel(wav_path):
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

    # Normalize amplitude FIRST (same order as training preprocessing)
    y = y / (np.max(np.abs(y)) + 1e-6)

    # Trim or pad to exactly TARGET_SAMPLES
    if len(y) > TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]
    else:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    return mel_db.astype(np.float32)


def step_mels():
    log.info("\n" + "="*60)
    log.info("  STEP 3 — Generating Mel Spectrograms")
    log.info("="*60)

    errors = 0

    for split in ["train", "dev", "test"]:
        in_dir  = os.path.join(AUDIO_DIR, split)
        out_dir = os.path.join(MELS_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(in_dir):
            log.info(f"  Skipping {split} — audio not found. Run audio step first.")
            continue

        wav_files = [f for f in os.listdir(in_dir) if f.endswith(".wav")]
        pending   = [f for f in wav_files
                     if not os.path.exists(os.path.join(out_dir, f.replace(".wav", ".npy")))]

        log.info(f"\n  {split}: {len(wav_files)} total | {len(pending)} to process "
                 f"| {len(wav_files) - len(pending)} already done")

        for fname in tqdm(pending, desc=f"  {split}"):
            in_path  = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname.replace(".wav", ".npy"))
            try:
                mel = generate_mel(in_path)
                np.save(out_path, mel)
            except Exception as e:
                log.info(f"  Error (mel) {fname}: {e}")
                errors += 1

    log.info(f"\n  Mel generation complete. Errors: {errors}")
    return errors == 0


# ==============================
# STEP 4 — EXTRACT FRAMES
# ==============================

def extract_frames_from_video(video_path, output_path, detector):
    """Extract FRAMES_PER_VIDEO face crops and save as a single .npy (uint8).

    Strategy: detect face once on the middle frame, then apply that bounding box
    to all 16 frames.  Deepfake videos have a fixed face position throughout, so
    one detection is sufficient and avoids 15 redundant MTCNN GPU calls per video.
    Frames are read by seeking directly (no sequential decode waste).
    """
    from PIL import Image as PILImage

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < FRAMES_PER_VIDEO:
        cap.release()
        return 0

    frame_indices = [
        int(i * total_frames / FRAMES_PER_VIDEO)
        for i in range(FRAMES_PER_VIDEO)
    ]

    # --- Step 1: detect face once on the middle frame ---
    mid_idx = frame_indices[len(frame_indices) // 2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return 0

    mid_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _    = detector.detect(PILImage.fromarray(mid_rgb))

    if boxes is None or len(boxes) == 0:
        cap.release()
        return 0

    areas           = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(x2), int(y2)

    # --- Step 2: seek to each target frame and apply saved crop ---
    saved_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = rgb[y1:y2, x1:x2]
            if face.size != 0:
                saved_frames.append(cv2.resize(face, (IMG_SIZE, IMG_SIZE)))

    cap.release()

    if not saved_frames:
        return 0

    while len(saved_frames) < FRAMES_PER_VIDEO:
        saved_frames.append(saved_frames[-1])

    np.save(output_path, np.stack(saved_frames[:FRAMES_PER_VIDEO]))
    return FRAMES_PER_VIDEO


def _read_mid_and_all_frames(args):
    """CPU-only: read all frames sequentially (videos are short ~125 frames),
    then pick 16 evenly-spaced ones. Sequential decode is much faster than
    17 random seeks on compressed H.264.
    """
    try:
        video_path, npy_path = args
        cap          = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < FRAMES_PER_VIDEO:
            cap.release()
            return None

        target_indices = set(
            int(i * total_frames / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)
        )
        mid_idx = int(total_frames / 2)

        # Sequential read — no seeking
        all_frames = {}
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in target_indices or idx == mid_idx:
                all_frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
        cap.release()

        if mid_idx not in all_frames:
            return None

        mid_rgb    = all_frames[mid_idx]
        frame_list = [all_frames[i] for i in sorted(target_indices) if i in all_frames]

        return (npy_path, mid_rgb, frame_list) if frame_list else None
    except Exception:
        return None


def step_frames():
    log.info("\n" + "="*60)
    log.info("  STEP 4 — Extracting Face Frames")
    log.info(f"  Device: {DEVICE}")
    log.info("="*60)

    if DEVICE == "cpu":
        log.warning(
            "  WARNING: CUDA not available — MTCNN is running on CPU.\n"
            "  Frame extraction will be very slow (10–15× slower than GPU).\n"
            "  Ensure your CUDA drivers are installed and torch.cuda.is_available() returns True."
        )

    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image as PILImage

    detector    = MTCNN(keep_all=True, device=DEVICE)
    IO_WORKERS  = 12   # CPU threads for video I/O (i9-14900K has 32 logical cores)
    DETECT_BATCH = 64  # MTCNN images per GPU call — light model, 16GB has headroom
    LOOKAHEAD   = DETECT_BATCH * 4  # keep enough futures ready to fill each batch
    errors      = 0

    for split in ["train", "dev", "test"]:
        csv_path = os.path.join(PROCESSED, f"{split}.csv")
        if not os.path.exists(csv_path):
            log.info(f"  Skipping {split} — {split}.csv not found.")
            continue

        df      = pd.read_csv(csv_path)
        out_dir = os.path.join(FRAMES_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        pending = []
        for _, row in df.iterrows():
            name     = os.path.splitext(os.path.basename(row["video_path"]))[0]
            npy_path = os.path.join(out_dir, name + ".npy")
            if not os.path.exists(npy_path):
                pending.append((row["video_path"], npy_path))

        log.info(f"\n  {split}: {len(df)} total | {len(pending)} to process "
                 f"| {len(df) - len(pending)} already done")

        # IO threads pre-read video frames; MTCNN runs batched on main thread.
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
            from collections import deque
            window       = deque()
            pending_iter = iter(pending)

            def refill():
                while len(window) < LOOKAHEAD:
                    try:
                        vp, npy = next(pending_iter)
                        window.append((vp, npy, pool.submit(_read_mid_and_all_frames, (vp, npy))))
                    except StopIteration:
                        break

            refill()

            pbar = tqdm(total=len(pending), desc=f"  {split}")
            while window:
                # Collect up to DETECT_BATCH ready results
                batch_results = []
                while window and len(batch_results) < DETECT_BATCH:
                    vp, npy_path_w, fut = window.popleft()
                    refill()
                    try:
                        result = fut.result()
                    except Exception:
                        result = None
                    if result is None:
                        errors += 1
                        pbar.update(1)
                        continue
                    batch_results.append(result)

                if not batch_results:
                    continue

                # Batch MTCNN detection — one GPU call for all mid frames
                mid_images = [PILImage.fromarray(r[1]) for r in batch_results]
                try:
                    batch_boxes, _ = detector.detect(mid_images)
                except Exception:
                    batch_boxes = [None] * len(batch_results)

                for (npy_path, mid_rgb, all_frames), boxes in zip(batch_results, batch_boxes):
                    pbar.update(1)
                    if boxes is None or len(boxes) == 0:
                        errors += 1
                        continue

                    areas           = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                    x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
                    x1, y1 = int(max(0, x1)), int(max(0, y1))
                    x2, y2 = int(x2), int(y2)

                    saved_frames = []
                    for rgb in all_frames:
                        face = rgb[y1:y2, x1:x2]
                        if face.size != 0:
                            saved_frames.append(cv2.resize(face, (IMG_SIZE, IMG_SIZE)))

                    if not saved_frames:
                        errors += 1
                        continue

                    while len(saved_frames) < FRAMES_PER_VIDEO:
                        saved_frames.append(saved_frames[-1])

                    try:
                        np.save(npy_path, np.stack(saved_frames[:FRAMES_PER_VIDEO]))
                    except Exception:
                        errors += 1

            pbar.close()

    log.info(f"\n  Frame extraction complete. Errors: {errors}")
    return True


# ==============================
# MAIN
# ==============================

STEP_MAP = {
    "metadata": step_metadata,
    "audio":    step_audio,
    "mels":     step_mels,
    "frames":   step_frames,
}

ALL_STEPS = ["metadata", "audio", "mels", "frames"]


def main():
    parser = argparse.ArgumentParser(description="Full dataset preprocessing pipeline")
    parser.add_argument(
        "--steps", nargs="+", default=ALL_STEPS,
        choices=ALL_STEPS,
        help="Steps to run (default: all)"
    )
    args = parser.parse_args()

    log.info("\n" + "="*60)
    log.info("  MULTIMODAL DEEPFAKE — FULL DATASET PREPROCESSING")
    log.info(f"  Dataset: {RAW_DIR}")
    log.info(f"  Output:  {PROCESSED}")
    log.info(f"  Steps:   {', '.join(args.steps)}")
    log.info("="*60)

    for step_name in args.steps:
        fn = STEP_MAP[step_name]
        fn()

    log.info("\n" + "="*60)
    log.info("  ALL PREPROCESSING COMPLETE")
    log.info("  You can now run training:")
    log.info("  python codeoptimization/train_pair.py --model M1")
    log.info("="*60 + "\n")


if __name__ == "__main__":
    main()
