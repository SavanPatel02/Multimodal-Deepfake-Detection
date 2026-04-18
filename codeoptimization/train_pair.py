"""
Unified training script for all 10 models across 5 pairs.

Usage:
    python train_pair.py --model M1
    python train_pair.py --model M4
    python train_pair.py --model M8

Results are saved to: results/<model_name>.json
Model checkpoints:    results/<model_name>.pth
"""

import os
import sys
import json
import argparse
import threading
import queue
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
# torchvision.transforms.v2 intentionally NOT used — incompatible with PyTorch 2.0.1+cu117

# Add codeoptimization to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger_setup import setup_logger

from multimodal_dataset import MultimodalDataset

# ==========================
# MODEL REGISTRY
# Maps model name → (module, class)
# ==========================

def get_model(name):
    name = name.upper()

    if name == "M1":
        from pair1_models import M1_ResNet18_CNN
        return M1_ResNet18_CNN()
    elif name == "M2":
        from pair1_models import M2_ResNet50_CNN
        return M2_ResNet50_CNN()
    elif name == "M3":
        from pair2_models import M3_ResNet18_CNNAudio
        return M3_ResNet18_CNNAudio()
    elif name == "M4":
        from pair2_models import M4_ResNet18_LSTMAudio
        return M4_ResNet18_LSTMAudio()
    elif name == "M5":
        from pair3_models import M5_MobileNetV3_LightCNN
        return M5_MobileNetV3_LightCNN()
    elif name == "M6":
        from pair3_models import M6_EfficientNetB0_CNN
        return M6_EfficientNetB0_CNN()
    elif name == "M7":
        from pair4_models import M7_ConcatFusion
        return M7_ConcatFusion()
    elif name == "M8":
        from pair4_models import M8_AttentionFusion
        return M8_AttentionFusion()
    elif name == "M9":
        from pair5_models import M9_VisualOnly
        return M9_VisualOnly()
    elif name == "M10":
        from pair5_models import M10_AudioOnly
        return M10_AudioOnly()
    else:
        raise ValueError(f"Unknown model: {name}. Choose from M1–M10.")


# ==========================
# CONFIG
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR      = os.path.join(BASE_DIR, "codeoptimization", "results")
RESULTS_DIR_SUB  = os.path.join(BASE_DIR, "codeoptimization", "results", "subset")
os.makedirs(RESULTS_DIR,     exist_ok=True)
os.makedirs(RESULTS_DIR_SUB, exist_ok=True)

BATCH_SIZES = {
    "M1":   64,
    "M2":   16,
    "M3":   64,
    "M4":   64,
    "M5":   96,
    "M6":   16,
    "M7":   64,
    "M8":   64,
    "M9":   64,
    "M10":  128,
}

EPOCHS             = 15
EPOCHS_SUBSET      = 10       # fewer epochs for subset quick-compare runs
LR                 = 5e-5
GRAD_CLIP          = 1.0
EARLY_STOP_PATIENCE      = 4
EARLY_STOP_PATIENCE_SUB  = 3  # tighter patience for subset runs
NUM_WORKERS        = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.90)  # hard limit: ~14.5 GB out of 16 GB

# ==========================
# PREFETCH LOADER
# One background thread reads the next batch from disk while
# the GPU processes the current batch. No worker processes —
# no RAM explosion, no spawn overhead, no pin_memory issues.
# ==========================

class PrefetchLoader:
    """Wraps a DataLoader with a single daemon thread that reads one batch
    ahead and moves it to GPU via a non-blocking CUDA stream."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self._len   = len(loader)

    def __len__(self):
        return self._len

    def __iter__(self):
        buf   = queue.Queue(maxsize=4)   # 4 batches ahead — overlaps transform+transfer
        stop  = object()                 # sentinel

        def reader(it):
            try:
                for batch in it:
                    buf.put(batch)
            finally:
                buf.put(stop)

        it     = iter(self.loader)
        thread = threading.Thread(target=reader, args=(it,), daemon=True)
        thread.start()

        stream = torch.cuda.Stream() if self.device == "cuda" else None

        while True:
            batch = buf.get()
            if batch is stop:
                break

            frames, mel, labels = batch

            if stream is not None:
                # Transfer to GPU asynchronously while main thread runs backward
                with torch.cuda.stream(stream):
                    frames = frames.to(self.device, non_blocking=True)
                    mel    = mel.to(self.device,    non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                # Ensure transfer is done before yielding
                torch.cuda.current_stream().wait_stream(stream)
            else:
                frames = frames.to(self.device)
                mel    = mel.to(self.device)
                labels = labels.to(self.device)

            yield frames, mel, labels

        thread.join()


# ==========================
# GPU AUGMENTATION
# Only random horizontal flip — implemented as a tensor slice (no memory copy,
# no HSV conversion, no v2 import).  Color jitter is handled on CPU in the
# dataset before normalization so pixel values are still in [0,1].
# Large batches (M10, batch=896) are skipped to avoid OOM.
# ==========================

def apply_gpu_aug(frames):
    """frames: (B, T, C, H, W) on GPU → maybe horizontally flipped."""
    # Skip for large batches (e.g. M10 batch=896 × 16 frames ≈ 8 GB flip copy)
    if frames.shape[0] > 256:
        return frames
    if torch.rand(1).item() > 0.5:
        # flip(-1) returns a new tensor but shares storage via negative strides
        frames = frames.flip(-1)
    return frames


# ==========================
# TRAINING LOOP
# ==========================

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for frames, mel, labels in tqdm(loader, desc="  Train", leave=False):
        # Data already on GPU via PrefetchLoader
        frames = apply_gpu_aug(frames)   # augment on GPU — fast
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE, enabled=DEVICE == "cuda"):
            outputs = model(frames, mel)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for frames, mel, labels in tqdm(loader, desc="  Val  ", leave=False):
            # Data already on GPU via PrefetchLoader
            with torch.amp.autocast(device_type=DEVICE, enabled=DEVICE == "cuda"):
                outputs = model(frames, mel)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return running_loss / len(loader), acc, f1


# ==========================
# MAIN
# ==========================

def main(model_name, subset=False):
    suffix = "_subset" if subset else ""
    log = setup_logger(f"train_pair_{model_name}{suffix}")

    results_dir = RESULTS_DIR_SUB if subset else RESULTS_DIR
    epochs      = EPOCHS_SUBSET   if subset else EPOCHS
    patience    = EARLY_STOP_PATIENCE_SUB if subset else EARLY_STOP_PATIENCE

    log.info(f"\n{'='*50}")
    log.info(f"  Training: {model_name}{'  [SUBSET]' if subset else ''}")
    log.info(f"  Device  : {DEVICE}")
    log.info(f"  Epochs  : {epochs}  |  Patience: {patience}")
    log.info(f"{'='*50}")

    proc_dir = os.path.join(BASE_DIR, "Data", "processed")
    train_csv = os.path.join(proc_dir, "train_subset.csv" if subset else "train.csv")
    dev_csv   = os.path.join(proc_dir, "dev_subset.csv"   if subset else "dev.csv")

    if subset and not os.path.exists(train_csv):
        log.error(f"  Subset CSVs not found. Run first:\n"
                  f"    python codeoptimization/create_subset.py")
        raise FileNotFoundError(train_csv)

    # Unimodal models skip loading unused data (saves ~9.7 MB/sample disk reads)
    MODALITY = {
        "M9":  "video",   # visual only — skip mel loading
        "M10": "audio",   # audio only  — skip frame loading
    }
    modality = MODALITY.get(model_name.upper(), "both")
    if modality != "both":
        log.info(f"  Modality: {modality} only (skipping unused {'mel' if modality == 'video' else 'frame'} loading)")

    # Datasets
    train_dataset = MultimodalDataset(
        csv_file=train_csv,
        frames_root=os.path.join(proc_dir, "frames", "train"),
        mel_root=os.path.join(proc_dir, "mels", "train"),
        split="train",
        modality=modality
    )
    dev_dataset = MultimodalDataset(
        csv_file=dev_csv,
        frames_root=os.path.join(proc_dir, "frames", "dev"),
        mel_root=os.path.join(proc_dir, "mels", "dev"),
        split="dev",
        modality=modality
    )

    batch_size = BATCH_SIZES[model_name.upper()]
    log.info(f"  Train samples: {len(train_dataset)}  |  Dev samples: {len(dev_dataset)}")
    log.info(f"  Batch size: {batch_size}")

    def make_loaders(bs):
        tl = PrefetchLoader(
            DataLoader(train_dataset, batch_size=bs, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=False),
            device=DEVICE
        )
        dl = PrefetchLoader(
            DataLoader(dev_dataset, batch_size=bs, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=False),
            device=DEVICE
        )
        return tl, dl

    train_loader, dev_loader = make_loaders(batch_size)

    # Class weights computed from the actual training CSV used
    train_df = pd.read_csv(train_csv)
    counts = train_df["label"].value_counts().sort_index().values
    weights = torch.tensor(1.0 / counts.astype(np.float32)).to(DEVICE)
    weights = weights / weights.sum()

    # Model, loss, optimizer, scheduler
    model = get_model(model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Parameters: {param_count:,}")

    # Training loop
    best_val_acc = 0
    best_val_f1 = 0
    epochs_no_improve = 0
    history = []
    start_epoch = 0

    best_model_path  = os.path.join(results_dir, f"{model_name}.pth")
    resume_ckpt_path = os.path.join(results_dir, f"{model_name}_resume.pth")

    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")

    # Resume from checkpoint if it exists
    if os.path.exists(resume_ckpt_path):
        log.info(f"\n  Resuming from checkpoint: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch        = ckpt["epoch"] + 1
        best_val_acc       = ckpt["best_val_acc"]
        best_val_f1        = ckpt["best_val_f1"]
        epochs_no_improve  = ckpt["epochs_no_improve"]
        history            = ckpt["history"]
        log.info(f"  Resumed at epoch {start_epoch} | Best val acc so far: {best_val_acc:.2f}%")

    for epoch in range(start_epoch, epochs):
        log.info(f"\nEpoch [{epoch+1}/{epochs}]  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Auto-reduce batch size on OOM instead of crashing
        while True:
            try:
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
                val_loss, val_acc, val_f1 = validate(model, dev_loader, criterion)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                log.warning(f"  OOM — reducing batch size to {batch_size} and retrying epoch")
                train_loader, dev_loader = make_loaders(batch_size)
        scheduler.step()
        torch.cuda.empty_cache()

        log.info(f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        log.info(f"  Val   → Loss: {val_loss:.4f}  | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 2),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 2),
            "val_f1":     round(val_f1, 4)
        })

        if val_f1 > best_val_f1:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            log.info(f"  Best model saved ({model_name}.pth)")
        else:
            epochs_no_improve += 1

        # Save resume checkpoint every epoch (overwrites previous)
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "best_val_acc":         best_val_acc,
            "best_val_f1":          best_val_f1,
            "epochs_no_improve":    epochs_no_improve,
            "history":              history,
        }, resume_ckpt_path)

        if epochs_no_improve >= patience:
            log.info(f"\n  Early stopping at epoch {epoch+1}.")
            break

    # Save results JSON
    results = {
        "model":        model_name,
        "parameters":   param_count,
        "best_val_acc": round(best_val_acc, 2),
        "best_val_f1":  round(best_val_f1, 4),
        "epochs_run":   len(history),
        "history":      history
    }

    results_path = os.path.join(results_dir, f"{model_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n  Results saved → {results_path}")
    log.info(f"  Best Val Acc : {best_val_acc:.2f}%")
    log.info(f"  Best Val F1  : {best_val_f1:.4f}")
    log.info(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deepfake detection model")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model to train: M1, M2, M3, M4, M5, M6, M7, M8, M9, M10"
    )
    parser.add_argument(
        "--subset", action="store_true",
        help="Use subset CSVs (train_subset.csv / dev_subset.csv) for quick comparison"
    )
    args = parser.parse_args()
    main(args.model, subset=args.subset)
