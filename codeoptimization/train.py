import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger_setup import setup_logger
from multimodal_dataset import MultimodalDataset
from multimodal_model import MultimodalDeepfakeModel

log = setup_logger("train")

# ==========================
# CONFIG
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BATCH_SIZE = 4
EPOCHS = 15
LR = 5e-5
GRAD_CLIP = 1.0          # FIX: gradient clipping max norm
EARLY_STOP_PATIENCE = 4  # FIX: stop if val acc doesn't improve for N epochs
NUM_WORKERS = 4          # FIX: parallel data loading (was 0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Using device: {DEVICE}")

# ==========================
# DATASETS
# ==========================

train_dataset = MultimodalDataset(
    csv_file=os.path.join(BASE_DIR, "Data/processed/train_subset.csv"),
    frames_root=os.path.join(BASE_DIR, "Data/processed/frames/train"),
    mel_root=os.path.join(BASE_DIR, "Data/processed/mels/train"),
    split="train"
)

dev_dataset = MultimodalDataset(
    csv_file=os.path.join(BASE_DIR, "Data/processed/dev_subset.csv"),
    frames_root=os.path.join(BASE_DIR, "Data/processed/frames/dev"),
    mel_root=os.path.join(BASE_DIR, "Data/processed/mels/dev"),
    split="dev"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True if DEVICE == "cuda" else False
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True if DEVICE == "cuda" else False
)

# ==========================
# CLASS WEIGHTS
# FIX: compute weights from training labels to handle class imbalance
# ==========================

import pandas as pd
import numpy as np

train_df = pd.read_csv(os.path.join(BASE_DIR, "Data/processed/train_subset.csv"))
class_counts = train_df["label"].value_counts().sort_index().values  # [count_real, count_fake]
class_weights = torch.tensor(
    1.0 / class_counts.astype(np.float32),
    dtype=torch.float32
).to(DEVICE)
class_weights = class_weights / class_weights.sum()  # normalize

log.info(f"Class weights → Real: {class_weights[0]:.4f}, Fake: {class_weights[1]:.4f}")

# ==========================
# MODEL
# ==========================

model = MultimodalDeepfakeModel().to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)  # FIX: weighted loss
optimizer = optim.Adam(model.parameters(), lr=LR)

# FIX: cosine annealing LR scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==========================
# TRAINING FUNCTION
# ==========================

def train_one_epoch():
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for frames, mel, labels in tqdm(train_loader):
        frames = frames.to(DEVICE)
        mel = mel.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(frames, mel)
        loss = criterion(outputs, labels)

        loss.backward()

        # FIX: gradient clipping before optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(train_loader), accuracy


# ==========================
# VALIDATION FUNCTION
# ==========================

def validate():
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, mel, labels in tqdm(dev_loader):
            frames = frames.to(DEVICE)
            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(frames, mel)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(dev_loader), accuracy


# ==========================
# MAIN TRAIN LOOP
# FIX: wrapped in __main__ guard — required on Windows with num_workers > 0
#      (without this, each worker process re-imports the module and
#       spawns more workers, causing an infinite process explosion)
# ==========================

if __name__ == "__main__":

    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):

        log.info(f"\nEpoch [{epoch+1}/{EPOCHS}]  LR: {scheduler.get_last_lr()[0]:.2e}")

        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()

        # Step scheduler after each epoch
        scheduler.step()

        log.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        log.info(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "multimodal_model.pth"))
            log.info("Best model saved!")
        else:
            epochs_no_improve += 1
            log.info(f"No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs.")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            log.info(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

    log.info("\nTraining completed.")
