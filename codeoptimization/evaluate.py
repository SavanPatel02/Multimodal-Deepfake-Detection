import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger_setup import setup_logger
from multimodal_dataset import MultimodalDataset
from multimodal_model import MultimodalDeepfakeModel

# ==========================
# CONFIG
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

log = setup_logger("evaluate")
log.info(f"Using device: {DEVICE}")

# ==========================
# LOAD TEST DATASET
# FIX: pass split="test" so no training augmentations are applied
# ==========================

test_dataset = MultimodalDataset(
    csv_file=os.path.join(BASE_DIR, "Data/processed/test_subset.csv"),
    frames_root=os.path.join(BASE_DIR, "Data/processed/frames/test"),
    mel_root=os.path.join(BASE_DIR, "Data/processed/mels/test"),
    split="test"   # FIX: was missing, defaulted to "train" → augmentations during eval
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==========================
# LOAD MODEL
# FIX: weights_only=True to avoid arbitrary code execution on checkpoint load
# ==========================

model = MultimodalDeepfakeModel().to(DEVICE)
model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "multimodal_model.pth"),
        map_location=DEVICE,
        weights_only=True   # FIX: security + deprecation
    )
)
model.eval()

# ==========================
# EVALUATION
# ==========================

all_preds = []
all_labels = []

with torch.no_grad():
    for frames, mel, labels in test_loader:
        frames = frames.to(DEVICE)
        mel = mel.to(DEVICE)

        outputs = model(frames, mel)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ==========================
# METRICS
# FIX: zero_division=0 to suppress warnings when a class has no predictions
# ==========================

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

log.info("\n===== TEST RESULTS =====")
log.info(f"Accuracy : {acc * 100:.2f}%")
log.info(f"Precision: {precision:.4f}")
log.info(f"Recall   : {recall:.4f}")
log.info(f"F1 Score : {f1:.4f}")
log.info("\nConfusion Matrix:")
log.info(f"\n{cm}")
