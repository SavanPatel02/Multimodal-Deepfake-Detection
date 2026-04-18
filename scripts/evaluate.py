import os
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

from multimodal_dataset import MultimodalDataset
from multimodal_model import MultimodalDeepfakeModel

# ==========================
# CONFIG
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4

print("Using device:", DEVICE)

# ==========================
# LOAD TEST DATASET
# ==========================

test_dataset = MultimodalDataset(
    csv_file=os.path.join(BASE_DIR, "Data/processed/test_subset.csv"),
    frames_root=os.path.join(BASE_DIR, "Data/processed/frames/test"),
    mel_root=os.path.join(BASE_DIR, "Data/processed/mels/test")
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==========================
# LOAD MODEL
# ==========================

model = MultimodalDeepfakeModel().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "multimodal_model.pth")))
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
# ==========================

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n===== TEST RESULTS =====")
print(f"Accuracy : {acc * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)