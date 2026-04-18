import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from multimodal_dataset import MultimodalDataset
from multimodal_model import MultimodalDeepfakeModel

# ==========================
# CONFIG
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BATCH_SIZE = 4
EPOCHS = 15   
LR = 5e-5     

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ==========================
# DATASETS (UPDATED)
# ==========================

train_dataset = MultimodalDataset(
    csv_file=os.path.join(BASE_DIR, "Data/processed/train_subset.csv"),
    frames_root=os.path.join(BASE_DIR, "Data/processed/frames/train"),
    mel_root=os.path.join(BASE_DIR, "Data/processed/mels/train"),
    split="train"   # 🔥 IMPORTANT
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
    num_workers=0
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================
# MODEL
# ==========================

model = MultimodalDeepfakeModel().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

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
# ==========================

best_val_acc = 0  

for epoch in range(EPOCHS):

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(BASE_DIR, "multimodal_model.pth"))
        print("Best model saved!")

print("\nTraining completed.")