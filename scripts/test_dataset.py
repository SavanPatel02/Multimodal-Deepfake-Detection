import os
from torch.utils.data import DataLoader
from multimodal_dataset import MultimodalDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset = MultimodalDataset(
    csv_file=os.path.join(BASE_DIR, "Data", "processed", "train_subset.csv"),
    frames_root=os.path.join(BASE_DIR, "Data", "processed", "frames", "train"),
    mel_root=os.path.join(BASE_DIR, "Data", "processed", "mels", "train")
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

frames, mel, label = next(iter(loader))

print("Frames shape:", frames.shape)
print("Mel shape:", mel.shape)
print("Label shape:", label.shape)