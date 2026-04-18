import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultimodalDataset(Dataset):
    def __init__(self, csv_file, frames_root, mel_root, split="train"):
        self.df = pd.read_csv(csv_file)
        self.frames_root = frames_root
        self.mel_root = mel_root
        self.split = split

        # ==========================
        # TRANSFORMS
        # ==========================

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),

                # 🔥 AUGMENTATIONS (VERY IMPORTANT)
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
                transforms.RandomRotation(10),

                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # NO augmentation for validation/test
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.df)

    # ==========================
    # LOAD VISUAL FRAMES
    # ==========================
    def load_frames(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_folder = os.path.join(self.frames_root, video_name)

        frames = []

        # 🔥 UPDATED → 16 FRAMES
        for i in range(1, 17):
            img_path = os.path.join(video_folder, f"frame_{i}.jpg")

            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
                frames.append(image)
            else:
                # fallback
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    frames.append(torch.zeros(3, 224, 224))

        frames = torch.stack(frames)  # (16, 3, 224, 224)
        return frames

    # ==========================
    # LOAD MEL SPECTROGRAM
    # ==========================
    def load_mel(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        mel_path = os.path.join(self.mel_root, video_name + ".npy")

        mel = np.load(mel_path)

        # 🔥 AUDIO AUGMENTATION
        if self.split == "train":
            # noise injection
            noise = np.random.normal(0, 0.01, mel.shape)
            mel = mel + noise

            # random gain
            mel = mel * np.random.uniform(0.8, 1.2)

        mel = torch.tensor(mel).float()
        mel = mel.unsqueeze(0)  # (1, 128, T)

        return mel

    # ==========================
    # GET ITEM
    # ==========================
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_path = row["video_path"]
        label = torch.tensor(row["label"]).long()

        frames = self.load_frames(video_path)
        mel = self.load_mel(video_path)

        return frames, mel, label