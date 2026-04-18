import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

# Fixed mel time dimension matching generate_mels.py settings:
# SR=16000, DURATION=4s, HOP_LENGTH=512 → ceil(64000/512) + 1 = 126
MEL_TIME_STEPS = 126

# ImageNet stats for normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class MultimodalDataset(Dataset):
    def __init__(self, csv_file, frames_root, mel_root, split="train", modality="both"):
        df = pd.read_csv(csv_file)
        self.frames_root = frames_root
        self.mel_root = mel_root
        self.split = split
        self.modality = modality  # "both", "video", "audio"

        # Scan the frames directory ONCE → build sets for O(1) lookup.
        dir_contents = os.listdir(frames_root) if os.path.isdir(frames_root) else []
        npy_set    = {f[:-4] for f in dir_contents if f.endswith(".npy")}
        folder_set = {f     for f in dir_contents if not f.endswith(".npy")}

        names = df["video_path"].apply(
            lambda p: os.path.splitext(os.path.basename(p))[0]
        )
        has_npy    = names.isin(npy_set)
        has_folder = names.isin(folder_set)
        has_frames = has_npy | has_folder

        npy_count    = has_npy.sum()
        folder_count = (has_folder & ~has_npy).sum()
        dropped      = (~has_frames).sum()

        if npy_count and folder_count:
            print(f"[Dataset/{split}] Mixed formats: {npy_count} .npy  |  {folder_count} JPEG folders")
        elif npy_count:
            print(f"[Dataset/{split}] Format: .npy  ({npy_count} samples)")
        else:
            print(f"[Dataset/{split}] Format: JPEG folders  ({folder_count} samples)")

        if dropped:
            print(f"[Dataset/{split}] Skipping {dropped} samples with no face frames "
                  f"({dropped / len(df) * 100:.2f}% of {len(df)})")

        self.df       = df[has_frames].reset_index(drop=True)
        self._has_npy = has_npy[has_frames].reset_index(drop=True)

        # PIL-based transform only needed for JPEG folder fallback
        if split == "train":
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    # ==========================
    # LOAD VISUAL FRAMES — FAST PATH
    # .npy: pure numpy/torch, no PIL conversion (5-10x faster)
    # JPEG folders: fallback to PIL pipeline
    # ==========================
    def load_frames(self, idx, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if self._has_npy.iloc[idx]:
            # ── FAST PATH: numpy → torch directly ─────────────────────────
            npy_path  = os.path.join(self.frames_root, video_name + ".npy")
            frames_np = np.load(npy_path)   # (16, 224, 224, 3) uint8

            # Pad to 16 if short
            if len(frames_np) < 16:
                pad = np.repeat(frames_np[-1:], 16 - len(frames_np), axis=0)
                frames_np = np.concatenate([frames_np, pad], axis=0)
            frames_np = frames_np[:16]

            # (16, 224, 224, 3) uint8 → (16, 3, 224, 224) float32 normalized
            frames_np = frames_np.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

            # Color jitter approximation for training (numpy, no PIL)
            if self.split == "train":
                brightness = np.random.uniform(0.8, 1.2)
                contrast   = np.random.uniform(0.8, 1.2)
                saturation = np.random.uniform(0.8, 1.2)
                frames_np = frames_np * brightness
                mean = frames_np.mean(axis=(2, 3), keepdims=True)
                frames_np = (frames_np - mean) * contrast + mean
                # Saturation: blend with grayscale
                gray = 0.299 * frames_np[:, 0:1] + 0.587 * frames_np[:, 1:2] + 0.114 * frames_np[:, 2:3]
                frames_np = gray + (frames_np - gray) * saturation
                frames_np = np.clip(frames_np, 0.0, 1.0)

            # ImageNet normalize
            frames_np = (frames_np - IMAGENET_MEAN) / IMAGENET_STD

            return torch.from_numpy(frames_np.copy())  # (16, 3, 224, 224)

        else:
            # ── FALLBACK: JPEG folders via PIL ─────────────────────────────
            video_folder = os.path.join(self.frames_root, video_name)
            frames = []
            for i in range(1, 17):
                img_path = os.path.join(video_folder, f"frame_{i}.jpg")
                if os.path.exists(img_path):
                    frames.append(self.transform(Image.open(img_path).convert("RGB")))
                else:
                    frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))

            while len(frames) < 16:
                frames.append(frames[-1])

            return torch.stack(frames[:16])

    # ==========================
    # LOAD MEL SPECTROGRAM
    # ==========================
    def load_mel(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        mel_path = os.path.join(self.mel_root, video_name + ".npy")

        if not os.path.exists(mel_path):
            return torch.zeros(1, 128, MEL_TIME_STEPS)

        mel = np.load(mel_path)

        if mel.shape[1] < MEL_TIME_STEPS:
            pad_width = MEL_TIME_STEPS - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mel = mel[:, :MEL_TIME_STEPS]

        if self.split == "train":
            noise = np.random.normal(0, 0.01, mel.shape)
            mel = mel + noise
            mel = mel * np.random.uniform(0.8, 1.2)
            mel = np.clip(mel, 0.0, 1.0)

        mel = torch.tensor(mel, dtype=torch.float32)
        mel = mel.unsqueeze(0)  # (1, 128, MEL_TIME_STEPS)

        return mel

    # ==========================
    # GET ITEM
    # ==========================
    # Dummy tensors for unused modalities are tiny (1-element) to avoid
    # wasting memory/bandwidth. The model ignores them anyway.
    _DUMMY_FRAMES = torch.zeros(16, 3, 1, 1)   # 192 bytes vs 9.6 MB
    _DUMMY_MEL    = torch.zeros(1, 1, 1)        # 4 bytes vs 64 KB

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_path = row["video_path"]
        label = torch.tensor(row["label"]).long()

        if self.modality == "audio":
            frames = self._DUMMY_FRAMES
            mel = self.load_mel(video_path)
        elif self.modality == "video":
            frames = self.load_frames(idx, video_path)
            mel = self._DUMMY_MEL
        else:
            frames = self.load_frames(idx, video_path)
            mel = self.load_mel(video_path)

        return frames, mel, label
