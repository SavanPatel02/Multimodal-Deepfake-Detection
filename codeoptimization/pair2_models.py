"""
PAIR 2 — Audio Processing Strategy
Research Question: Is frequency-based (CNN) or temporal-sequence (LSTM) better for fake audio?

M3: ResNet18 + CNN audio   (treats mel as a 2D image)
M4: ResNet18 + LSTM audio  (treats mel as a time sequence)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ==========================
# SHARED VISUAL BACKBONE
# ResNet18 used by both M3 and M4
# ==========================

class VisualBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.feature_extractor = resnet
        self.output_dim = 512

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x).view(B, T, -1)
        return features.max(dim=1)[0]  # (B, 512)


# ==========================
# M3 — ResNet18 + CNN Audio
# Treats mel spectrogram as a 2D image (frequency x time)
# Same as M1 — included here as the direct comparison baseline
# ==========================

class M3_ResNet18_CNNAudio(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.visual = VisualBackbone()

        self.audio = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.audio_dim = 128

        fusion_dim = self.visual.output_dim + self.audio_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, frames, mel):
        v = self.visual(frames)                          # (B, 512)
        a = self.audio(mel).view(mel.size(0), -1)        # (B, 128)
        return self.classifier(torch.cat([v, a], dim=1))


# ==========================
# M4 — ResNet18 + LSTM Audio
# Treats mel spectrogram as a time sequence (T steps, 128 mel bins each)
# Bidirectional LSTM captures both past and future context in the audio
# ==========================

class LSTMAudioBackbone(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True      # forward + backward pass over time
        )
        # bidirectional doubles the output: hidden_size * 2
        self.output_dim = hidden_size * 2

        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, 1, 128, T)
        x = x.squeeze(1)                # (B, 128, T)
        x = x.permute(0, 2, 1)         # (B, T, 128) — sequence of T mel frames
        _, (hidden, _) = self.lstm(x)

        # hidden: (num_layers*2, B, hidden_size) for bidirectional
        # take the last layer's forward and backward hidden states
        fwd = hidden[-2]               # (B, hidden_size)
        bwd = hidden[-1]               # (B, hidden_size)
        out = torch.cat([fwd, bwd], dim=1)  # (B, hidden_size*2)
        return self.proj(out)          # (B, 128)


class M4_ResNet18_LSTMAudio(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.visual = VisualBackbone()
        self.audio = LSTMAudioBackbone()

        fusion_dim = self.visual.output_dim + 128   # proj output is 128

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, frames, mel):
        v = self.visual(frames)     # (B, 512)
        a = self.audio(mel)         # (B, 128)
        return self.classifier(torch.cat([v, a], dim=1))
