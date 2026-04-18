"""
PAIR 5 — Unimodal vs Multimodal
Research Question: Is multimodal actually better than using just one signal?

M9:  ResNet18 only  — visual/face signal alone, ignores audio
M10: Custom CNN only — audio signal alone, ignores video frames

These are the control models. If M1–M8 (multimodal) don't clearly beat
M9 and M10, it means the multimodal fusion isn't adding value.

Expected outcome: Neither unimodal model should consistently beat
the full multimodal models, proving fusion is worthwhile.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ==========================
# M9 — Visual Only (ResNet18)
# Detects deepfakes using face frames alone — no audio
# Still accepts mel as input but ignores it (keeps training loop uniform)
# ==========================

class M9_VisualOnly(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.visual = resnet
        self.visual_dim = 512

        self.classifier = nn.Sequential(
            nn.Linear(self.visual_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, frames, mel):
        # mel is accepted but intentionally ignored
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        v = self.visual(frames).view(B, T, -1).max(dim=1)[0]  # (B, 512)
        return self.classifier(v)


# ==========================
# M10 — Audio Only (Custom CNN)
# Detects deepfakes using mel spectrogram alone — no video
# Still accepts frames as input but ignores them (keeps training loop uniform)
# ==========================

class M10_AudioOnly(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

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

        self.classifier = nn.Sequential(
            nn.Linear(self.audio_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, frames, mel):
        # frames is accepted but intentionally ignored
        a = self.audio(mel).view(mel.size(0), -1)  # (B, 128)
        return self.classifier(a)
