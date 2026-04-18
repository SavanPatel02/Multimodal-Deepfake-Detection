"""
PAIR 1 — Backbone Depth (Visual)
Research Question: Does a deeper visual backbone improve deepfake detection?

M1: ResNet18 + Custom CNN  (lightweight baseline)
M2: ResNet50 + Custom CNN  (deeper visual backbone)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ==========================
# SHARED AUDIO BACKBONE
# Used by both M1 and M2
# ==========================

class AudioBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
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
        self.output_dim = 128

    def forward(self, x):
        x = self.cnn(x)
        return x.view(x.size(0), -1)  # (B, 128)


# ==========================
# M1 — ResNet18 + Custom CNN
# Baseline model (lightweight)
# ==========================

class M1_ResNet18_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.visual = resnet
        self.visual_dim = 512

        self.audio = AudioBackbone()

        fusion_dim = self.visual_dim + self.audio.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, frames, mel):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        v = self.visual(frames).view(B, T, -1).max(dim=1)[0]   # (B, 512)
        a = self.audio(mel)                                      # (B, 128)
        return self.classifier(torch.cat([v, a], dim=1))


# ==========================
# M2 — ResNet50 + Custom CNN
# Deeper visual backbone (2048-dim vs 512-dim)
# ==========================

class M2_ResNet50_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.visual = resnet
        self.visual_dim = 2048              # ResNet50 produces 2048-dim features

        self.audio = AudioBackbone()

        fusion_dim = self.visual_dim + self.audio.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),     # wider first layer to handle 2048+128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, frames, mel):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        v = self.visual(frames).view(B, T, -1).max(dim=1)[0]   # (B, 2048)
        a = self.audio(mel)                                      # (B, 128)
        return self.classifier(torch.cat([v, a], dim=1))
