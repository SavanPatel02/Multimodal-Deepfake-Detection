"""
PAIR 3 — Efficiency
Research Question: Can a lightweight model match heavier ones for real-world deployment?

M5: MobileNetV3-Small + Lightweight CNN  (fastest, smallest)
M6: EfficientNet-B0   + Custom CNN       (efficient but stronger)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ==========================
# SHARED FULL AUDIO BACKBONE
# Used by M6
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
        return self.cnn(x).view(x.size(0), -1)


# ==========================
# LIGHTWEIGHT AUDIO CNN
# Used by M5 — 2 layers instead of 3, fewer filters
# ==========================

class LightweightAudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.output_dim = 32

    def forward(self, x):
        return self.cnn(x).view(x.size(0), -1)


# ==========================
# M5 — MobileNetV3-Small + Lightweight CNN
# Designed for speed — smallest model in the study
# MobileNetV3-Small: ~2.5M params, optimized for mobile/edge
# ==========================

class M5_MobileNetV3_LightCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        # MobileNetV3: features → avgpool → flatten → 576-dim
        self.visual_features = mobilenet.features
        self.visual_avgpool = mobilenet.avgpool
        self.visual_dim = 576

        self.audio = LightweightAudioCNN()

        fusion_dim = self.visual_dim + self.audio.output_dim  # 576 + 32

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, frames, mel):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)

        # Extract visual features frame by frame
        v = self.visual_features(frames)        # (B*T, 576, h, w)
        v = self.visual_avgpool(v)              # (B*T, 576, 1, 1)
        v = v.view(B, T, -1)                   # (B, T, 576)
        v = v.max(dim=1)[0]                    # (B, 576)

        a = self.audio(mel)                     # (B, 32)
        return self.classifier(torch.cat([v, a], dim=1))


# ==========================
# M6 — EfficientNet-B0 + Custom CNN
# Efficient but stronger — compound scaling of depth/width/resolution
# EfficientNet-B0: ~5.3M params, best accuracy-per-FLOP in its class
# ==========================

class M6_EfficientNetB0_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        # EfficientNet: features (ends with 1280-ch conv) → avgpool → classifier
        self.visual_features = efficientnet.features
        self.visual_avgpool = nn.AdaptiveAvgPool2d(1)
        self.visual_dim = 1280

        self.audio = AudioBackbone()

        fusion_dim = self.visual_dim + self.audio.output_dim  # 1280 + 128

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
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

        v = self.visual_features(frames)        # (B*T, 1280, h, w)
        v = self.visual_avgpool(v)              # (B*T, 1280, 1, 1)
        v = v.view(B, T, -1)                   # (B, T, 1280)
        v = v.max(dim=1)[0]                    # (B, 1280)

        a = self.audio(mel)                     # (B, 128)
        return self.classifier(torch.cat([v, a], dim=1))
