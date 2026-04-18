"""
PAIR 4 — Fusion Strategy
Research Question: Does smarter fusion (cross-attention) beat simple concatenation?

M7: ResNet18 + Custom CNN + Concatenation  (simple fusion, same as M1)
M8: ResNet18 + Custom CNN + Cross-Attention (each modality attends to the other)

Key idea in M8:
  - Visual features "ask" the audio: "which audio patterns match what I see?"
  - Audio features "ask" the visual: "which face regions match what I hear?"
  - This cross-modal attention produces richer joint representations
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ==========================
# SHARED BACKBONES
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
        return self.feature_extractor(x).view(B, T, -1).max(dim=1)[0]


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
# M7 — Concatenation Fusion (baseline)
# Simply stacks visual and audio features, lets the classifier figure it out
# ==========================

class M7_ConcatFusion(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.visual = VisualBackbone()
        self.audio = AudioBackbone()

        fusion_dim = self.visual.output_dim + self.audio.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, frames, mel):
        v = self.visual(frames)
        a = self.audio(mel)
        return self.classifier(torch.cat([v, a], dim=1))


# ==========================
# CROSS-ATTENTION FUSION MODULE
# Used by M8
#
# How it works:
#   1. Project both modalities to a common dimension (attn_dim)
#   2. Visual attends to Audio: visual features query the audio features
#   3. Audio attends to Visual: audio features query the visual features
#   4. Concatenate attended outputs → classify
# ==========================

class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim=512, audio_dim=128, attn_dim=256, num_heads=4):
        super().__init__()

        # Project both modalities to the same dimension
        self.visual_proj = nn.Linear(visual_dim, attn_dim)
        self.audio_proj = nn.Linear(audio_dim, attn_dim)

        # PyTorch MultiheadAttention: (query, key, value)
        self.v_attends_a = nn.MultiheadAttention(attn_dim, num_heads, batch_first=True)
        self.a_attends_v = nn.MultiheadAttention(attn_dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(attn_dim)
        self.norm2 = nn.LayerNorm(attn_dim)

        # Final classifier on attended features
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes := 2)
        )

    def forward(self, visual_feat, audio_feat):
        # Add sequence dim for MultiheadAttention: (B, 1, attn_dim)
        v = self.visual_proj(visual_feat).unsqueeze(1)   # (B, 1, attn_dim)
        a = self.audio_proj(audio_feat).unsqueeze(1)     # (B, 1, attn_dim)

        # Visual queries audio: "what audio pattern is consistent with this face?"
        v_ctx, _ = self.v_attends_a(query=v, key=a, value=a)
        v_ctx = self.norm1((v + v_ctx).squeeze(1))       # residual + norm (B, attn_dim)

        # Audio queries visual: "which face region matches what I hear?"
        a_ctx, _ = self.a_attends_v(query=a, key=v, value=v)
        a_ctx = self.norm2((a + a_ctx).squeeze(1))       # residual + norm (B, attn_dim)

        fused = torch.cat([v_ctx, a_ctx], dim=1)         # (B, attn_dim*2)
        return self.classifier(fused)


# ==========================
# M8 — Cross-Attention Fusion
# Each modality actively attends to the other before classification
# ==========================

class M8_AttentionFusion(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.visual = VisualBackbone()
        self.audio = AudioBackbone()

        self.fusion = CrossAttentionFusion(
            visual_dim=self.visual.output_dim,
            audio_dim=self.audio.output_dim,
            attn_dim=256,
            num_heads=4
        )

    def forward(self, frames, mel):
        v = self.visual(frames)     # (B, 512)
        a = self.audio(mel)         # (B, 128)
        return self.fusion(v, a)
