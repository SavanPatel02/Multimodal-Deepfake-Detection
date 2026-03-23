import torch
import torch.nn as nn
import torchvision.models as models


# ==========================
# VISUAL BACKBONE
# ==========================

class VisualBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 512

    def forward(self, x):
        # x shape: (B, 8, 3, 224, 224)
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(B, T, -1)

        # Temporal average pooling
        features = features.max(dim=1)[0]  # (B, 512)

        return features


# ==========================
# AUDIO BACKBONE
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
        # x shape: (B, 1, 128, T)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # (B, 128)
        return x


# ==========================
# MULTIMODAL FUSION MODEL
# ==========================

class MultimodalDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.visual_backbone = VisualBackbone()
        self.audio_backbone = AudioBackbone()

        fusion_dim = (
            self.visual_backbone.output_dim +
            self.audio_backbone.output_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )   

    def forward(self, frames, mel):

        visual_features = self.visual_backbone(frames)
        audio_features = self.audio_backbone(mel)

        fused = torch.cat((visual_features, audio_features), dim=1)

        output = self.classifier(fused)

        return output