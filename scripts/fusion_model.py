import torch
import torch.nn as nn
import torchvision.models as models


class MultimodalFusionModel(nn.Module):
    def __init__(self):
        super(MultimodalFusionModel, self).__init__()

        # =============================
        # Visual Backbone (ResNet18)
        # =============================
        self.visual_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.visual_model.fc = nn.Identity()  # remove classifier
        visual_feature_dim = 512

        # =============================
        # Audio CNN
        # =============================
        self.audio_model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        audio_feature_dim = 128

        # =============================
        # Fusion Layer
        # =============================
        self.classifier = nn.Sequential(
            nn.Linear(visual_feature_dim + audio_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, frames, mel):

        # frames: (B, 8, 3, 224, 224)
        B, T, C, H, W = frames.shape

        frames = frames.view(B * T, C, H, W)
        visual_features = self.visual_model(frames)
        visual_features = visual_features.view(B, T, -1)
        visual_features = visual_features.mean(dim=1)  # average over frames

        # mel: (B, 1, 128, T)
        audio_features = self.audio_model(mel)
        audio_features = audio_features.view(B, -1)

        fused = torch.cat((visual_features, audio_features), dim=1)

        output = self.classifier(fused)

        return output