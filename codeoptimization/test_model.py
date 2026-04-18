import torch
from multimodal_model import MultimodalDeepfakeModel

model = MultimodalDeepfakeModel()
model = model.cuda()

# FIX: use 16 frames to match what MultimodalDataset actually produces (was 8)
frames = torch.randn(2, 16, 3, 224, 224).cuda()

# mel time dim = 126 (SR=16000, DURATION=4s, HOP_LENGTH=512)
mel = torch.randn(2, 1, 128, 126).cuda()

output = model(frames, mel)

print("Output shape:", output.shape)  # expected: (2, 2)
