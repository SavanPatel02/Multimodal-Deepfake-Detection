import torch
from multimodal_model import MultimodalDeepfakeModel

model = MultimodalDeepfakeModel()
model = model.cuda()

frames = torch.randn(2, 8, 3, 224, 224).cuda()
mel = torch.randn(2, 1, 128, 126).cuda()

output = model(frames, mel)

print("Output shape:", output.shape)