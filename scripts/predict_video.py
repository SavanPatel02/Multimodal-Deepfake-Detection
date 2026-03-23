import os
import cv2
import torch
import librosa
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from multimodal_model import MultimodalDeepfakeModel
from moviepy import VideoFileClip

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "multimodal_model.pth")

VIDEO_PATH = r"C:\Users\savan\Downloads\1.mp4"
print("Video path:", VIDEO_PATH)

FRAMES_PER_VIDEO = 16
IMG_SIZE = 224

SR = 16000
MAX_AUDIO_LEN = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# LOAD MODEL
# ==============================

model = MultimodalDeepfakeModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ==============================
# IMAGE TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

detector = MTCNN(keep_all=False, device=device)

# ==============================
# FRAME EXTRACTION
# ==============================

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = [
        int(i * total_frames / FRAMES_PER_VIDEO)
        for i in range(FRAMES_PER_VIDEO)
    ]

    frames = []
    current = 0
    saved = 0

    while cap.isOpened() and saved < FRAMES_PER_VIDEO:

        ret, frame = cap.read()
        if not ret:
            break

        if current in frame_indices:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = detector.detect(rgb)

            if boxes is not None:

                x1, y1, x2, y2 = boxes[0]
                x1, y1 = int(max(0, x1)), int(max(0, y1))
                x2, y2 = int(x2), int(y2)

                face = rgb[y1:y2, x1:x2]

                if face.size == 0:
                    current += 1
                    continue

                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                face = transform(Image.fromarray(face))

                frames.append(face)
                saved += 1

        current += 1

    cap.release()

    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

    frames = torch.stack(frames).unsqueeze(0).to(device)

    return frames


# ==============================
# MEL SPECTROGRAM EXTRACTION
# ==============================

def extract_mel(video_path):

    clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"

    clip.audio.write_audiofile(audio_path, logger=None)

    y, sr = librosa.load(audio_path, sr=SR)

    max_len = SR * MAX_AUDIO_LEN

    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=128
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    # same normalization used during training
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)

    mel = torch.tensor(mel).float().unsqueeze(0).unsqueeze(0).to(device)

    os.remove(audio_path)

    return mel


# ==============================
# RUN EXTRACTION
# ==============================

frames = extract_frames(VIDEO_PATH)
mel = extract_mel(VIDEO_PATH)

print("Frames mean:", round(frames.mean().item(), 4))
print("Mel mean:", round(mel.mean().item(), 4))

# ==============================
# MODEL INFERENCE
# ==============================

with torch.no_grad():

    # full fusion
    fusion_logits = model(frames, mel)
    fusion_probs = F.softmax(fusion_logits, dim=1)

    # visual only
    zero_mel = torch.zeros_like(mel)
    visual_logits = model(frames, zero_mel)
    visual_probs = F.softmax(visual_logits, dim=1)

    # audio only
    zero_frames = torch.zeros_like(frames)
    audio_logits = model(zero_frames, mel)
    audio_probs = F.softmax(audio_logits, dim=1)


# ==============================
# PRINT RESULTS
# ==============================

def print_result(name, probs):

    confidence, prediction = torch.max(probs, dim=1)

    label = "Fake" if prediction.item() == 1 else "Real"

    print(f"\n===== {name} =====")
    print("Prediction :", label)
    print("Confidence :", round(confidence.item()*100, 2), "%")


print_result("FACE ONLY", visual_probs)
print_result("AUDIO ONLY", audio_probs)
print_result("FUSION", fusion_probs)