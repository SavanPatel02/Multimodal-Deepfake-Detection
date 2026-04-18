import os
import sys
import cv2
import torch
import librosa
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from moviepy import VideoFileClip

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger_setup import setup_logger
from multimodal_model import MultimodalDeepfakeModel

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "multimodal_model.pth")

# FIX: accept video path as CLI argument instead of hardcoded path
if len(sys.argv) < 2:
    print("Usage: python predict_video.py <path_to_video>")
    sys.exit(1)

VIDEO_PATH = sys.argv[1]
log = setup_logger(f"predict_{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}")
log.info(f"Video path: {VIDEO_PATH}")

FRAMES_PER_VIDEO = 16
IMG_SIZE = 224

SR = 16000
MAX_AUDIO_LEN = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

# ==============================
# LOAD MODEL
# FIX: weights_only=True for security
# ==============================

model = MultimodalDeepfakeModel()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
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
# FIX: use try/finally to ensure temp file is always cleaned up
# ==============================

def extract_mel(video_path):

    audio_path = "temp_audio.wav"

    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, logger=None)
        clip.close()

        y, sr = librosa.load(audio_path, sr=SR)

        # Step 1: amplitude normalize FIRST — must match generate_mels.py order
        y = y / (np.max(np.abs(y)) + 1e-6)

        # Step 2: trim or pad to exactly 4 seconds
        max_len = SR * MAX_AUDIO_LEN
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        else:
            y = y[:max_len]

        # Step 3: mel spectrogram — all params must match generate_mels.py exactly
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )

        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)

        # Step 4: enforce fixed time dimension to match what model was trained on
        # SR=16000, 4s, hop=512 → 126 time steps
        MEL_TIME_STEPS = 126
        if mel.shape[1] < MEL_TIME_STEPS:
            mel = np.pad(mel, ((0, 0), (0, MEL_TIME_STEPS - mel.shape[1])))
        else:
            mel = mel[:, :MEL_TIME_STEPS]

        mel = torch.tensor(mel).float().unsqueeze(0).unsqueeze(0).to(device)

        return mel

    finally:
        # FIX: always remove temp file, even if an exception occurred
        if os.path.exists(audio_path):
            os.remove(audio_path)


# ==============================
# RUN EXTRACTION
# ==============================

frames = extract_frames(VIDEO_PATH)
mel = extract_mel(VIDEO_PATH)

log.info(f"Frames mean: {round(frames.mean().item(), 4)}")
log.info(f"Mel mean:    {round(mel.mean().item(), 4)}")

# ==============================
# MODEL INFERENCE
# ==============================

with torch.no_grad():

    # Full fusion
    fusion_logits = model(frames, mel)
    fusion_probs = F.softmax(fusion_logits, dim=1)

    # Visual only
    zero_mel = torch.zeros_like(mel)
    visual_logits = model(frames, zero_mel)
    visual_probs = F.softmax(visual_logits, dim=1)

    # Audio only
    zero_frames = torch.zeros_like(frames)
    audio_logits = model(zero_frames, mel)
    audio_probs = F.softmax(audio_logits, dim=1)


# ==============================
# PRINT RESULTS
# ==============================

def print_result(name, probs):
    confidence, prediction = torch.max(probs, dim=1)
    label = "Fake" if prediction.item() == 1 else "Real"
    log.info(f"\n===== {name} =====")
    log.info(f"Prediction : {label}")
    log.info(f"Confidence : {round(confidence.item() * 100, 2)} %")


print_result("FACE ONLY", visual_probs)
print_result("AUDIO ONLY", audio_probs)
print_result("FUSION", fusion_probs)
