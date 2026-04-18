"""
Streamlit Dashboard — Multimodal Deepfake Detection
10-Model Comparison Study on LAV-DF Dataset

Run:
    streamlit run codeoptimization/dashboard.py
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F_torch
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==========================
# PATHS
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data", "Processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "subset")

# ==========================
# MODEL METADATA
# ==========================

MODEL_INFO = {
    "M1":  {"pair": "Pair 1", "visual": "ResNet18",       "audio": "Custom CNN",       "fusion": "Concat",          "question": "Does deeper visual backbone help?"},
    "M2":  {"pair": "Pair 1", "visual": "ResNet50",        "audio": "Custom CNN",       "fusion": "Concat",          "question": "Does deeper visual backbone help?"},
    "M3":  {"pair": "Pair 2", "visual": "ResNet18",        "audio": "CNN Audio",        "fusion": "Concat",          "question": "CNN vs LSTM for audio processing?"},
    "M4":  {"pair": "Pair 2", "visual": "ResNet18",        "audio": "LSTM Audio",       "fusion": "Concat",          "question": "CNN vs LSTM for audio processing?"},
    "M5":  {"pair": "Pair 3", "visual": "MobileNetV3-S",   "audio": "Lightweight CNN",  "fusion": "Concat",          "question": "Can lightweight models compete?"},
    "M6":  {"pair": "Pair 3", "visual": "EfficientNet-B0", "audio": "Custom CNN",       "fusion": "Concat",          "question": "Can lightweight models compete?"},
    "M7":  {"pair": "Pair 4", "visual": "ResNet18",        "audio": "Custom CNN",       "fusion": "Concat",          "question": "Cross-attention vs concatenation?"},
    "M8":  {"pair": "Pair 4", "visual": "ResNet18",        "audio": "Custom CNN",       "fusion": "Cross-Attention", "question": "Cross-attention vs concatenation?"},
    "M9":  {"pair": "Pair 5", "visual": "ResNet18",        "audio": "-- (Visual Only)", "fusion": "--",              "question": "Is multimodal better than unimodal?"},
    "M10": {"pair": "Pair 5", "visual": "-- (Audio Only)",  "audio": "Custom CNN",       "fusion": "--",              "question": "Is multimodal better than unimodal?"},
}

PAIR_COLORS = {
    "Pair 1": "#636EFA",
    "Pair 2": "#EF553B",
    "Pair 3": "#00CC96",
    "Pair 4": "#AB63FA",
    "Pair 5": "#FFA15A",
}

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
MEL_TIME_STEPS = 126


# ==========================
# DATA LOADING HELPERS
# ==========================

def load_all_results():
    """Load all M1-M10 JSON result files."""
    results = {}
    for m in MODEL_INFO:
        path = os.path.join(RESULTS_DIR, f"{m}.json")
        if os.path.exists(path):
            with open(path) as f:
                results[m] = json.load(f)
    return results


def load_frames_npy(npy_path):
    """Load preprocessed face frames from .npy."""
    frames = np.load(npy_path)
    if len(frames) < 16:
        pad = np.repeat(frames[-1:], 16 - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    frames = frames[:16]
    frames_norm = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    frames_norm = (frames_norm - IMAGENET_MEAN) / IMAGENET_STD
    return frames, torch.from_numpy(frames_norm.copy()).unsqueeze(0)  # raw + tensor


def load_frames_from_dir(frame_dir):
    """Load face frames from a directory of JPG images (frame_1.jpg, frame_2.jpg, ...)."""
    from PIL import Image
    frame_files = sorted(
        [f for f in os.listdir(frame_dir) if f.lower().endswith('.jpg')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    frames = []
    for fname in frame_files[:FRAMES_PER_VIDEO]:
        img = Image.open(os.path.join(frame_dir, fname)).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        frames.append(np.array(img))
    frames = np.array(frames, dtype=np.uint8)  # (N, H, W, 3)
    if len(frames) < FRAMES_PER_VIDEO:
        pad = np.repeat(frames[-1:], FRAMES_PER_VIDEO - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    frames_norm = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    frames_norm = (frames_norm - IMAGENET_MEAN) / IMAGENET_STD
    return frames, torch.from_numpy(frames_norm.copy()).unsqueeze(0)


def _load_frames(frames_dir, video_name):
    """Load frames from either a .npy file or a directory of JPGs."""
    npy_path = os.path.join(frames_dir, video_name + ".npy")
    dir_path = os.path.join(frames_dir, video_name)
    if os.path.exists(npy_path):
        return load_frames_npy(npy_path)
    elif os.path.isdir(dir_path):
        return load_frames_from_dir(dir_path)
    return None, None


def _has_frames(frames_dir, video_name):
    """Return True if frames exist as a .npy file or a JPG directory."""
    return (
        os.path.exists(os.path.join(frames_dir, video_name + ".npy"))
        or os.path.isdir(os.path.join(frames_dir, video_name))
    )


def load_mel_npy(npy_path):
    """Load preprocessed mel spectrogram from .npy."""
    mel = np.load(npy_path)
    if mel.shape[1] < MEL_TIME_STEPS:
        mel = np.pad(mel, ((0, 0), (0, MEL_TIME_STEPS - mel.shape[1])))
    else:
        mel = mel[:, :MEL_TIME_STEPS]
    return mel, torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # raw + tensor


@st.cache_resource
def load_model(model_name):
    """Load a trained model (cached)."""
    from train_pair import get_model
    checkpoint_path = os.path.join(RESULTS_DIR, f"{model_name}.pth")
    if not os.path.exists(checkpoint_path):
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ==========================
# VIDEO PROCESSING (for upload)
# ==========================

FRAMES_PER_VIDEO = 16
IMG_SIZE = 224
SR = 16000
MAX_AUDIO_LEN = 4


@st.cache_resource
def get_face_detector():
    """Load MTCNN face detector (cached)."""
    from facenet_pytorch import MTCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MTCNN(keep_all=False, device=device)


def extract_frames_from_video(video_path):
    """Extract 16 face-cropped frames from a video file using MTCNN.
    Returns (raw_frames_uint8, normalized_tensor)."""
    import cv2
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    detector = get_face_detector()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, None

    frame_indices = [int(i * total_frames / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)]

    raw_faces = []
    tensor_faces = []
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
                x2, y2 = int(min(rgb.shape[1], x2)), int(min(rgb.shape[0], y2))
                face = rgb[y1:y2, x1:x2]

                if face.size > 0:
                    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    raw_faces.append(face_resized)
                    tensor_faces.append(transform(Image.fromarray(face_resized)))
                    saved += 1

        current += 1

    cap.release()

    # Pad if fewer than 16 faces found
    while len(raw_faces) < FRAMES_PER_VIDEO:
        if raw_faces:
            raw_faces.append(raw_faces[-1])
            tensor_faces.append(tensor_faces[-1])
        else:
            raw_faces.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
            tensor_faces.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_array = np.array(raw_faces[:FRAMES_PER_VIDEO])
    frames_tensor = torch.stack(tensor_faces[:FRAMES_PER_VIDEO]).unsqueeze(0).to(device)
    return raw_array, frames_tensor


def extract_mel_from_video(video_path):
    """Extract mel spectrogram from a video's audio track.
    Returns (raw_mel_2d, mel_tensor)."""
    import librosa
    from moviepy import VideoFileClip

    audio_path = None
    try:
        # Write audio to a temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = tmp.name
        tmp.close()

        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, logger=None)
        clip.close()

        y, _ = librosa.load(audio_path, sr=SR)

        # Normalize amplitude
        y = y / (np.max(np.abs(y)) + 1e-6)

        # Pad/trim to 4 seconds
        max_len = SR * MAX_AUDIO_LEN
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        else:
            y = y[:max_len]

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=512, n_mels=128)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)

        # Fix time dimension
        if mel.shape[1] < MEL_TIME_STEPS:
            mel = np.pad(mel, ((0, 0), (0, MEL_TIME_STEPS - mel.shape[1])))
        else:
            mel = mel[:, :MEL_TIME_STEPS]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        return mel, mel_tensor

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Deepfake Detection Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================
# SIDEBAR NAVIGATION
# ==========================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Training Results", "Model Comparison", "Test on Videos", "Upload & Predict", "Per-Model Details"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Multimodal Deepfake Detection**")
st.sidebar.markdown("10-Model Comparison Study")
st.sidebar.markdown("Dataset: LAV-DF")

# Load results once
all_results = load_all_results()


# ================================================================
# PAGE 1: OVERVIEW
# ================================================================

if page == "Overview":
    st.title("Multimodal Deepfake Detection")
    st.markdown("### 10-Model Comparison Study on LAV-DF Dataset")

    st.markdown("---")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Trained", f"{len(all_results)}/10")
    with col2:
        best_model = max(all_results.items(), key=lambda x: x[1]["best_val_acc"])
        st.metric("Best Accuracy", f"{best_model[1]['best_val_acc']:.2f}%", delta=best_model[0])
    with col3:
        best_f1 = max(all_results.items(), key=lambda x: x[1]["best_val_f1"])
        st.metric("Best F1 Score", f"{best_f1[1]['best_val_f1']:.4f}", delta=best_f1[0])
    with col4:
        st.metric("Dataset", "LAV-DF", delta="136,304 videos")

    st.markdown("---")

    # Research pairs
    st.subheader("5 Research Pairs")

    pairs_data = {
        "Pair 1 — Backbone Depth": {
            "question": "Does a deeper visual backbone improve detection?",
            "models": "M1 (ResNet18) vs M2 (ResNet50)",
        },
        "Pair 2 — Audio Strategy": {
            "question": "CNN (2D spatial) vs LSTM (temporal sequence) for audio?",
            "models": "M3 (CNN Audio) vs M4 (LSTM Audio)",
        },
        "Pair 3 — Efficiency": {
            "question": "Can lightweight models match heavier ones?",
            "models": "M5 (MobileNetV3) vs M6 (EfficientNet-B0)",
        },
        "Pair 4 — Fusion Strategy": {
            "question": "Does cross-attention beat simple concatenation?",
            "models": "M7 (Concat) vs M8 (Cross-Attention)",
        },
        "Pair 5 — Unimodal vs Multimodal": {
            "question": "Is multimodal fusion actually better?",
            "models": "M9 (Visual Only) vs M10 (Audio Only)",
        },
    }

    for i, (pair_name, info) in enumerate(pairs_data.items()):
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown(f"**{pair_name}**")
            st.caption(info["models"])
        with col_b:
            st.markdown(f"*{info['question']}*")

    st.markdown("---")

    # Architecture table
    st.subheader("Model Architectures")
    arch_data = []
    for m, info in MODEL_INFO.items():
        r = all_results.get(m, {})
        arch_data.append({
            "Model": m,
            "Pair": info["pair"],
            "Visual Backbone": info["visual"],
            "Audio Backbone": info["audio"],
            "Fusion": info["fusion"],
            "Parameters": f"{r.get('parameters', 0):,}" if r else "—",
            "Val Acc (%)": f"{r['best_val_acc']:.2f}" if r else "—",
            "Val F1": f"{r['best_val_f1']:.4f}" if r else "—",
        })

    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    # Dataset info
    st.markdown("---")
    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Full Dataset**")
        st.markdown("- Train: 78,703 videos")
        st.markdown("- Dev: 31,501 videos")
        st.markdown("- Test: 26,100 videos")
    with col2:
        st.markdown("**Subset (used for training)**")
        st.markdown("- Train: 25,000 (6,751 real / 18,249 fake)")
        st.markdown("- Dev: 5,000 (1,313 real / 3,687 fake)")
        st.markdown("- Test: 2,000 (1,000 real / 1,000 fake)")
    with col3:
        st.markdown("**Input Format**")
        st.markdown("- Frames: 16 x 224x224 face crops")
        st.markdown("- Audio: 128x126 mel spectrogram")
        st.markdown("- Duration: 4 seconds per video")


# ================================================================
# PAGE 2: TRAINING RESULTS
# ================================================================

elif page == "Training Results":
    st.title("Training Results")

    # Overall ranking
    st.subheader("Model Rankings")

    ranking_data = []
    for m, r in sorted(all_results.items(), key=lambda x: x[1]["best_val_acc"], reverse=True):
        info = MODEL_INFO[m]
        ranking_data.append({
            "Model": m,
            "Pair": info["pair"],
            "Visual": info["visual"],
            "Audio": info["audio"],
            "Fusion": info["fusion"],
            "Val Acc (%)": r["best_val_acc"],
            "Val F1": r["best_val_f1"],
            "Epochs": r["epochs_run"],
            "Params": r["parameters"],
        })

    df_rank = pd.DataFrame(ranking_data)
    df_rank.index = range(1, len(df_rank) + 1)
    df_rank.index.name = "Rank"
    st.dataframe(df_rank, use_container_width=True)

    st.markdown("---")

    # Bar chart — Accuracy
    st.subheader("Validation Accuracy Comparison")

    models_sorted = sorted(all_results.keys(), key=lambda m: all_results[m]["best_val_acc"], reverse=True)
    accs = [all_results[m]["best_val_acc"] for m in models_sorted]
    colors = [PAIR_COLORS[MODEL_INFO[m]["pair"]] for m in models_sorted]

    fig_acc = go.Figure(data=[
        go.Bar(
            x=models_sorted,
            y=accs,
            marker_color=colors,
            text=[f"{a:.2f}%" for a in accs],
            textposition="outside",
        )
    ])
    fig_acc.update_layout(
        yaxis_title="Validation Accuracy (%)",
        yaxis_range=[0, 105],
        height=450,
        template="plotly_white",
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # Bar chart — F1
    st.subheader("Validation F1 Score Comparison")

    f1s = [all_results[m]["best_val_f1"] for m in models_sorted]

    fig_f1 = go.Figure(data=[
        go.Bar(
            x=models_sorted,
            y=f1s,
            marker_color=colors,
            text=[f"{f:.4f}" for f in f1s],
            textposition="outside",
        )
    ])
    fig_f1.update_layout(
        yaxis_title="Validation F1 Score",
        yaxis_range=[0, 1.1],
        height=450,
        template="plotly_white",
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    # Pair legend
    st.markdown("**Color Legend:**")
    legend_cols = st.columns(5)
    for i, (pair, color) in enumerate(PAIR_COLORS.items()):
        with legend_cols[i]:
            st.markdown(f'<span style="color:{color}; font-size:20px;">&#9632;</span> {pair}', unsafe_allow_html=True)


# ================================================================
# PAGE 3: MODEL COMPARISON (Training Curves)
# ================================================================

elif page == "Model Comparison":
    st.title("Model Comparison — Training Curves")

    # Model selector
    selected_models = st.multiselect(
        "Select models to compare",
        options=list(all_results.keys()),
        default=["M1", "M2", "M9", "M10"],
    )

    if not selected_models:
        st.warning("Please select at least one model.")
    else:
        # Metric selector
        metric = st.radio(
            "Metric",
            ["Validation Accuracy", "Validation F1", "Training Loss", "Validation Loss"],
            horizontal=True,
        )

        metric_map = {
            "Validation Accuracy": "val_acc",
            "Validation F1": "val_f1",
            "Training Loss": "train_loss",
            "Validation Loss": "val_loss",
        }
        key = metric_map[metric]

        fig = go.Figure()

        for m in selected_models:
            r = all_results[m]
            history = r["history"]
            epochs = [h["epoch"] for h in history]
            values = [h[key] for h in history]
            color = PAIR_COLORS[MODEL_INFO[m]["pair"]]

            fig.add_trace(go.Scatter(
                x=epochs,
                y=values,
                mode="lines+markers",
                name=f"{m} ({MODEL_INFO[m]['visual']})",
                line=dict(color=color, width=2),
                marker=dict(size=6),
            ))

        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title=metric,
            height=500,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Side-by-side training curves
        st.markdown("---")
        st.subheader("Train vs Validation Loss")

        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Training Loss", "Validation Loss"),
        )

        for m in selected_models:
            history = all_results[m]["history"]
            epochs = [h["epoch"] for h in history]
            color = PAIR_COLORS[MODEL_INFO[m]["pair"]]

            fig2.add_trace(
                go.Scatter(x=epochs, y=[h["train_loss"] for h in history],
                           mode="lines+markers", name=m, line=dict(color=color, width=2),
                           marker=dict(size=5), legendgroup=m, showlegend=True),
                row=1, col=1
            )
            fig2.add_trace(
                go.Scatter(x=epochs, y=[h["val_loss"] for h in history],
                           mode="lines+markers", name=m, line=dict(color=color, width=2, dash="dash"),
                           marker=dict(size=5), legendgroup=m, showlegend=False),
                row=1, col=2
            )

        fig2.update_layout(height=400, template="plotly_white")
        fig2.update_xaxes(title_text="Epoch")
        fig2.update_yaxes(title_text="Loss")
        st.plotly_chart(fig2, use_container_width=True)

        # Parameters comparison
        st.markdown("---")
        st.subheader("Model Size Comparison")

        params_data = []
        for m in selected_models:
            r = all_results[m]
            params_data.append({"Model": m, "Parameters": r["parameters"], "Val Acc (%)": r["best_val_acc"]})

        df_params = pd.DataFrame(params_data)

        fig3 = px.scatter(
            df_params,
            x="Parameters",
            y="Val Acc (%)",
            text="Model",
            size="Parameters",
            color="Model",
            size_max=40,
        )
        fig3.update_traces(textposition="top center")
        fig3.update_layout(
            xaxis_title="Number of Parameters",
            yaxis_title="Validation Accuracy (%)",
            height=450,
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)


# ================================================================
# PAGE 4: TEST ON VIDEOS
# ================================================================

elif page == "Test on Videos":
    st.title("Test Models on Videos")
    st.markdown("Select videos from the dataset and see how all 10 models predict.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Device: **{device}**")

    # Split selector
    split = st.selectbox("Dataset Split", ["test", "dev", "train"], index=0)

    # Load CSV
    csv_path = os.path.join(DATA_DIR, f"{split}_subset.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(DATA_DIR, f"{split}.csv")

    df = pd.read_csv(csv_path)
    df["name"] = df["video_path"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])

    frames_dir = os.path.join(DATA_DIR, "frames", split)
    mels_dir = os.path.join(DATA_DIR, "mels", split)

    df["has_frames"] = df["name"].apply(lambda n: _has_frames(frames_dir, n))
    df["has_mels"] = df["name"].apply(lambda n: os.path.exists(os.path.join(mels_dir, n + ".npy")))
    df["has_both"] = df["has_frames"] & df["has_mels"]

    available = df[df["has_both"]].reset_index(drop=True)
    real_videos = available[available["label"] == 0]["name"].tolist()
    fake_videos = available[available["label"] == 1]["name"].tolist()

    st.markdown(f"Available: **{len(real_videos)}** real, **{len(fake_videos)}** fake videos")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Select Real Videos**")
        selected_real = st.multiselect(
            "Real videos (label=0)",
            options=real_videos,
            default=real_videos[:3] if len(real_videos) >= 3 else real_videos,
        )
    with col2:
        st.markdown("**Select Fake Videos**")
        selected_fake = st.multiselect(
            "Fake videos (label=1)",
            options=fake_videos,
            default=fake_videos[:3] if len(fake_videos) >= 3 else fake_videos,
        )

    all_selected = [(v, 0) for v in selected_real] + [(v, 1) for v in selected_fake]

    # Model selector
    models_to_test = st.multiselect(
        "Models to test",
        options=list(MODEL_INFO.keys()),
        default=list(MODEL_INFO.keys()),
    )

    if st.button("Run Predictions", type="primary", use_container_width=True):
        if not all_selected:
            st.warning("Please select at least one video.")
        elif not models_to_test:
            st.warning("Please select at least one model.")
        else:
            # Load models
            progress = st.progress(0, text="Loading models...")
            loaded_models = {}
            for i, m in enumerate(models_to_test):
                model = load_model(m)
                if model is not None:
                    loaded_models[m] = model
                progress.progress((i + 1) / len(models_to_test), text=f"Loaded {m}")

            # Run predictions
            results_list = []
            total = len(all_selected)

            for idx, (video_name, true_label) in enumerate(all_selected):
                progress.progress((idx + 1) / total, text=f"Predicting {video_name}.mp4...")

                mel_path = os.path.join(mels_dir, video_name + ".npy")

                raw_frames, frames_tensor = _load_frames(frames_dir, video_name)
                raw_mel, mel_tensor = load_mel_npy(mel_path)

                frames_tensor = frames_tensor.to(device)
                mel_tensor = mel_tensor.to(device)

                row = {"Video": f"{video_name}.mp4", "Ground Truth": "REAL" if true_label == 0 else "FAKE"}

                for m_name, model in loaded_models.items():
                    with torch.no_grad():
                        logits = model(frames_tensor, mel_tensor)
                        probs = F_torch.softmax(logits, dim=1)
                        confidence, pred = torch.max(probs, dim=1)
                        pred_label = pred.item()
                        conf_val = confidence.item() * 100

                    pred_str = "REAL" if pred_label == 0 else "FAKE"
                    correct = pred_label == true_label
                    row[m_name] = f"{'✅' if correct else '❌'} {pred_str} ({conf_val:.1f}%)"

                results_list.append(row)

            progress.empty()

            # Results table
            st.markdown("---")
            st.subheader("Prediction Results")
            df_results = pd.DataFrame(results_list)
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Accuracy summary
            st.markdown("---")
            st.subheader("Model Accuracy on Selected Videos")

            acc_data = []
            for m in loaded_models:
                correct = sum(1 for r in results_list if r[m].startswith("✅"))
                total_v = len(results_list)
                acc_data.append({
                    "Model": m,
                    "Correct": correct,
                    "Total": total_v,
                    "Accuracy (%)": round(correct / total_v * 100, 1),
                    "Status": "PERFECT" if correct == total_v else f"{total_v - correct} wrong",
                })

            df_acc = pd.DataFrame(acc_data)

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_acc["Model"],
                        y=df_acc["Accuracy (%)"],
                        marker_color=[PAIR_COLORS[MODEL_INFO[m]["pair"]] for m in df_acc["Model"]],
                        text=[f"{a:.1f}%" for a in df_acc["Accuracy (%)"]],
                        textposition="outside",
                    )
                ])
                fig.update_layout(
                    yaxis_title="Accuracy (%)",
                    yaxis_range=[0, 110],
                    height=400,
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(df_acc[["Model", "Correct", "Accuracy (%)", "Status"]], hide_index=True)

            # Show sample frame + mel for the last processed video
            st.markdown("---")
            st.subheader("Sample Preprocessed Data")
            last_video = all_selected[-1][0]
            last_label = all_selected[-1][1]

            st.markdown(f"**Video:** {last_video}.mp4 ({'REAL' if last_label == 0 else 'FAKE'})")

            # Show frames
            raw_frames_show, _ = _load_frames(frames_dir, last_video)

            st.markdown("**Extracted Face Frames (16 keyframes):**")
            frame_cols = st.columns(8)
            for i in range(min(16, len(raw_frames_show))):
                with frame_cols[i % 8]:
                    st.image(raw_frames_show[i], caption=f"F{i+1}", width=100)

            # Show mel spectrogram
            mel_path = os.path.join(mels_dir, last_video + ".npy")
            raw_mel_show = np.load(mel_path)

            st.markdown("**Mel Spectrogram:**")
            fig_mel = px.imshow(
                raw_mel_show,
                aspect="auto",
                color_continuous_scale="magma",
                labels=dict(x="Time", y="Mel Bin", color="Amplitude"),
            )
            fig_mel.update_layout(height=300, template="plotly_white")
            st.plotly_chart(fig_mel, use_container_width=True)


# ================================================================
# PAGE 5: UPLOAD & PREDICT
# ================================================================

elif page == "Upload & Predict":
    st.title("Upload a Video & Predict")
    st.markdown("Upload any video file, select models, and get real-time deepfake predictions.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Device: **{device}**")

    # Upload
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv", "webm"])

    # Model selection
    models_to_use = st.multiselect(
        "Select models to run",
        options=list(MODEL_INFO.keys()),
        default=["M1", "M2", "M7", "M9"],
        key="upload_models",
    )

    if uploaded_file is not None:
        # Show video preview
        st.markdown("---")
        st.subheader("Uploaded Video")
        st.video(uploaded_file)

        if st.button("Analyze Video", type="primary", use_container_width=True):
            if not models_to_use:
                st.warning("Please select at least one model.")
            else:
                # Save uploaded file to temp location
                tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp_video.write(uploaded_file.read())
                tmp_video_path = tmp_video.name
                tmp_video.close()

                try:
                    # Step 1: Extract frames
                    st.markdown("---")
                    progress = st.progress(0, text="Extracting face frames (MTCNN)...")

                    raw_frames, frames_tensor = extract_frames_from_video(tmp_video_path)

                    if frames_tensor is None:
                        st.error("Could not read video frames. Please check the file.")
                    else:
                        # Show extracted faces
                        faces_found = sum(1 for f in raw_frames if f.sum() > 0)
                        progress.progress(40, text=f"Extracted {faces_found} face frames. Generating mel spectrogram...")

                        st.subheader("Extracted Face Frames")
                        frame_cols = st.columns(8)
                        for i in range(min(16, len(raw_frames))):
                            with frame_cols[i % 8]:
                                st.image(raw_frames[i], caption=f"F{i+1}", width=100)

                        # Step 2: Extract mel spectrogram
                        raw_mel, mel_tensor = extract_mel_from_video(tmp_video_path)

                        st.subheader("Mel Spectrogram")
                        fig_mel = px.imshow(
                            raw_mel,
                            aspect="auto",
                            color_continuous_scale="magma",
                            labels=dict(x="Time", y="Mel Bin", color="Amplitude"),
                        )
                        fig_mel.update_layout(height=250, template="plotly_white")
                        st.plotly_chart(fig_mel, use_container_width=True)

                        progress.progress(60, text="Loading models...")

                        # Step 3: Run models
                        loaded = {}
                        for i, m in enumerate(models_to_use):
                            model = load_model(m)
                            if model is not None:
                                loaded[m] = model
                            frac = 60 + int(20 * (i + 1) / len(models_to_use))
                            progress.progress(frac, text=f"Loaded {m}...")

                        progress.progress(85, text="Running predictions...")

                        # Run inference
                        pred_results = []
                        for m_name, model in loaded.items():
                            with torch.no_grad():
                                logits = model(frames_tensor, mel_tensor)
                                probs = F_torch.softmax(logits, dim=1)
                                real_prob = probs[0][0].item() * 100
                                fake_prob = probs[0][1].item() * 100
                                confidence, pred = torch.max(probs, dim=1)
                                pred_label = "FAKE" if pred.item() == 1 else "REAL"
                                conf_val = confidence.item() * 100

                            pred_results.append({
                                "Model": m_name,
                                "Prediction": pred_label,
                                "Confidence": conf_val,
                                "Real %": real_prob,
                                "Fake %": fake_prob,
                                "Architecture": f"{MODEL_INFO[m_name]['visual']} + {MODEL_INFO[m_name]['audio']}",
                            })

                        progress.progress(100, text="Done!")

                        # Step 4: Display results
                        st.markdown("---")
                        st.subheader("Prediction Results")

                        # Big verdict from majority vote
                        all_preds = [p["Prediction"] for p in pred_results]
                        fake_c = all_preds.count("FAKE")
                        real_c = all_preds.count("REAL")
                        majority_verdict = "FAKE" if fake_c > real_c else "REAL"
                        avg_conf = sum(p["Confidence"] for p in pred_results) / len(pred_results)

                        verdict_color = "#FF4B4B" if majority_verdict == "FAKE" else "#00CC96"
                        st.markdown(
                            f'<div style="text-align:center; padding:20px; background-color:{verdict_color}20; '
                            f'border-radius:10px; border:2px solid {verdict_color};">'
                            f'<h1 style="color:{verdict_color}; margin:0;">{majority_verdict}</h1>'
                            f'<p style="font-size:18px; margin:5px 0 0 0;">'
                            f'{avg_conf:.1f}% avg confidence ({real_c} REAL / {fake_c} FAKE out of {len(pred_results)} models)</p></div>',
                            unsafe_allow_html=True,
                        )

                        st.markdown("")

                        # Per-model results table
                        df_pred = pd.DataFrame(pred_results)
                        st.dataframe(df_pred, use_container_width=True, hide_index=True)

                        # Confidence bar chart
                        st.subheader("Model Confidence Breakdown")

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name="Real %",
                            x=[r["Model"] for r in pred_results],
                            y=[r["Real %"] for r in pred_results],
                            marker_color="#00CC96",
                            text=[f'{r["Real %"]:.1f}%' for r in pred_results],
                            textposition="inside",
                        ))
                        fig.add_trace(go.Bar(
                            name="Fake %",
                            x=[r["Model"] for r in pred_results],
                            y=[r["Fake %"] for r in pred_results],
                            marker_color="#FF4B4B",
                            text=[f'{r["Fake %"]:.1f}%' for r in pred_results],
                            textposition="inside",
                        ))
                        fig.update_layout(
                            barmode="stack",
                            yaxis_title="Probability (%)",
                            yaxis_range=[0, 105],
                            height=400,
                            template="plotly_white",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Consensus
                        predictions = [r["Prediction"] for r in pred_results]
                        fake_count = predictions.count("FAKE")
                        real_count = predictions.count("REAL")
                        total_m = len(predictions)

                        st.subheader("Model Consensus")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Models say REAL", f"{real_count}/{total_m}")
                        with col2:
                            st.metric("Models say FAKE", f"{fake_count}/{total_m}")
                        with col3:
                            if fake_count == total_m:
                                st.metric("Consensus", "FAKE (unanimous)")
                            elif real_count == total_m:
                                st.metric("Consensus", "REAL (unanimous)")
                            else:
                                majority = "FAKE" if fake_count > real_count else "REAL"
                                st.metric("Consensus", f"{majority} (majority)")

                        progress.empty()

                finally:
                    if os.path.exists(tmp_video_path):
                        os.remove(tmp_video_path)


# ================================================================
# PAGE 6: PER-MODEL DETAILS
# ================================================================

elif page == "Per-Model Details":
    st.title("Per-Model Details")

    selected = st.selectbox("Select Model", list(MODEL_INFO.keys()))

    if selected not in all_results:
        st.error(f"No results found for {selected}")
    else:
        r = all_results[selected]
        info = MODEL_INFO[selected]

        # Header
        st.subheader(f"{selected} — {info['visual']} + {info['audio']}")
        st.markdown(f"**Research Question:** *{info['question']}*")

        st.markdown("---")

        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Val Accuracy", f"{r['best_val_acc']:.2f}%")
        with col2:
            st.metric("Val F1 Score", f"{r['best_val_f1']:.4f}")
        with col3:
            st.metric("Epochs", r["epochs_run"])
        with col4:
            st.metric("Parameters", f"{r['parameters']:,}")
        with col5:
            st.metric("Pair", info["pair"])

        st.markdown("---")

        # Training curves
        st.subheader("Training History")

        history = r["history"]
        epochs = [h["epoch"] for h in history]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy / F1"),
        )

        color = PAIR_COLORS[info["pair"]]

        fig.add_trace(go.Scatter(x=epochs, y=[h["train_loss"] for h in history],
                                  mode="lines+markers", name="Train Loss", line=dict(color=color)),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=epochs, y=[h["val_loss"] for h in history],
                                  mode="lines+markers", name="Val Loss", line=dict(color=color, dash="dash")),
                      row=1, col=2)

        fig.add_trace(go.Scatter(x=epochs, y=[h["train_acc"] for h in history],
                                  mode="lines+markers", name="Train Acc", line=dict(color=color)),
                      row=2, col=1)

        fig.add_trace(go.Scatter(x=epochs, y=[h["val_acc"] for h in history],
                                  mode="lines+markers", name="Val Acc", line=dict(color=color)),
                      row=2, col=2)

        fig.add_trace(go.Scatter(x=epochs, y=[h["val_f1"] * 100 for h in history],
                                  mode="lines+markers", name="Val F1 x100", line=dict(color="#FF6692", dash="dot")),
                      row=2, col=2)

        fig.update_layout(height=600, template="plotly_white", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Epoch-by-epoch table
        st.subheader("Epoch-by-Epoch Metrics")

        history_df = pd.DataFrame(history)
        history_df.columns = ["Epoch", "Train Loss", "Train Acc (%)", "Val Loss", "Val Acc (%)", "Val F1"]
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        # Architecture details
        st.markdown("---")
        st.subheader("Architecture")
        st.markdown(f"""
| Component | Details |
|-----------|---------|
| **Visual Backbone** | {info['visual']} |
| **Audio Backbone** | {info['audio']} |
| **Fusion Strategy** | {info['fusion']} |
| **Pair** | {info['pair']} |
| **Total Parameters** | {r['parameters']:,} |
| **Epochs Trained** | {r['epochs_run']} |
| **Best Val Accuracy** | {r['best_val_acc']:.2f}% |
| **Best Val F1** | {r['best_val_f1']:.4f} |
""")

        # Pair comparison
        st.markdown("---")
        st.subheader(f"Pair Comparison — {info['pair']}")

        pair_models = [m for m, i in MODEL_INFO.items() if i["pair"] == info["pair"]]
        pair_data = []
        for m in pair_models:
            if m in all_results:
                pr = all_results[m]
                pair_data.append({
                    "Model": m,
                    "Visual": MODEL_INFO[m]["visual"],
                    "Audio": MODEL_INFO[m]["audio"],
                    "Val Acc (%)": pr["best_val_acc"],
                    "Val F1": pr["best_val_f1"],
                    "Winner": "👑" if pr["best_val_acc"] == max(all_results[pm]["best_val_acc"] for pm in pair_models if pm in all_results) else "",
                })

        st.dataframe(pd.DataFrame(pair_data), use_container_width=True, hide_index=True)
