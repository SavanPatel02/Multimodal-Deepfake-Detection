import os
import numpy as np
import librosa
from tqdm import tqdm

# ==========================
# CONFIGURATION
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUDIO_DIR = os.path.join(BASE_DIR, "Data", "processed", "audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "processed", "mels")

SAMPLE_RATE = 16000
DURATION = 4  # seconds
TARGET_LENGTH = SAMPLE_RATE * DURATION

N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_split(split_name):
    input_dir = os.path.join(AUDIO_DIR, split_name)
    output_split_dir = os.path.join(OUTPUT_DIR, split_name)

    os.makedirs(output_split_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    print(f"\nGenerating Mel Spectrograms for {split_name}...")

    for file in tqdm(files):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_split_dir, file.replace(".wav", ".npy"))

        # Skip if already processed
        if os.path.exists(output_path):
            continue

        try:
            # ==========================
            # Load Audio
            # ==========================
            y, sr = librosa.load(input_path, sr=SAMPLE_RATE)

            # ==========================
            # Audio Normalization
            # ==========================
            y = y / (np.max(np.abs(y)) + 1e-6)

            # ==========================
            # Trim or Pad Audio
            # ==========================
            if len(y) > TARGET_LENGTH:
                y = y[:TARGET_LENGTH]
            else:
                y = np.pad(y, (0, TARGET_LENGTH - len(y)))

            # ==========================
            # Generate Mel Spectrogram
            # ==========================
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            )

            # ==========================
            # Convert to Log Scale
            # ==========================
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # ==========================
            # Normalize Spectrogram (0–1)
            # ==========================
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

            # ==========================
            # Save Mel Spectrogram
            # ==========================
            np.save(output_path, mel_db.astype(np.float32))

        except Exception as e:
            print(f"Error processing {file}: {e}")


def main():
    process_split("train")
    process_split("dev")
    process_split("test")

    print("\nMel Spectrogram generation completed successfully.")


if __name__ == "__main__":
    main()