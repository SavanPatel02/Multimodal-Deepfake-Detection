import os
import pandas as pd
import librosa
import soundfile as sf
from moviepy import VideoFileClip
from tqdm import tqdm

# ==========================
# CONFIGURATION
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_DIR = os.path.join(BASE_DIR, "Data", "processed")
RAW_VIDEO_DIR = os.path.join(BASE_DIR, "Data", "raw", "LAV-DF")
OUTPUT_AUDIO_DIR = os.path.join(BASE_DIR, "Data", "processed", "audio")

TARGET_SR = 16000  # 16kHz mono

os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)


def extract_audio_from_video(video_path, output_path):
    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio

        if audio is None:
            return False

        temp_wav = output_path.replace(".wav", "_temp.wav")
        audio.write_audiofile(temp_wav, logger=None)

        # Load and resample
        y, sr = librosa.load(temp_wav, sr=TARGET_SR, mono=True)

        sf.write(output_path, y, TARGET_SR)

        os.remove(temp_wav)
        clip.close()

        return True

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False


def process_split(csv_file):
    df = pd.read_csv(os.path.join(CSV_DIR, csv_file))

    split_name = csv_file.replace("_subset.csv", "")
    split_output_dir = os.path.join(OUTPUT_AUDIO_DIR, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    print(f"\nProcessing {split_name}...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_rel_path = row["video_path"].replace(RAW_VIDEO_DIR + os.sep, "")
        video_path = os.path.join(RAW_VIDEO_DIR, video_rel_path)

        file_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(split_output_dir, file_name + ".wav")

        if not os.path.exists(output_path):
            extract_audio_from_video(video_path, output_path)


def main():
    process_split("train_subset.csv")
    process_split("dev_subset.csv")
    process_split("test_subset.csv")

    print("\nAudio extraction completed successfully.")


if __name__ == "__main__":
    main()