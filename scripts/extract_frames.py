import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN

# ==========================
# CONFIGURATION
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_DIR = os.path.join(BASE_DIR, "Data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "processed", "frames")

FRAMES_PER_VIDEO = 16
IMG_SIZE = 224

# ==========================
# INITIALIZE MTCNN
# ==========================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

detector = MTCNN(
    keep_all=True,
    device=device
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# FACE EXTRACTION FUNCTION
# ==========================

def extract_faces_from_video(video_path, output_folder):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < FRAMES_PER_VIDEO:
        cap.release()
        return

    frame_indices = [
        int(i * total_frames / FRAMES_PER_VIDEO)
        for i in range(FRAMES_PER_VIDEO)
    ]

    saved_count = 0
    current_frame = 0

    while cap.isOpened() and saved_count < FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frame_indices:

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            boxes, probs = detector.detect(rgb_frame)

            if boxes is not None and len(boxes) > 0:

                # Select largest detected face
                areas = [
                    (box[2] - box[0]) * (box[3] - box[1])
                    for box in boxes
                ]

                largest_idx = areas.index(max(areas))
                x1, y1, x2, y2 = boxes[largest_idx]

                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = int(x2)
                y2 = int(y2)

                face = rgb_frame[y1:y2, x1:x2]

                if face.size != 0:
                    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

                    output_path = os.path.join(
                        output_folder,
                        f"frame_{saved_count + 1}.jpg"
                    )

                    cv2.imwrite(
                        output_path,
                        cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                    )

                    saved_count += 1

        current_frame += 1

    cap.release()


# ==========================
# PROCESS SPLIT
# ==========================

def process_split(csv_file):

    df = pd.read_csv(os.path.join(CSV_DIR, csv_file))
    split_name = csv_file.replace("_subset.csv", "")
    split_output_dir = os.path.join(OUTPUT_DIR, split_name)

    os.makedirs(split_output_dir, exist_ok=True)

    print(f"\nProcessing {split_name}...")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        video_path = row["video_path"]
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        video_output_folder = os.path.join(split_output_dir, video_name)

        # Skip if already processed
        if (
            os.path.exists(video_output_folder)
            and len(os.listdir(video_output_folder)) == FRAMES_PER_VIDEO
        ):
            continue

        os.makedirs(video_output_folder, exist_ok=True)

        extract_faces_from_video(video_path, video_output_folder)


# ==========================
# MAIN
# ==========================

def main():
    process_split("train_subset.csv")
    process_split("dev_subset.csv")
    process_split("test_subset.csv")

    print("\nFrame extraction completed successfully.")


if __name__ == "__main__":
    main()