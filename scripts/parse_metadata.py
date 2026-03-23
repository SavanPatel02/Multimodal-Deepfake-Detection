import json
import os
import pandas as pd
from tqdm import tqdm


# CONFIGURATION


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METADATA_PATH = os.path.join(BASE_DIR, "Data", "raw", "LAV-DF", "metadata.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "processed")
VIDEO_BASE_PATH = os.path.join(BASE_DIR, "Data", "raw", "LAV-DF")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("Loading metadata...")

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    train_data = []
    dev_data = []
    test_data = []

    print("Parsing metadata...")

    for entry in tqdm(metadata):

        filename = entry["file"]
        split = entry["split"]

      
        # LABEL DEFINITION FOR LAV-DF
        
        is_fake = (
            entry["modify_video"] or
            entry["modify_audio"] or
            entry["n_fakes"] > 0
        )

        label = 1 if is_fake else 0  # 1 = Fake, 0 = Real

        video_path = os.path.join(VIDEO_BASE_PATH, filename)

        row = {
            "video_path": video_path,
            "label": label,
            "split": split
        }

        if split == "train":
            train_data.append(row)
        elif split == "dev":
            dev_data.append(row)
        elif split == "test":
            test_data.append(row)

    # Convert to DataFrame
    train_df = pd.DataFrame(train_data)
    dev_df = pd.DataFrame(dev_data)
    test_df = pd.DataFrame(test_data)

    # Save CSV files
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(OUTPUT_DIR, "dev.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print("\nCSV files generated successfully!")
    print(f"Train samples: {len(train_df)}")
    print(f"Dev samples: {len(dev_df)}")
    print(f"Test samples: {len(test_df)}")


if __name__ == "__main__":
    main()