import os
import pandas as pd

# ==========================
# CONFIGURATION
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "Data", "processed")

TRAIN_SIZE = 10000
DEV_SIZE = 2000
TEST_SIZE = 2000

RANDOM_SEED = 42


def create_subset(file_name, sample_size):
    path = os.path.join(PROCESSED_DIR, file_name)
    df = pd.read_csv(path)

    # Stratified sampling to maintain real/fake balance
    subset_df = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(
          n=min(len(x), sample_size // 2),
          random_state=RANDOM_SEED
      ))
      .reset_index(drop=True)
)

    output_name = file_name.replace(".csv", "_subset.csv")
    subset_df.to_csv(os.path.join(PROCESSED_DIR, output_name), index=False)

    print(f"{output_name} created with {len(subset_df)} samples")


def main():
    create_subset("train.csv", TRAIN_SIZE)
    create_subset("dev.csv", DEV_SIZE)
    create_subset("test.csv", TEST_SIZE)


if __name__ == "__main__":
    main()