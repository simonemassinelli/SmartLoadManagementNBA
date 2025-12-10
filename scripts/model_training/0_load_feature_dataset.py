import os
import pandas as pd
from datasets import load_dataset

def load_or_download_nba_dataset(
    hf_repo="Jackdsada/nba_game_features",
    filename="nba_game_features.csv",
    local_dir="../../data"
):
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        print(f"Loading dataset from local file: {local_path}")
        return pd.read_csv(local_path)

    print(f"Dataset not found locally. Downloading from Hugging Face: {hf_repo}")

    ds = load_dataset(hf_repo, data_files=filename)
    df = ds["train"].to_pandas()

    df.to_csv(local_path, index=False)
    print(f"Saved dataset to: {local_path}")

    return df


df = load_or_download_nba_dataset()
print(df.head())
