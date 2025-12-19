#!/Users/angerami/Desktop/Materials/baseball-states/venv/bin/python3
"""Download Statcast data and save as HuggingFace dataset."""

from pybaseball import statcast
from datasets import Dataset
import pandas as pd
from pathlib import Path

# Config
START_DATE = "2024-04-01"
END_DATE = "2024-04-30"
CACHE_DIR = Path("data/cache")
OUTPUT_DIR = Path("data/pitches")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cache_file = CACHE_DIR / f"statcast_{START_DATE}_{END_DATE}.csv"

print(f"Downloading Statcast data: {START_DATE} to {END_DATE}")

# Download or load from cache
if cache_file.exists():
    print(f"Loading from cache: {cache_file}")
    df = pd.read_csv(cache_file)
else:
    print("Downloading from Baseball Savant...")
    df = statcast(start_dt=START_DATE, end_dt=END_DATE)
    df.to_csv(cache_file, index=False)
    print(f"Cached to: {cache_file}")

print(f"Total pitches: {len(df):,}")

# Convert to HF dataset and save
dataset = Dataset.from_pandas(df)
dataset.save_to_disk(OUTPUT_DIR)

print(f"Saved to: {OUTPUT_DIR}")
print(f"Columns: {len(dataset.features)}")
print(f"Rows: {len(dataset):,}")
