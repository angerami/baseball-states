from datasets import Dataset, concatenate_datasets
from pathlib import Path
import pandas as pd
from pybaseball import statcast

# Only keep columns you actually need
KEEP_COLUMNS = [
    'game_pk', 'at_bat_number', 'pitch_number',
    'outs_when_up', 'on_1b', 'on_2b', 'on_3b',
    'events',
    'batter', 'pitcher',
    'inning', 'inning_topbot',
    'game_year', 'game_type'  # For filtering rules
]
CACHE_DIR = Path("data/cache")
OUTPUT_DIR = Path("data/pitches")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = []
for year in range(2015, 2024):
    START_DATE = f'{year}-01-01'
    END_DATE = f'{year}-12-31'
    cache_file = CACHE_DIR / f"statcast_{START_DATE}_{END_DATE}.csv"
    
    print(f"Processing {year}...")
    
    if cache_file.exists():
        df = pd.read_csv(cache_file, usecols=KEEP_COLUMNS)  # Only read needed columns
    else:
        df = statcast(start_dt=START_DATE, end_dt=END_DATE)
        df.to_csv(cache_file, index=False)
        df = df[KEEP_COLUMNS]  # Prune immediately after download
    
    dataset = Dataset.from_pandas(df, preserve_index=False)
    datasets.append(dataset)
    del df
    
    print(f"  {len(dataset):,} pitches")

print("\nCombining datasets...")
combined_dataset = concatenate_datasets(datasets)
combined_dataset.save_to_disk(OUTPUT_DIR)

print(f"\nSaved to: {OUTPUT_DIR}")
print(f"Total pitches: {len(combined_dataset):,}")