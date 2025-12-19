#!/Users/angerami/Desktop/Materials/baseball-states/venv/bin/python3
"""Group pitch-level data into plate appearances and save as HuggingFace dataset."""

from datasets import Dataset
import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/pitches")
OUTPUT_DIR = Path("data/plate_appearances")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def encode_game_state(row):
    """Encode game state as (outs, bases) where bases is 0-7."""
    outs = int(row['outs_when_up']) if pd.notna(row['outs_when_up']) else 0
    on_1b = 1 if pd.notna(row.get('on_1b')) else 0
    on_2b = 2 if pd.notna(row.get('on_2b')) else 0
    on_3b = 4 if pd.notna(row.get('on_3b')) else 0
    return (outs, on_1b + on_2b + on_3b)

print(f"Loading pitch data from: {INPUT_DIR}")
dataset = Dataset.load_from_disk(INPUT_DIR)
df = dataset.to_pandas()
print(f"Loaded {len(df):,} pitches")

print("Grouping by plate appearance...")
grouped = df.groupby(['game_pk', 'at_bat_number'])

pa_records = []
for (game_pk, at_bat_number), pitches in grouped:
    pitches = pitches.sort_values('pitch_number') if 'pitch_number' in pitches.columns else pitches
    first = pitches.iloc[0]
    last = pitches.iloc[-1]

    initial_outs, initial_bases = encode_game_state(first)

    pa_records.append({
        'game_pk': game_pk,
        'at_bat_number': at_bat_number,
        'num_pitches': len(pitches),
        'initial_outs': initial_outs,
        'initial_bases': initial_bases,
        'outcome': last['events'] if pd.notna(last['events']) else None,
        'batter': first.get('batter'),
        'pitcher': first.get('pitcher'),
        'inning': first.get('inning'),
        'inning_topbot': first.get('inning_topbot'),
    })

pa_df = pd.DataFrame(pa_records)
print(f"Created {len(pa_df):,} plate appearances")

# Save as HF dataset
pa_dataset = Dataset.from_pandas(pa_df)
pa_dataset.save_to_disk(OUTPUT_DIR)

print(f"Saved to: {OUTPUT_DIR}")
print(f"\nSummary:")
print(f"  Total PAs: {len(pa_dataset):,}")
print(f"  Avg pitches/PA: {pa_df['num_pitches'].mean():.2f}")
print(f"\nTop outcomes:")
print(pa_df['outcome'].value_counts().head(10))
