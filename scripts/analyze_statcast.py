#!/usr/bin/env python3
"""
Download and analyze Statcast pitch data for April 2024
"""

import pandas as pd
import numpy as np
from pybaseball import statcast
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("=" * 80)
print("DOWNLOADING STATCAST DATA FOR APRIL 2024")
print("=" * 80)
print("\nThis may take a few minutes...\n")

# Download Statcast data for April 2024
# April 2024: 2024-04-01 to 2024-04-30
start_date = "2024-04-01"
end_date = "2024-04-30"

try:
    data = statcast(start_dt=start_date, end_dt=end_date)

    # Save to CSV for future use
    data.to_csv('statcast_april_2024.csv', index=False)
    print(f"✓ Data downloaded and saved to statcast_april_2024.csv")
    print(f"✓ Total pitches: {len(data):,}")

    print("\n" + "=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)

    # Display dataset shape
    print(f"\nDataset Shape: {data.shape[0]:,} rows × {data.shape[1]} columns")

    # Display all available columns
    print("\n" + "=" * 80)
    print("AVAILABLE COLUMNS")
    print("=" * 80)
    print("\nColumn listing (total: {}):\n".format(len(data.columns)))

    for i, col in enumerate(data.columns, 1):
        print(f"{i:3d}. {col}")

    # Basic statistics
    print("\n" + "=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)

    # Key categorical variables
    print("\n--- Game State Variables ---")
    if 'balls' in data.columns:
        print(f"\nBalls distribution:\n{data['balls'].value_counts().sort_index()}")
    if 'strikes' in data.columns:
        print(f"\nStrikes distribution:\n{data['strikes'].value_counts().sort_index()}")
    if 'outs_when_up' in data.columns:
        print(f"\nOuts distribution:\n{data['outs_when_up'].value_counts().sort_index()}")

    print("\n--- Pitch Types ---")
    if 'pitch_type' in data.columns:
        print(f"\nPitch type counts:\n{data['pitch_type'].value_counts()}")

    print("\n--- Events ---")
    if 'events' in data.columns:
        print(f"\nTop 10 events:\n{data['events'].value_counts().head(10)}")

    # Numerical statistics
    print("\n--- Key Numerical Variables ---")
    numeric_cols = ['release_speed', 'release_spin_rate', 'effective_speed',
                    'launch_speed', 'launch_angle', 'hit_distance_sc',
                    'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle']

    available_numeric_cols = [col for col in numeric_cols if col in data.columns]

    if available_numeric_cols:
        print(f"\n{data[available_numeric_cols].describe()}")

    # Memory usage
    print("\n" + "=" * 80)
    print("MEMORY USAGE")
    print("=" * 80)
    memory_usage = data.memory_usage(deep=True).sum() / 1024**2
    print(f"\nTotal memory usage: {memory_usage:.2f} MB")

    print("\n" + "=" * 80)
    print("DATA PREVIEW (First 5 rows of selected columns)")
    print("=" * 80)

    preview_cols = ['game_date', 'player_name', 'pitch_type', 'release_speed',
                    'balls', 'strikes', 'outs_when_up', 'events', 'description']
    available_preview_cols = [col for col in preview_cols if col in data.columns]

    if available_preview_cols:
        print(f"\n{data[available_preview_cols].head()}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nData file saved: statcast_april_2024.csv")
    print(f"Total pitches analyzed: {len(data):,}")

except Exception as e:
    print(f"Error downloading or processing data: {e}")
    import traceback
    traceback.print_exc()


def encode_game_state(row):
    """
    Encode the game state from a Statcast data row.

    Returns a tuple of (outs, bases_state) where:
    - outs: 0, 1, or 2
    - bases_state: 0-7 representing which bases are occupied
      0 = empty
      1 = 1st only
      2 = 2nd only
      3 = 1st & 2nd
      4 = 3rd only
      5 = 1st & 3rd
      6 = 2nd & 3rd
      7 = bases loaded
    """
    # Get outs (0, 1, or 2)
    outs = int(row['outs_when_up']) if pd.notna(row['outs_when_up']) else 0

    # Encode bases as binary: 1st base = 1, 2nd base = 2, 3rd base = 4
    # Sum them to get 0-7
    on_1b = 1 if pd.notna(row.get('on_1b')) else 0
    on_2b = 2 if pd.notna(row.get('on_2b')) else 0
    on_3b = 4 if pd.notna(row.get('on_3b')) else 0

    bases_state = on_1b + on_2b + on_3b

    return (outs, bases_state)


def decode_bases_state(bases_state):
    """
    Convert bases state number (0-7) to human-readable string.
    """
    if bases_state == 0:
        return "Empty"

    bases = []
    if bases_state & 1:  # Check bit 0
        bases.append("1st")
    if bases_state & 2:  # Check bit 1
        bases.append("2nd")
    if bases_state & 4:  # Check bit 2
        bases.append("3rd")

    return " & ".join(bases)


# Test the encoding function on some example rows
if 'data' in locals() and len(data) > 0:
    print("\n" + "=" * 80)
    print("TESTING GAME STATE ENCODING")
    print("=" * 80)

    # Sample a few different game states
    print("\nTesting on sample rows:\n")

    # Get some example rows with different states
    sample_indices = [0, 100, 1000, 5000, 10000]

    for idx in sample_indices:
        if idx < len(data):
            row = data.iloc[idx]
            outs, bases = encode_game_state(row)
            bases_desc = decode_bases_state(bases)

            print(f"Row {idx}:")
            print(f"  Outs: {outs}")
            print(f"  Bases State: {bases} ({bases_desc})")
            print(f"  Raw: 1B={row.get('on_1b', 'N/A')}, "
                  f"2B={row.get('on_2b', 'N/A')}, "
                  f"3B={row.get('on_3b', 'N/A')}")
            print()
