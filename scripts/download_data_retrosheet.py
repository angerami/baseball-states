#!/usr/bin/env python3
"""
Download and skim Retrosheet play-by-play data to only essential columns for game state modeling.
"""
import pandas as pd
import subprocess
import argparse
from pathlib import Path
import shutil
from datasets import Dataset

# Columns needed to match Statcast preprocessing
COLUMNS_TO_KEEP = [
    # Game identification
    'gid',
    'date',
    
    # Inning context
    'inning',
    'top_bot',
    
    # Game state (pre-event)
    'outs_pre',
    'br1_pre',
    'br2_pre', 
    'br3_pre',
    'score_v',
    'score_h',
    
    # Outcome columns
    'pa',
    'single',
    'double',
    'triple',
    'hr',
    'walk',
    'k',
    'roe',
    'fc',
    'sf',
    'sh',
    'othout',
    
    # Post-state for validation
    'outs_post',
    'runs',
]

RETROSHEET_URL = 'https://www.retrosheet.org/downloads/plays/plays.zip'

def download_retrosheet(cache_dir: Path, force: bool = False) -> Path:
    """Download Retrosheet zip file if not cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / 'retrosheet.full_data.zip'
    
    if zip_path.exists() and not force:
        print(f"Found cached zip: {zip_path}")
        return zip_path
    
    print(f"Downloading from {RETROSHEET_URL}...")
    cmd = ['curl', '-C', '-', '-o', str(zip_path), RETROSHEET_URL]
    subprocess.run(cmd, check=True)
    print(f"Downloaded to {zip_path}")
    return zip_path

def unzip_retrosheet(zip_path: Path, delete_zip: bool = False) -> Path:
    """Unzip retrosheet data."""
    csv_path = zip_path.parent / 'retrosheet.full_data.csv'
    
    if csv_path.exists():
        print(f"Found cached CSV: {csv_path}")
        return csv_path
    
    print(f"Unzipping {zip_path.name}...")
    subprocess.run(['unzip', '-o', str(zip_path), '-d', str(zip_path.parent)], check=True)
    
    # Rename plays.csv to our convention
    extracted = zip_path.parent / 'plays.csv'
    if extracted.exists():
        extracted.rename(csv_path)
        print(f"Renamed to {csv_path.name}")
    
    if delete_zip:
        zip_path.unlink()
        print(f"Deleted {zip_path.name}")
    
    return csv_path

def skim_retrosheet_csv(input_path: Path, output_path: Path, chunk_size: int = 100_000):
    """Read large Retrosheet CSV in chunks and save only needed columns."""
    print(f"\nSkimming {input_path.name}...")
    print(f"Columns to keep: {len(COLUMNS_TO_KEEP)}")

    first_chunk = True
    total_rows = 0

    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size, usecols=COLUMNS_TO_KEEP)):
        # Filter to actual plate appearances
        chunk = chunk[chunk['pa'] == 1].copy()

        # Write chunk to output
        chunk.to_csv(
            output_path,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )

        total_rows += len(chunk)
        first_chunk = False

        if (i + 1) % 10 == 0:
            print(f"  Processed {(i + 1) * chunk_size:,} rows, kept {total_rows:,} PAs")

    print(f"\nDone! Saved {total_rows:,} plate appearances to {output_path.name}")
    print(f"Original size: {input_path.stat().st_size / 1e9:.2f} GB")
    print(f"Skimmed size: {output_path.stat().st_size / 1e9:.2f} GB")

def csv_to_hf_dataset(csv_path: Path, output_path: Path, chunk_size: int = 100_000):
    """Load CSV and save as HuggingFace Dataset, keeping only necessary columns."""
    print(f"\nConverting {csv_path.name} to HuggingFace Dataset...")

    # Only keep columns needed for preprocessing
    KEEP_FOR_PREPROCESSING = [
        'gid',       # game identifier
        'date',      # for extracting year
        'inning',    # inning number
        'top_bot',   # top/bottom of inning
        'outs_pre',  # outs before PA
        'br1_pre',   # runner on 1st
        'br2_pre',   # runner on 2nd
        'br3_pre',   # runner on 3rd
        # Outcome columns
        'single', 'double', 'triple', 'hr', 'walk', 'k',
        'roe', 'fc', 'sf', 'sh', 'othout'
    ]

    chunks = []
    total_rows = 0

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, usecols=KEEP_FOR_PREPROCESSING)):
        chunks.append(chunk)
        total_rows += len(chunk)

        if (i + 1) % 10 == 0:
            print(f"  Processed {(i + 1) * chunk_size:,} rows, kept {total_rows:,} PAs")

    print("Combining chunks...")
    df = pd.concat(chunks, ignore_index=True)

    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print(f"Converting to HuggingFace Dataset...")
    dataset = Dataset.from_pandas(df, preserve_index=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_path)

    print(f"\n✓ Saved {len(dataset):,} plate appearances to {output_path}")
    print(f"Dataset size on disk: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1e9:.2f} GB")
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Download and skim Retrosheet play-by-play data')
    parser.add_argument('--download', action='store_true', help='Download data from Retrosheet')
    parser.add_argument('--cache-dir', type=str, default='./data/retrosheet',
                       help='Directory for caching downloads')
    parser.add_argument('--output', type=str, default='./data/retrosheet/plays_skimmed.csv',
                       help='Output path for skimmed CSV')
    parser.add_argument('--hf-output', type=str, default='./data/retrosheet_pas',
                       help='Output path for HuggingFace dataset')
    parser.add_argument('--save-hf', action='store_true',
                       help='Save as HuggingFace dataset instead of CSV')
    parser.add_argument('--delete-zip', action='store_true',
                       help='Delete zip file after extraction')
    parser.add_argument('--force', action='store_true',
                       help='Force full download and processing pipeline')

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)
    hf_output_path = Path(args.hf_output)

    # Determine what needs to be done
    if args.save_hf:
        # Check if HF dataset already exists
        if hf_output_path.exists() and not args.force:
            print(f"HuggingFace dataset already exists: {hf_output_path}")
            print("Use --force to reprocess")
            return

        # We need the skimmed CSV to create the HF dataset
        csv_path = output_path

        # If CSV doesn't exist, create it first
        if not csv_path.exists() or args.force:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Get input CSV (from cache or download)
            if args.download or args.force:
                zip_path = download_retrosheet(cache_dir, force=args.force)
                full_csv_path = unzip_retrosheet(zip_path, delete_zip=args.delete_zip)
            else:
                # Look for cached CSV
                full_csv_path = cache_dir / 'plays.csv'
                if not full_csv_path.exists():
                    print(f"No cached data found at {full_csv_path}")
                    print("Use --download to fetch from Retrosheet")
                    return

            # Skim the CSV
            skim_retrosheet_csv(full_csv_path, csv_path)

        # Convert to HuggingFace dataset
        csv_to_hf_dataset(csv_path, hf_output_path)

    else:
        # Original CSV-only behavior
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if output already exists
        if output_path.exists() and not args.force:
            print(f"Skimmed data already exists: {output_path}")
            print("Use --force to reprocess")
            return

        # Get input CSV (from cache or download)
        if args.download or args.force:
            zip_path = download_retrosheet(cache_dir, force=args.force)
            csv_path = unzip_retrosheet(zip_path, delete_zip=args.delete_zip)
        else:
            # Look for cached CSV
            csv_path = cache_dir / 'plays.csv'
            if not csv_path.exists():
                print(f"No cached data found at {csv_path}")
                print("Use --download to fetch from Retrosheet")
                return

        # Skim the CSV
        skim_retrosheet_csv(csv_path, output_path)

if __name__ == '__main__':
    main()