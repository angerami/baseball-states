# scripts/run_pipeline.py
from datasets import load_from_disk
from baseball_states.preprocessor import preprocess_to_pa_map, pa_to_sequence_map, sequence_to_tokens_map
from baseball_states.tokenizer import GameStateTokenizer

DATA_DIR = "data"
PITCHES_DS = f'{DATA_DIR}/pitches'
PA_DS = f'{DATA_DIR}/plate_appearances'
SEQUENCE_DS = f'{DATA_DIR}/sequences_inning'
TOKENIZED_DS = f'{DATA_DIR}/tokens_inning'


def run_pipeline(
    pitches_path=PITCHES_DS,
    pa_path=PA_DS,
    seq_path=SEQUENCE_DS,
    tokenized_path=TOKENIZED_DS,
    inning_level=True,
    skip_extra_innings=False,
    filter_special_extra_innings=True,
    filter_invalid_starts=True,
    filter_game_types=True
):
    # Step 1: Preprocessing
    print("Step 1: Grouping pitches into plate appearances...")
    pitches_dataset = load_from_disk(pitches_path)
    pa_dataset = pitches_dataset.map(
        preprocess_to_pa_map,
        batched=True,
        batch_size=10000,
        remove_columns=pitches_dataset.column_names
    )

    # Sort the entire dataset to ensure chronological order (only if Statcast format)
    if 'game_pk' in pa_dataset.column_names:
        print("  Sorting plate appearances...")
        pa_dataset = pa_dataset.sort(['game_pk', 'inning', 'inning_topbot', 'at_bat_number'])
    else:
        print("  Skipping sort (Retrosheet data already sorted)")

    pa_dataset.save_to_disk(pa_path)
    print(f"  Saved {len(pa_dataset):,} plate appearances")
    
    # Step 2: Sequencing
    print("\nStep 2: Creating sequences...")
    print(f"  Filters: skip_extra={skip_extra_innings}, filter_special_extra={filter_special_extra_innings}")
    print(f"           filter_invalid_starts={filter_invalid_starts}, filter_game_types={filter_game_types}")

    sequence_dataset = pa_dataset.map(
        lambda examples: pa_to_sequence_map(
            examples,
            inning_level=inning_level,
            skip_extra_innings=skip_extra_innings,
            filter_special_extra_innings=filter_special_extra_innings,
            filter_invalid_starts=filter_invalid_starts,
            filter_game_types=filter_game_types
        ),
        batched=True,
        batch_size=5000,
        remove_columns=pa_dataset.column_names
    )
    sequence_dataset.save_to_disk(seq_path)
    print(f"  Saved {len(sequence_dataset):,} sequences")
    
    # Step 3: Tokenization
    print("\nStep 3: Tokenizing sequences...")

    tokenizer = GameStateTokenizer()
    tokenized_dataset = sequence_dataset.map(
        lambda examples: sequence_to_tokens_map(examples, tokenizer=tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=['sequence']
    )
    tokenized_dataset.save_to_disk(tokenized_path)
    print(f"  Saved {len(tokenized_dataset):,} tokenized sequences")
    
    print("\n✓ Pipeline complete!")
    return tokenized_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run preprocessing pipeline on baseball data')
    parser.add_argument('--pitches-path', type=str, default=PITCHES_DS,
                       help='Input path for pitch/PA data')
    parser.add_argument('--pa-path', type=str, default=PA_DS,
                       help='Output path for plate appearances')
    parser.add_argument('--seq-path', type=str, default=SEQUENCE_DS,
                       help='Output path for sequences')
    parser.add_argument('--tokenized-path', type=str, default=TOKENIZED_DS,
                       help='Output path for tokenized data')
    parser.add_argument('--inning-level', action='store_true', default=True,
                       help='Create sequences per inning (default: True)')
    parser.add_argument('--game-level', action='store_true',
                       help='Create sequences per game instead of per inning')
    parser.add_argument('--skip-extra-innings', action='store_true',
                       help='Skip all innings > 9')
    parser.add_argument('--no-filter-special-extra', action='store_true',
                       help='Include extra innings with special rules')
    parser.add_argument('--no-filter-invalid-starts', action='store_true',
                       help='Include innings with invalid starting states')
    parser.add_argument('--no-filter-game-types', action='store_true',
                       help='Include non-regular season games')

    args = parser.parse_args()

    run_pipeline(
        pitches_path=args.pitches_path,
        pa_path=args.pa_path,
        seq_path=args.seq_path,
        tokenized_path=args.tokenized_path,
        inning_level=not args.game_level,
        skip_extra_innings=args.skip_extra_innings,
        filter_special_extra_innings=not args.no_filter_special_extra,
        filter_invalid_starts=not args.no_filter_invalid_starts,
        filter_game_types=not args.no_filter_game_types
    )