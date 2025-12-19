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
    inning_level=True
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
    pa_dataset.save_to_disk(pa_path)
    print(f"  Saved {len(pa_dataset):,} plate appearances")
    
    # Step 2: Sequencing
    print("\nStep 2: Creating sequences...")
    sequence_dataset = pa_dataset.map(
        lambda examples: pa_to_sequence_map(examples, inning_level=inning_level),
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
    run_pipeline()