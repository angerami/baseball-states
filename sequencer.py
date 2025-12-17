#!/Users/angerami/Desktop/Materials/baseball-states/venv/bin/python3
"""
Create token sequences from plate appearance data.
Sequences represent game states with special tokens for game/inning boundaries.
"""

from datasets import Dataset
import pandas as pd
from pathlib import Path
from typing import List, Literal
from dataclasses import dataclass

# Centralized token definitions
BASES_LABELS = ['Empty', '1st', '2nd', '1st&2nd', '3rd', '1st&3rd', '2nd&3rd', 'Loaded']

SPECIAL_TOKENS = {
    'start_game': '<START_GAME>',
    'end_game': '<END_GAME>',
    'start_inning': '<START_INNING>',
    'end_inning': '<END_INNING>',
}

def get_state_token(outs: int, bases: int) -> str:
    """Convert (outs, bases) to token string like '1_2nd&3rd'."""
    return f"{outs}_{BASES_LABELS[bases]}"

def get_all_state_tokens() -> List[str]:
    """Get all 24 possible state tokens."""
    tokens = []
    for outs in range(3):
        for bases in range(8):
            tokens.append(get_state_token(outs, bases))
    return tokens

def get_vocabulary() -> List[str]:
    """Get complete vocabulary: special tokens + state tokens."""
    return list(SPECIAL_TOKENS.values()) + get_all_state_tokens()

@dataclass
class SequenceConfig:
    """Configuration for sequence generation."""
    level: Literal['inning', 'game'] = 'inning'
    include_game_boundaries: bool = True
    include_inning_boundaries: bool = True

class GameStateSequencer:
    """Creates token sequences from plate appearance data."""

    def __init__(self, data_path: str = "data/plate_appearances"):
        """Load PA dataset."""
        self.data_path = Path(data_path)
        dataset = Dataset.load_from_disk(self.data_path)
        self.df = dataset.to_pandas()

        # Rename columns for consistency
        self.df = self.df.rename(columns={
            'initial_outs': 'outs',
            'initial_bases': 'bases'
        })

        self.vocab = get_vocabulary()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def create_sequences(self, config: SequenceConfig = SequenceConfig()) -> List[List[str]]:
        """
        Create token sequences based on configuration.

        Returns:
            List of sequences, where each sequence is a list of token strings.
        """
        if config.level == 'inning':
            return self._create_inning_sequences(config)
        elif config.level == 'game':
            return self._create_game_sequences(config)
        else:
            raise ValueError(f"Unknown level: {config.level}")

    def _create_inning_sequences(self, config: SequenceConfig) -> List[List[str]]:
        """Create one sequence per inning."""
        sequences = []

        # Group by game and inning
        for (game_pk, inning, inning_topbot), group in self.df.groupby(['game_pk', 'inning', 'inning_topbot']):
            group = group.sort_values('at_bat_number')

            sequence = []
            if config.include_inning_boundaries:
                sequence.append(SPECIAL_TOKENS['start_inning'])

            for _, row in group.iterrows():
                token = get_state_token(int(row['outs']), int(row['bases']))
                sequence.append(token)

            if config.include_inning_boundaries:
                sequence.append(SPECIAL_TOKENS['end_inning'])

            sequences.append(sequence)

        return sequences

    def _create_game_sequences(self, config: SequenceConfig) -> List[List[str]]:
        """Create one sequence per game."""
        sequences = []

        for game_pk, game_df in self.df.groupby('game_pk'):
            game_df = game_df.sort_values(['inning', 'inning_topbot', 'at_bat_number'])

            sequence = []
            if config.include_game_boundaries:
                sequence.append(SPECIAL_TOKENS['start_game'])

            current_inning = None
            for _, row in game_df.iterrows():
                inning_key = (row['inning'], row['inning_topbot'])

                # Add inning boundaries if needed
                if config.include_inning_boundaries and inning_key != current_inning:
                    if current_inning is not None:
                        sequence.append(SPECIAL_TOKENS['end_inning'])
                    sequence.append(SPECIAL_TOKENS['start_inning'])
                    current_inning = inning_key

                token = get_state_token(int(row['outs']), int(row['bases']))
                sequence.append(token)

            # Close final inning
            if config.include_inning_boundaries:
                sequence.append(SPECIAL_TOKENS['end_inning'])

            if config.include_game_boundaries:
                sequence.append(SPECIAL_TOKENS['end_game'])

            sequences.append(sequence)

        return sequences

    def sequences_to_ids(self, sequences: List[List[str]]) -> List[List[int]]:
        """Convert token sequences to integer ID sequences."""
        return [[self.token_to_id[token] for token in seq] for seq in sequences]

    def save_sequences(self, sequences: List[List[str]], output_path: str):
        """Save sequences to disk as HuggingFace dataset."""
        data = {'sequence': sequences, 'length': [len(seq) for seq in sequences]}
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        dataset.save_to_disk(output_path)
        print(f"Saved {len(sequences)} sequences to {output_path}")

def main():
    """Standalone execution for testing/debugging."""
    print("=" * 80)
    print("GAME STATE SEQUENCER")
    print("=" * 80)

    # Initialize
    print("\nLoading data...")
    sequencer = GameStateSequencer()
    print(f"Loaded {len(sequencer.df):,} plate appearances")
    print(f"Vocabulary size: {len(sequencer.vocab)}")

    # Show vocabulary
    print("\nSpecial tokens:")
    for name, token in SPECIAL_TOKENS.items():
        print(f"  {name:20s} -> {token}")

    print(f"\nState tokens (24 total):")
    state_tokens = get_all_state_tokens()
    for i, token in enumerate(state_tokens[:8]):
        print(f"  {token}")
    print(f"  ... ({len(state_tokens)} total)")

    # Create inning sequences
    print("\n" + "=" * 80)
    print("CREATING INNING SEQUENCES")
    print("=" * 80)

    config = SequenceConfig(level='inning', include_inning_boundaries=True)
    sequences = sequencer.create_sequences(config)

    print(f"\nCreated {len(sequences):,} inning sequences")
    print(f"Average sequence length: {sum(len(s) for s in sequences) / len(sequences):.1f}")

    print("\nFirst 5 sequences:")
    for i, seq in enumerate(sequences[:5]):
        print(f"\n  Sequence {i+1} (length={len(seq)}):")
        print(f"    {' -> '.join(seq[:10])}{'...' if len(seq) > 10 else ''}")

    # Save
    output_path = "data/sequences_inning"
    sequencer.save_sequences(sequences, output_path)

    # Create game sequences
    print("\n" + "=" * 80)
    print("CREATING GAME SEQUENCES")
    print("=" * 80)

    config = SequenceConfig(
        level='game',
        include_game_boundaries=True,
        include_inning_boundaries=True
    )
    sequences = sequencer.create_sequences(config)

    print(f"\nCreated {len(sequences):,} game sequences")
    print(f"Average sequence length: {sum(len(s) for s in sequences) / len(sequences):.1f}")

    print("\nFirst sequence preview:")
    seq = sequences[0]
    print(f"  Length: {len(seq)}")
    print(f"  Tokens: {' -> '.join(seq[:15])}...")

    # Save
    output_path = "data/sequences_game"
    sequencer.save_sequences(sequences, output_path)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

if __name__ == "__main__":
    main()
