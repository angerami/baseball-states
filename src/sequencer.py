from datasets import Dataset
import pandas as pd
from pathlib import Path
from typing import List, Literal
from dataclasses import dataclass
from baseball_states.constants import Vocabulary

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
                sequence.append(Vocabulary.START_INNING)

            for _, row in group.iterrows():
                token = Vocabulary.format_state(int(row['outs']), int(row['bases']))
                sequence.append(token)

            if config.include_inning_boundaries:
                sequence.append(Vocabulary.END_INNING)

            sequences.append(sequence)

        return sequences

    def _create_game_sequences(self, config: SequenceConfig) -> List[List[str]]:
        """Create one sequence per game."""
        sequences = []

        for game_pk, game_df in self.df.groupby('game_pk'):
            game_df = game_df.sort_values(['inning', 'inning_topbot', 'at_bat_number'])

            sequence = []
            if config.include_game_boundaries:
                sequence.append(Vocabulary.START_GAME)

            current_inning = None
            for _, row in game_df.iterrows():
                inning_key = (row['inning'], row['inning_topbot'])

                # Add inning boundaries if needed
                if config.include_inning_boundaries and inning_key != current_inning:
                    if current_inning is not None:
                        sequence.append(Vocabulary.END_INNING)
                    sequence.append(Vocabulary.START_INNING)
                    current_inning = inning_key

                token = Vocabulary.format_state(int(row['outs']), int(row['bases']))
                sequence.append(token)

            # Close final inning
            if config.include_inning_boundaries:
                sequence.append(Vocabulary.END_INNING)

            if config.include_game_boundaries:
                sequence.append(Vocabulary.END_GAME)

            sequences.append(sequence)

        return sequences

    def to_pandas(self, sequences):
        data = {'sequence': sequences, 'length': [len(seq) for seq in sequences]}
        return pd.DataFrame(data)
    
    def save_sequences(self, sequences, output_path: str):
        """Save sequences to disk as HuggingFace dataset."""

        df = sequences if isinstance(sequences, pd.DataFrame) else self.to_pandas(sequences) 
        dataset = Dataset.from_pandas(df)
        dataset.save_to_disk(output_path)
        print(f"Saved {len(sequences)} sequences to {output_path}")
