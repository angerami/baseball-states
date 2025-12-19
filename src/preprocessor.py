"""Group pitch-level data into plate appearances and save as HuggingFace dataset."""

from datasets import Dataset
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PreprocessorConfig:
    """Configuration for preprocessing pitch data into plate appearances."""
    input_path: str = "data/pitches"
    output_path: str = "data/plate_appearances"


class PlateAppearancePreprocessor:
    """Converts pitch-level data into plate appearance records."""

    def __init__(self, config: PreprocessorConfig = PreprocessorConfig()):
        """Initialize preprocessor with configuration."""
        self.config = config
        self.input_path = Path(config.input_path)
        self.output_path = Path(config.output_path)

    def encode_game_state(self, row):
        """Encode game state as (outs, bases) where bases is 0-7."""
        outs = int(row['outs_when_up']) if pd.notna(row['outs_when_up']) else 0
        on_1b = 1 if pd.notna(row.get('on_1b')) else 0
        on_2b = 2 if pd.notna(row.get('on_2b')) else 0
        on_3b = 4 if pd.notna(row.get('on_3b')) else 0
        return (outs, on_1b + on_2b + on_3b)

    def process(self) -> pd.DataFrame:
        """
        Process pitch-level data into plate appearances.

        Returns:
            DataFrame containing plate appearance records.
        """
        print(f"Loading pitch data from: {self.input_path}")
        dataset = Dataset.load_from_disk(self.input_path)
        df = dataset.to_pandas()
        print(f"Loaded {len(df):,} pitches")

        print("Grouping by plate appearance...")
        grouped = df.groupby(['game_pk', 'at_bat_number'])

        pa_records = []
        for (game_pk, at_bat_number), pitches in grouped:
            pitches = pitches.sort_values('pitch_number') if 'pitch_number' in pitches.columns else pitches
            first = pitches.iloc[0]
            last = pitches.iloc[-1]

            initial_outs, initial_bases = self.encode_game_state(first)

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

        return pa_df

    def save(self, df: pd.DataFrame):
        """Save plate appearances as HuggingFace dataset."""
        self.output_path.mkdir(parents=True, exist_ok=True)

        pa_dataset = Dataset.from_pandas(df)
        pa_dataset.save_to_disk(self.output_path)

        print(f"Saved to: {self.output_path}")
        print("\nSummary:")
        print(f"  Total PAs: {len(pa_dataset):,}")
        print(f"  Avg pitches/PA: {df['num_pitches'].mean():.2f}")
        print("\nTop outcomes:")
        print(df['outcome'].value_counts().head(10))

    def run(self) -> pd.DataFrame:
        """Run the complete preprocessing pipeline."""
        df = self.process()
        self.save(df)
        return df