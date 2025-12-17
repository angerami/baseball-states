# Baseball Game States

Encodes baseball game states from Statcast pitch data for ML/RL applications.

## Setup

```bash
source venv/bin/activate
```

## Usage

```bash
# 1. Download pitch data (saves to data/pitches/)
python download_data.py

# 2. Create plate appearance dataset (saves to data/plate_appearances/)
python create_pa_dataset.py
```

## Working with Data

```python
from datasets import Dataset

# Load datasets
pitches = Dataset.load_from_disk('data/pitches')
pas = Dataset.load_from_disk('data/plate_appearances')

# Convert to pandas for analysis
df = pas.to_pandas()
```

## Game State Encoding

States encoded as `(outs, bases)`:
- **outs**: 0, 1, 2
- **bases**: 0-7 (binary: 1st=1, 2nd=2, 3rd=4)
  - 0=Empty, 1=1st, 2=2nd, 3=1st&2nd, 4=3rd, 5=1st&3rd, 6=2nd&3rd, 7=Loaded

Total: 24 possible states (3 × 8)

### Example

```python
row = df.iloc[0]  # Get first row from DataFrame
outs = row['initial_outs']  # 0, 1, or 2
bases = row['initial_bases']  # 0-7
```

## Data

- **data/pitches/** - Pitch-level data (~117k pitches, April 2024)
- **data/plate_appearances/** - PA-level data (~30k PAs)
- **data/cache/** - CSV cache from Baseball Savant
