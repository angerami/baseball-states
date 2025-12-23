# baseball_states/preprocessor.py
import pandas as pd
from baseball_states.constants import Vocabulary

# Module defines map functions used to transform raw download into
# format usable by dataset.py

def _detect_data_format(df):
    """Detect whether data is Statcast or Retrosheet format.

    Args:
        df: DataFrame with raw data

    Returns:
        str: 'statcast' or 'retrosheet'
    """
    # Statcast has game_pk, Retrosheet has gid
    if 'game_pk' in df.columns:
        return 'statcast'
    elif 'gid' in df.columns:
        return 'retrosheet'
    else:
        raise ValueError("Unable to detect data format. Expected 'game_pk' (Statcast) or 'gid' (Retrosheet) column.")

def _preprocess_statcast_to_pa(df):
    """Convert Statcast pitch-level data to plate appearance records.

    Args:
        df: DataFrame with Statcast columns (game_pk, at_bat_number, pitch_number, etc.)

    Returns:
        DataFrame with standardized PA records
    """
    grouped = df.groupby(['game_pk', 'at_bat_number'], sort=False)

    pa_records = []
    for (game_pk, at_bat_number), pitches in grouped:
        pitches = pitches.sort_values('pitch_number') if 'pitch_number' in pitches.columns else pitches
        first = pitches.iloc[0]
        last = pitches.iloc[-1]

        # Encode initial state
        outs = int(first['outs_when_up']) if pd.notna(first['outs_when_up']) else 0
        on_1b = 1 if pd.notna(first.get('on_1b')) else 0
        on_2b = 2 if pd.notna(first.get('on_2b')) else 0
        on_3b = 4 if pd.notna(first.get('on_3b')) else 0
        bases = on_1b + on_2b + on_3b

        pa_records.append({
            'game_pk': game_pk,
            'at_bat_number': at_bat_number,
            'outs': outs,
            'bases': bases,
            'outcome': last['events'] if pd.notna(last['events']) else None,
            'batter': first.get('batter'),
            'pitcher': first.get('pitcher'),
            'inning': first.get('inning'),
            'inning_topbot': first.get('inning_topbot'),
            'game_year': first.get('game_year'),
            'game_type': first.get('game_type'),
        })

    result_df = pd.DataFrame(pa_records)
    result_df = result_df.sort_values(['game_pk', 'inning', 'inning_topbot', 'at_bat_number'])
    return result_df

def _preprocess_retrosheet_to_pa(df):
    """Convert Retrosheet PA-level data to standardized plate appearance records.

    Args:
        df: DataFrame with Retrosheet columns (gid, inning, top_bot, outs_pre, br1_pre, etc.)

    Returns:
        DataFrame with standardized PA records (matching Statcast output format)
    """
    pa_records = []

    for idx, row in df.iterrows():
        # Encode initial state from Retrosheet columns
        outs = int(row['outs_pre']) if pd.notna(row['outs_pre']) else 0

        # Retrosheet uses br1_pre, br2_pre, br3_pre (runner IDs or NaN)
        on_1b = 1 if pd.notna(row.get('br1_pre')) else 0
        on_2b = 2 if pd.notna(row.get('br2_pre')) else 0
        on_3b = 4 if pd.notna(row.get('br3_pre')) else 0
        bases = on_1b + on_2b + on_3b

        # Determine outcome from Retrosheet event columns
        # Retrosheet has: single, double, triple, hr, walk, k, roe, fc, sf, sh, othout
        outcome = None
        if pd.notna(row.get('single')) and row['single'] == 1:
            outcome = 'single'
        elif pd.notna(row.get('double')) and row['double'] == 1:
            outcome = 'double'
        elif pd.notna(row.get('triple')) and row['triple'] == 1:
            outcome = 'triple'
        elif pd.notna(row.get('hr')) and row['hr'] == 1:
            outcome = 'home_run'
        elif pd.notna(row.get('walk')) and row['walk'] == 1:
            outcome = 'walk'
        elif pd.notna(row.get('k')) and row['k'] == 1:
            outcome = 'strikeout'
        elif pd.notna(row.get('roe')) and row['roe'] == 1:
            outcome = 'field_error'
        elif pd.notna(row.get('fc')) and row['fc'] == 1:
            outcome = 'field_out'  # fielder's choice
        elif pd.notna(row.get('sf')) and row['sf'] == 1:
            outcome = 'sac_fly'
        elif pd.notna(row.get('sh')) and row['sh'] == 1:
            outcome = 'sac_bunt'
        elif pd.notna(row.get('othout')) and row['othout'] == 1:
            outcome = 'field_out'

        # Extract year from date (YYYYMMDD format)
        game_year = None
        if pd.notna(row.get('date')):
            date_str = str(row['date'])
            if len(date_str) >= 4:
                try:
                    game_year = int(date_str[:4])
                except ValueError:
                    pass

        # Map top_bot to inning_topbot format ('Top'/'Bot')
        inning_topbot = None
        if pd.notna(row.get('top_bot')):
            inning_topbot = 'Top' if row['top_bot'] == 1 else 'Bot'

        pa_records.append({
            'game_pk': row.get('gid'),  # Use gid as game identifier
            'at_bat_number': idx,  # Retrosheet doesn't have this; use row index
            'outs': outs,
            'bases': bases,
            'outcome': outcome,
            'batter': None,  # Not available in skimmed Retrosheet data
            'pitcher': None,  # Not available in skimmed Retrosheet data
            'inning': row.get('inning'),
            'inning_topbot': inning_topbot,
            'game_year': game_year,
            'game_type': 'R',  # Retrosheet data is typically regular season
        })

    result_df = pd.DataFrame(pa_records)
    # Sort by game and inning to ensure chronological order
    result_df = result_df.sort_values(['game_pk', 'inning', 'inning_topbot', 'at_bat_number'])
    return result_df

def preprocess_to_pa_map(examples, data_format='auto'):
    """Map function: Group contiguous pitches into plate appearances.

    For use with datasets.Dataset.map(batched=True).

    Supports both Statcast (pitch-level) and Retrosheet (PA-level) formats.

    Args:
        examples: Batch of examples (pitches for Statcast, PAs for Retrosheet)
        data_format: 'statcast', 'retrosheet', or 'auto' (default: auto-detect)

    Note: Raw Statcast data comes in reverse chronological order, so we sort by
    game_pk and at_bat_number to get chronological ordering.
    Retrosheet data is already at PA level, so it just gets passed through with format conversion.
    """
    # Convert LazyBatch to dict if necessary
    if not isinstance(examples, dict):
        examples = dict(examples)

    df = pd.DataFrame(examples)

    # Auto-detect format if needed
    if data_format == 'auto':
        data_format = _detect_data_format(df)

    # Delegate to appropriate helper
    if data_format == 'statcast':
        result_df = _preprocess_statcast_to_pa(df)
    elif data_format == 'retrosheet':
        # Retrosheet is already at PA level, just convert format
        result_df = _preprocess_retrosheet_to_pa(df)
    else:
        raise ValueError(f"Unknown data_format: {data_format}. Expected 'statcast', 'retrosheet', or 'auto'.")

    return result_df.to_dict('list')

# Encode `pa` as `sequence` of vocabulary elements
def pa_to_sequence_map(examples, inning_level=True, skip_extra_innings=False,
                       filter_special_extra_innings=True, filter_invalid_starts=True,
                       filter_game_types=True):
    """Map function: Convert contiguous plate appearances into token sequences.

    For use with datasets.Dataset.map(batched=True).

    Assumes plate appearances are in chronological order (fixed in preprocess_to_pa_map).

    Args:
        examples: Batch of plate appearances
        inning_level: Create sequences per inning (True) or per game (False)
        skip_extra_innings: Skip all innings > 9
        filter_special_extra_innings: Skip extra innings with special rules (runner on 2nd)
        filter_invalid_starts: Skip innings that don't start with valid state
        filter_game_types: Skip non-regular season games (playoffs, etc.)
    """
    from baseball_states.game_rules import should_include_inning, should_include_year, is_valid_inning_start

    sequences = []
    current_sequence = []
    current_group = None
    should_skip_current_group = False
    first_pa_of_group = None

    n = len(examples['game_pk'])
    for i in range(n):
        if inning_level:
            group_key = (
                examples['game_pk'][i],
                examples['inning'][i],
                examples['inning_topbot'][i]
            )
        else:
            group_key = examples['game_pk'][i]

        if group_key != current_group:
            # Save previous sequence if valid
            if current_sequence and not should_skip_current_group:
                current_sequence.append(Vocabulary.END_INNING if inning_level else Vocabulary.END_GAME)
                sequences.append(current_sequence)

            current_sequence = [Vocabulary.START_INNING if inning_level else Vocabulary.START_GAME]
            current_group = group_key
            first_pa_of_group = i
            should_skip_current_group = False

            # Apply filters
            year = examples.get('game_year', [None] * n)[i]
            game_type = examples.get('game_type', [None] * n)[i]
            inning = examples['inning'][i]
            outs = examples['outs'][i]
            bases = examples['bases'][i]

            # Filter by year blacklist
            if year is not None and not should_include_year(year):
                should_skip_current_group = True
                continue

            # Filter by game type (skip non-regular season)
            if filter_game_types and game_type is not None:
                # Regular season games are typically 'R', playoffs are 'P', etc.
                if game_type != 'R':
                    should_skip_current_group = True
                    continue

            # Filter by inning rules
            if not should_include_inning(inning, year, skip_extra_innings, filter_special_extra_innings):
                should_skip_current_group = True
                continue

            # Filter by valid starting state
            if filter_invalid_starts and not is_valid_inning_start(outs, bases):
                should_skip_current_group = True
                continue

        if should_skip_current_group:
            continue

        token = Vocabulary.format_state(examples['outs'][i], examples['bases'][i])
        current_sequence.append(token)

    # Save final sequence if valid
    if current_sequence and not should_skip_current_group:
        current_sequence.append(Vocabulary.END_INNING if inning_level else Vocabulary.END_GAME)
        sequences.append(current_sequence)

    return {
        'sequence': sequences,
        'length': [len(seq) for seq in sequences]
    }

# Tokenize `sequence` producing `tokens`
def sequence_to_tokens_map(examples, tokenizer):
    """Map function: Tokenize 'sequence' into 'tokens'.

    For use with datasets.Dataset.map(batched=True).
    """
    token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in examples['sequence']]
    return {'tokens': token_ids, 'length': [len(ids) for ids in token_ids]}