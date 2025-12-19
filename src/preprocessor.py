# baseball_states/preprocessor.py
import pandas as pd
from baseball_states.constants import Vocabulary

# Module defines map functions used to transform raw download into 
# format usable by dataset.py 

def preprocess_to_pa_map(examples):
    """Map function: Group contiguous pitches into plate appearances.

    For use with datasets.Dataset.map(batched=True).
    """
    # Convert LazyBatch to dict if necessary
    if not isinstance(examples, dict):
        examples = dict(examples)

    df = pd.DataFrame(examples)
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
        })
    
    return pd.DataFrame(pa_records).to_dict('list')

# Encode `pa` as `sequence` of vocabulary elements
def pa_to_sequence_map(examples, inning_level=True):
    """Map function: Convert contiguous plate appearances into token sequences.
    
    For use with datasets.Dataset.map(batched=True).
    """
    sequences = []
    current_sequence = []
    current_group = None
    
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
            if current_sequence:
                current_sequence.append(Vocabulary.END_INNING if inning_level else Vocabulary.END_GAME)
                sequences.append(current_sequence)
            
            current_sequence = [Vocabulary.START_INNING if inning_level else Vocabulary.START_GAME]
            current_group = group_key
        
        token = Vocabulary.format_state(examples['outs'][i], examples['bases'][i])
        current_sequence.append(token)
    
    if current_sequence:
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