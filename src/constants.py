class Vocabulary:
    PAD = '<PAD>'
    START_GAME = '<START_GAME>'
    END_GAME = '<END_GAME>'
    START_INNING = '<START_INNING>'
    END_INNING = '<END_INNING>'
    
    SPECIAL = [PAD, START_GAME, END_GAME, START_INNING, END_INNING]
    STATE = [f'OUT{o}_BASE{b}' for o in range(3) for b in range(8)]
    OUTCOME = ['SINGLE', 'DOUBLE', 'TRIPLE', 'HOME_RUN', 'WALK', 'STRIKEOUT']
    
    ALL = SPECIAL + STATE + OUTCOME
    
    @staticmethod
    def format_state(outs, bases):
        return f"OUT{outs}_BASE{bases}"
    
class TrainingConstants:
    IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss default

# In constants.py or a new rules.py

def parse_state(token: str):
    """Parse 'OUT{o}_BASE{b}' -> (outs, bases) or None for special tokens."""
    if token.startswith('OUT') and '_BASE' in token:
        parts = token.split('_')
        outs = int(parts[0][3:])  # 'OUT0' -> 0
        bases = int(parts[1][4:])  # 'BASE3' -> 3
        return (outs, bases)
    return None
