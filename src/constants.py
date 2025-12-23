class Vocabulary:
    PAD = '<PAD>'
    START_GAME = '<START_GAME>'
    END_GAME = '<END_GAME>'
    START_INNING = '<START_INNING>'
    END_INNING = '<END_INNING>'
    
    STATE = [f'OUT{o}_BASE{b}' for o in range(3) for b in range(8)]
    SPECIAL = [PAD, START_GAME, END_GAME, START_INNING, END_INNING]
    OUTCOME = []
    # OUTCOME = ['SINGLE', 'DOUBLE', 'TRIPLE', 'HOME_RUN', 'WALK', 'STRIKEOUT']
    # + OUTCOME
    
    ALL = SPECIAL + STATE
    
    @staticmethod
    def format_state(outs, bases):
        return f"OUT{outs}_BASE{bases}"
    
class TrainingConstants:
    IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss default
