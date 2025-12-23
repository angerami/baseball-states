"""Baseball game rules: transition legality, run scoring, and data filtering."""

import torch
from baseball_states.constants import Vocabulary
from baseball_states.utils import parse_state, count_runners

# --- Data filtering configuration ---

# Years to exclude from training (e.g., strike-shortened, COVID seasons)
BLACKLIST_YEARS = [
    # 2020,  # COVID shortened season - uncomment if needed
]

# MLB rule change: Runner on second in extra innings (started in 2020)
EXTRA_INNING_RUNNER_START_YEAR = 2020

# Regular season is innings 1-9, extras are 10+
REGULAR_INNING_LIMIT = 9


# --- Data filtering functions ---

def should_include_inning(inning: int, year: int = None, skip_extra_innings: bool = False,
                          filter_special_extra_innings: bool = True) -> bool:
    """Determine if an inning should be included in training data.

    Args:
        inning: Inning number (1-based)
        year: Year of the game (optional, for rule-specific filtering)
        skip_extra_innings: If True, exclude all innings > 9
        filter_special_extra_innings: If True, exclude extra innings with special rules
                                      (e.g., runner on 2nd starting in 2020)

    Returns:
        bool: True if inning should be included
    """
    # Skip extra innings entirely if requested
    if skip_extra_innings and inning > REGULAR_INNING_LIMIT:
        return False

    # Filter extra innings with special rules
    if filter_special_extra_innings and inning > REGULAR_INNING_LIMIT:
        if year is not None and year >= EXTRA_INNING_RUNNER_START_YEAR:
            return False

    return True


def should_include_year(year: int) -> bool:
    """Determine if a year should be included in training data.

    Args:
        year: Year of the game

    Returns:
        bool: True if year should be included
    """
    return year not in BLACKLIST_YEARS


def is_valid_inning_start(outs: int, bases: int) -> bool:
    """Check if a state is a valid start for an inning.

    Valid starts are:
    - 0 outs, nobody on base (bases=0)

    Args:
        outs: Number of outs (0-2)
        bases: Base state code (0-7)

    Returns:
        bool: True if this is a valid inning start
    """
    return outs == 0 and bases == 0


# --- Forbidden transitions ---

def is_forbidden_transition(from_token: str, to_token: str) -> bool:
    """Return True if transition violates baseball rules."""
    from_state = parse_state(from_token)
    to_state = parse_state(to_token)
    
    # Special token rules
    if from_token == Vocabulary.START_INNING:
        return to_state is None or to_state[0] != 0
    
    if from_token == Vocabulary.END_INNING:
        return True  # END_INNING shouldn't predict anything
    
    # State-to-state rules
    if from_state is not None and to_state is not None:
        from_outs, _ = from_state
        to_outs, _ = to_state
        if to_outs < from_outs:
            return True
    
    # State to END_INNING is always legal
    if from_state is not None and to_token == Vocabulary.END_INNING:
        return False
    
    return False


# --- Matrix builders ---

_forbidden_matrix = None
_runs_matrix = None


def get_forbidden_matrix(tokenizer):
    """Get or build the forbidden transition matrix (cached)."""
    global _forbidden_matrix
    if _forbidden_matrix is None:
        _forbidden_matrix = build_forbidden_matrix(tokenizer)
    return _forbidden_matrix


def build_forbidden_matrix(tokenizer) -> torch.Tensor:
    """Precompute (vocab, vocab) matrix where 1 = forbidden transition."""
    vocab_size = len(tokenizer)
    forbidden = torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
    
    for from_id in range(vocab_size):
        from_token = tokenizer._convert_id_to_token(from_id)
        for to_id in range(vocab_size):
            to_token = tokenizer._convert_id_to_token(to_id)
            if is_forbidden_transition(from_token, to_token):
                forbidden[from_id, to_id] = 1.0
    
    return forbidden


def get_runs_matrix(tokenizer):
    """Get or build the runs scored matrix (cached)."""
    global _runs_matrix
    if _runs_matrix is None:
        _runs_matrix = build_runs_matrix(tokenizer)
    return _runs_matrix


def build_runs_matrix(tokenizer) -> torch.Tensor:
    """
    Precompute (vocab, vocab) matrix of runs scored per transition.
    
    Runs scored = runners_left + batter_scored - runners_now - outs_added
    where batter_scored = 1 if batter scored (e.g., home run)
    
    This is derived from conservation: 
        runners_before + 1 (batter) = runners_after + outs_added + runs_scored
    So:
        runs_scored = runners_before + 1 - runners_after - outs_added
    """
    vocab_size = len(tokenizer)
    runs = torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
    
    for from_id in range(vocab_size):
        from_token = tokenizer._convert_id_to_token(from_id)
        from_state = parse_state(from_token)
        if from_state is None:
            continue
            
        from_outs, from_bases = from_state
        runners_before = count_runners(from_bases)
        
        for to_id in range(vocab_size):
            to_token = tokenizer._convert_id_to_token(to_id)
            to_state = parse_state(to_token)
            
            if to_state is None:
                # Transition to END_INNING: 3rd out made, no runs score
                if to_token == Vocabulary.END_INNING:
                    runs[from_id, to_id] = 0
                continue
            
            to_outs, to_bases = to_state
            runners_after = count_runners(to_bases)
            outs_added = to_outs - from_outs
            
            # Conservation: runners_before + 1 = runners_after + outs_added + runs_scored
            runs_scored = runners_before + 1 - runners_after - outs_added
            runs[from_id, to_id] = max(0, runs_scored)  # clamp to non-negative
    
    return runs


# --- Sequence scoring ---

def compute_runs_scored(sequences: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Compute total runs scored for each sequence.
    
    Args:
        sequences: (batch, seq_len) token IDs
        tokenizer: GameStateTokenizer instance
        
    Returns:
        (batch,) tensor of runs scored per sequence
    """
    runs_matrix = get_runs_matrix(tokenizer).to(sequences.device)
    
    # Get transition pairs: from_ids[t] -> to_ids[t]
    from_ids = sequences[:, :-1]  # (batch, seq_len - 1)
    to_ids = sequences[:, 1:]     # (batch, seq_len - 1)
    
    # Look up runs for each transition
    runs_per_transition = runs_matrix[from_ids, to_ids]  # (batch, seq_len - 1)
    
    # Sum over sequence
    total_runs = runs_per_transition.sum(dim=-1)  # (batch,)
    
    return total_runs

