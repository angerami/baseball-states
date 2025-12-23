"""Utility functions for baseball state prediction"""
import numpy as np
import torch
import time
from typing import Literal, Tuple

FormatType = Literal["latex", "html", "unicode", "matplotlib", "markdown"]


def get_device():
    """Get the best available device for training/inference

    Returns:
        torch.device: CUDA GPU (Google Colab), MPS (Apple Silicon), or CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")      # NVIDIA GPU (Google Colab)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")       # Apple Silicon GPU (Your Mac)
    else:
        device = torch.device("cpu")       # Default fallback
    return device


def get_unique_name(run_name):
    return f"{run_name}-{time.strftime('%Y-%m%d-%H%M%S')}"


def decode_base_state(base_code):
    """Convert base code (0-7) to runner configuration string.

    Args:
        base_code: Integer 0-7 representing binary encoding of runners
                  bit 0 = 1st base, bit 1 = 2nd base, bit 2 = 3rd base

    Returns:
        String representation like "1--", "-2-", "123", "---"
    """
    if base_code == 0:
        return "---"

    bases = []
    if base_code & 1:  # bit 0: first base
        bases.append("1")
    else:
        bases.append("-")

    if base_code & 2:  # bit 1: second base
        bases.append("2")
    else:
        bases.append("-")

    if base_code & 4:  # bit 2: third base
        bases.append("3")
    else:
        bases.append("-")

    return "".join(bases)


def format_token_display(token_str, use_symbols=False):
    """Format a token string for human-readable display.

    Args:
        token_str: Token like "OUT0_BASE0", "OUT2_BASE7", "<START_INNING>", etc.
        use_symbols: If True, use diamond/circle symbols for visual representation

    Returns:
        Formatted string like "0 out, ---", "2 out, 123", "<START_INNING>"
        Or with symbols: "○○○ ◇◇◇" (if use_symbols=True)
    """
    # Special tokens - return as is
    if token_str.startswith("<") and token_str.endswith(">"):
        return token_str

    # Outcome tokens - return as is
    if token_str in ['SINGLE', 'DOUBLE', 'TRIPLE', 'HOME_RUN', 'WALK', 'STRIKEOUT']:
        return token_str

    # State tokens - parse and format
    if token_str.startswith("OUT") and "_BASE" in token_str:
        try:
            parts = token_str.split("_BASE")
            outs = int(parts[0].replace("OUT", ""))
            base_code = int(parts[1])

            if use_symbols:
                return format_state_symbols(outs, base_code)
            else:
                bases = decode_base_state(base_code)
                out_str = "out" if outs == 1 else "outs"
                return f"{outs} {out_str}, {bases}"
        except (ValueError, IndexError):
            return token_str

    # Fallback - return original
    return token_str


def format_state_symbols(outs, base_code):
    """Format game state using symbols.

    Args:
        outs: Number of outs (0-2)
        base_code: Base state code (0-7)

    Returns:
        String like "○○○ ◇◇◇" (0 outs, empty) or "●●○ ◇◆◇" (2 outs, runner on 2nd)

    Symbol key:
        ● = out recorded
        ○ = no out
        ◆ = runner on base
        ◇ = base empty
    """
    # Format outs: ● for recorded out, ○ for not out
    out_symbols = '●' * outs + '○' * (3 - outs)

    # Format bases: ◆ for runner, ◇ for empty
    # Base order: 1st (bit 0), 2nd (bit 1), 3rd (bit 2)
    first_base = '◆' if (base_code & 1) else '◇'
    second_base = '◆' if (base_code & 2) else '◇'
    third_base = '◆' if (base_code & 4) else '◇'

    return f"{out_symbols} {first_base}{second_base}{third_base}"


def parse_state(token: str):
    """Parse 'OUT{o}_BASE{b}' -> (outs, bases) or None for special tokens.

    Args:
        token: Token string like "OUT0_BASE3" or "<START_INNING>"

    Returns:
        tuple: (outs, bases) where outs is 0-2 and bases is 0-7, or None for special tokens
    """
    if token.startswith('OUT') and '_BASE' in token:
        parts = token.split('_')
        outs = int(parts[0][3:])  # 'OUT0' -> 0
        bases = int(parts[1][4:])  # 'BASE3' -> 3
        return (outs, bases)
    return None


def count_runners(base_code):
    """Count number of runners on base from base code.

    Args:
        base_code: Integer 0-7 representing binary encoding of runners
                  bit 0 = 1st base, bit 1 = 2nd base, bit 2 = 3rd base

    Returns:
        int: Number of runners (0-3)

    Examples:
        >>> count_runners(0)  # ---
        0
        >>> count_runners(1)  # 1--
        1
        >>> count_runners(7)  # 123
        3
        >>> count_runners(5)  # 1-3
        2
    """
    count = 0
    if base_code & 1:  # bit 0: first base
        count += 1
    if base_code & 2:  # bit 1: second base
        count += 1
    if base_code & 4:  # bit 2: third base
        count += 1
    return count


# ============================================================================
# Visualization helpers for n-gram analysis
# ============================================================================

def flatten_conditional_for_heatmap(p_conditional: np.ndarray) -> np.ndarray:
    """
    Flatten P(x_n | x_1, ..., x_{n-1}) from shape (V,)^n to (V^(n-1), V).

    Rows: all possible (n-1)-length histories
    Columns: x_n (next token)

    Args:
        p_conditional: Array of shape (vocab,)^n

    Returns:
        Array of shape (vocab^(n-1), vocab)
    """
    n = len(p_conditional.shape)
    vocab_size = p_conditional.shape[0]

    # Reshape: move last axis to end, flatten others
    # (V, V, ..., V, V) -> (V^(n-1), V)
    num_histories = vocab_size ** (n - 1)
    return p_conditional.reshape(num_histories, vocab_size)


def unflatten_history_index(flat_idx: int, n: int, vocab_size: int) -> Tuple[int, ...]:
    """
    Convert flat history index back to (x_1, ..., x_{n-1}) tuple.

    Example: flat_idx=5, n=3, vocab=26 -> (0, 5) meaning "history (token_0, token_5)"
    """
    history = []
    for _ in range(n - 1):
        history.append(flat_idx % vocab_size)
        flat_idx //= vocab_size
    return tuple(reversed(history))
