"""Evaluation metrics for baseball sequence modeling."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from collections import Counter
from scipy.spatial.distance import jensenshannon

from baseball_states.game_rules import get_forbidden_matrix, compute_runs_scored


# ============================================================================
# Training/Eval Metrics
# ============================================================================

def compute_illegal_probability(predictions, input_ids, tokenizer):
    """
    Compute mean probability mass on forbidden transitions.

    Args:
        predictions: (batch, seq, vocab) logits
        input_ids: (batch, seq) current state token IDs
        tokenizer: GameStateTokenizer instance

    Returns:
        float: mean probability assigned to illegal next states
    """
    forbidden_matrix = get_forbidden_matrix(tokenizer)
    device = predictions.device if torch.is_tensor(predictions) else 'cpu'
    forbidden_matrix = forbidden_matrix.to(device)

    # Convert to tensor if numpy
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(input_ids, np.ndarray):
        input_ids = torch.from_numpy(input_ids)

    probs = F.softmax(predictions, dim=-1)  # (batch, seq, vocab)
    mask = forbidden_matrix[input_ids]  # (batch, seq, vocab)

    illegal_mass = (probs * mask).sum(dim=-1)  # (batch, seq)
    return illegal_mass.mean().item()


def compute_sequence_metrics(eval_pred, tokenizer=None):
    """
    Compute accuracy metrics, perplexity, and illegal transition probability.

    Args:
        eval_pred: EvalPrediction object with predictions and label_ids
        tokenizer: GameStateTokenizer (optional, will create if needed)

    Returns:
        Dictionary of metric names and values for logging
    """
    predictions, labels = eval_pred

    # Get input_ids from labels (shifted back by 1)
    # labels[t] = input_ids[t+1], so input_ids[t] = labels[t-1]
    # For simplicity, we use labels shifted
    if torch.is_tensor(labels):
        input_ids = labels.clone()
    else:
        input_ids = np.copy(labels)

    # Ensure we're working with numpy arrays for accuracy metrics
    preds_np = predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else predictions
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels

    # predictions shape: (batch_size, seq_len, vocab_size)
    predicted_ids = np.argmax(preds_np, axis=-1)

    # Mask out padding tokens (label == -100)
    mask = labels_np != -100

    # Top-1 exact match accuracy
    correct = (predicted_ids == labels_np) & mask
    accuracy = float(correct.sum() / mask.sum())

    # Top-3 accuracy
    top3_ids = np.argsort(preds_np, axis=-1)[..., -3:]
    labels_expanded = np.expand_dims(labels_np, axis=-1)
    top3_correct = np.any(top3_ids == labels_expanded, axis=-1) & mask
    top3_accuracy = float(top3_correct.sum() / mask.sum())

    # Top-5 accuracy
    top5_ids = np.argsort(preds_np, axis=-1)[..., -5:]
    top5_correct = np.any(top5_ids == labels_expanded, axis=-1) & mask
    top5_accuracy = float(top5_correct.sum() / mask.sum())

    # Compute perplexity from predictions
    log_probs = preds_np - np.log(np.sum(np.exp(preds_np), axis=-1, keepdims=True))
    batch_size, seq_len, vocab_size = preds_np.shape
    labels_safe = np.where(mask, labels_np, 0)
    true_log_probs = log_probs[np.arange(batch_size)[:, None], np.arange(seq_len), labels_safe]
    loss = -true_log_probs[mask].mean()
    perplexity = float(np.exp(loss))

    # Compute illegal transition probability
    # Need tokenizer for this metric
    if tokenizer is None:
        from baseball_states.tokenizer import GameStateTokenizer
        tokenizer = GameStateTokenizer()

    # Get previous state by shifting labels
    input_ids_prev = np.zeros_like(labels_np)
    input_ids_prev[:, 1:] = labels_np[:, :-1]
    input_ids_prev[:, 0] = tokenizer.bos_token_id  # START_GAME or START_INNING
    # Replace -100 with PAD for indexing
    input_ids_prev = np.where(input_ids_prev == -100, tokenizer.pad_token_id, input_ids_prev)

    illegal_prob = compute_illegal_probability(predictions, input_ids_prev, tokenizer)

    # Compute runs scored - need to convert predicted_ids to tensor for compute_runs_scored
    predicted_ids_tensor = torch.from_numpy(predicted_ids) if isinstance(predicted_ids, np.ndarray) else predicted_ids
    runs_scored_tensor = compute_runs_scored(predicted_ids_tensor, tokenizer)
    runs_scored = float(runs_scored_tensor.mean().item())  # Convert to Python float

    return {
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "top5_accuracy": top5_accuracy,
        "perplexity": perplexity,
        "illegal_prob": illegal_prob,
        "runs_scored": runs_scored,
    }


# ============================================================================
# N-gram Analysis - Model-Direct Methods (no generation needed)
# ============================================================================

def get_model_conditional_probs(
    model: torch.nn.Module,
    tokenizer,
    n: int,
    device: str = "cpu"
) -> np.ndarray:
    """
    Query model for P(x_n | x_1, ..., x_{n-1}) for all possible histories.

    Args:
        model: Trained transformer model
        tokenizer: GameStateTokenizer instance
        n: n-gram order (1 for unigram, 2 for bigram, 3 for trigram, etc.)
        device: device to run model on

    Returns:
        Array of shape (vocab_size,)^n where arr[x_1,...,x_{n-1},x_n] = P(x_n | x_1,...,x_{n-1})
        For n=1: shape (vocab_size,) - P(x_1 | START)
        For n=2: shape (vocab_size, vocab_size) - P(x_2 | x_1)
        For n=3: shape (vocab_size, vocab_size, vocab_size) - P(x_3 | x_1, x_2)
    """
    model.eval()
    vocab_size = tokenizer.vocab_size
    start_token = tokenizer.bos_token_id

    shape = tuple([vocab_size] * n)
    probs = np.zeros(shape)

    # Generate all possible (n-1)-length histories
    import itertools
    if n == 1:
        # Special case: unigram is just P(x_1 | START)
        histories = [()]
    else:
        histories = list(itertools.product(range(vocab_size), repeat=n-1))

    with torch.no_grad():
        for history in histories:
            # Create input: [start_token, x_1, ..., x_{n-1}]
            input_ids = torch.tensor([[start_token] + list(history)], device=device)

            # Create attention mask (all ones since no padding)
            attention_mask = torch.ones_like(input_ids)

            # Get logits for next token
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1, :]  # Last position

            # Convert to probabilities
            probs_next = torch.softmax(logits, dim=0).cpu().numpy()

            # Store in array
            if n == 1:
                probs[:] = probs_next
            else:
                idx = history + (slice(None),)
                probs[idx] = probs_next

    return probs


def compute_ngram_counts(sequences: List[List[int]], n: int) -> Counter:
    """
    Compute n-gram counts from sequences.

    Args:
        sequences: List of token sequences
        n: n-gram order (1=unigram, 2=bigram, etc.)

    Returns:
        Counter mapping n-gram tuples to counts
    """
    counts = Counter()
    for seq in sequences:
        if len(seq) < n:
            continue
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i+n])
            counts[ngram] += 1
    return counts


def counts_to_conditional_prob(counts: Counter, n: int, vocab_size: int) -> np.ndarray:
    """
    Convert n-gram counts to conditional probability P(x_n | x_1, ..., x_{n-1}).

    Args:
        counts: n-gram counts
        n: n-gram order
        vocab_size: number of tokens

    Returns:
        Array of shape (vocab_size,)^n
        where arr[x_1, x_2, ..., x_{n-1}, x_n] = P(x_n | x_1, ..., x_{n-1})
    """
    shape = tuple([vocab_size] * n)
    probs = np.zeros(shape)

    # Get (n-1)-gram counts for normalization
    context_counts = Counter()
    for ngram, count in counts.items():
        context = ngram[:-1] if n > 1 else ()
        context_counts[context] += count

    # Fill conditional probabilities
    for ngram, count in counts.items():
        context = ngram[:-1] if n > 1 else ()
        if context_counts[context] > 0:
            probs[ngram] = count / context_counts[context]

    return probs


def compute_history_freq(ngram_counts: Counter, n: int, vocab_size: int) -> np.ndarray:
    """
    Get P(x_1...x_{n-1}) from n-gram counts.

    Returns: array of shape (vocab_size,)^(n-1)
    """
    history_counts = Counter()
    for ngram in ngram_counts:
        history = ngram[:-1] if n > 1 else ()
        history_counts[history] += ngram_counts[ngram]

    total = sum(history_counts.values())
    shape = tuple([vocab_size] * (n - 1)) if n > 1 else (1,)
    history_freq = np.zeros(shape)

    for history, count in history_counts.items():
        if n == 1:
            history_freq[0] = count / total
        else:
            history_freq[history] = count / total

    return history_freq


# ============================================================================
# Divergence Metrics
# ============================================================================

def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    KL(P || Q) = sum P(x) log(P(x) / Q(x))

    Args:
        p, q: Probability distributions (must have same shape and sum to 1)
        epsilon: Small value to avoid log(0)
    """
    p_safe = np.maximum(p, epsilon)
    q_safe = np.maximum(q, epsilon)
    return np.sum(p_safe * np.log(p_safe / q_safe))


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence: JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    """
    # Flatten for scipy
    p_flat = p.flatten()
    q_flat = q.flatten()
    return jensenshannon(p_flat, q_flat)


def compute_conditional_kl_per_history(
    p_data: np.ndarray,
    p_model: np.ndarray,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Compute KL(P_data(x_n | history) || P_model(x_n | history)) for each history.

    Args:
        p_data, p_model: Arrays of shape (vocab,)^n representing P(x_n | x_1,...,x_{n-1})

    Returns:
        Array of shape (vocab,)^(n-1) with KL divergence for each history
    """
    # Sum over last axis (x_n) to compute KL for each history
    p_data_safe = np.maximum(p_data, epsilon)
    p_model_safe = np.maximum(p_model, epsilon)

    kl_per_history = np.sum(p_data_safe * np.log(p_data_safe / p_model_safe), axis=-1)

    return kl_per_history


# ============================================================================
# High-Level Metric Computations
# ============================================================================

def compute_joint_distribution_from_model(
    model: torch.nn.Module,
    tokenizer,
    n: int,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute joint distribution P(x_1, ..., x_n) from model by chain rule.
    P(x_1, ..., x_n) = P(x_1) * P(x_2|x_1) * ... * P(x_n|x_1,...,x_{n-1})

    Args:
        model: Trained model
        tokenizer: GameStateTokenizer
        n: n-gram order
        device: device to run model on

    Returns:
        Joint probability array of shape (vocab_size,)^n
    """
    vocab_size = tokenizer.vocab_size

    # Get all conditional distributions
    conditionals = []
    for i in range(1, n + 1):
        p_i = get_model_conditional_probs(model, tokenizer, i, device)
        conditionals.append(p_i)

    # Build joint via chain rule
    if n == 1:
        return conditionals[0]
    elif n == 2:
        # P(x1, x2) = P(x1) * P(x2|x1)
        p_x1 = conditionals[0]  # shape (V,)
        p_x2_given_x1 = conditionals[1]  # shape (V, V)
        return p_x1[:, None] * p_x2_given_x1
    elif n == 3:
        # P(x1, x2, x3) = P(x1) * P(x2|x1) * P(x3|x1,x2)
        p_x1 = conditionals[0]  # shape (V,)
        p_x2_given_x1 = conditionals[1]  # shape (V, V)
        p_x3_given_x1x2 = conditionals[2]  # shape (V, V, V)
        return p_x1[:, None, None] * p_x2_given_x1[:, :, None] * p_x3_given_x1x2
    else:
        raise NotImplementedError(f"Joint distribution for n={n} not implemented")


def compute_joint_distribution_from_data(
    sequences: List[List[int]],
    n: int,
    vocab_size: int
) -> np.ndarray:
    """
    Compute empirical joint distribution P_data(x_1, ..., x_n) from sequences.

    Returns:
        Joint probability array of shape (vocab_size,)^n
    """
    counts = compute_ngram_counts(sequences, n)
    shape = tuple([vocab_size] * n)
    p_joint = np.zeros(shape)

    for ngram, count in counts.items():
        p_joint[ngram] = count

    p_joint /= p_joint.sum()
    return p_joint


def compute_conditional_divergence(
    data_sequences: List[List[int]],
    model: torch.nn.Module,
    tokenizer,
    n: int,
    device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """
    M3: Per-history conditional KL divergence comparing data to model.

    Args:
        data_sequences: Training sequences
        model: Trained model
        tokenizer: GameStateTokenizer
        n: n-gram order
        device: device to run model on

    Returns:
        Dict with:
            - "per_history": KL for each history, shape (vocab,)^(n-1)
            - "weighted_avg": scalar, weighted by P_data(history)
            - "p_data": conditional probs from data
            - "p_model": conditional probs from model
            - "history_freq": P_data(history)
    """
    vocab_size = tokenizer.vocab_size

    # Get data conditional probs
    counts_data = compute_ngram_counts(data_sequences, n=n)
    p_data = counts_to_conditional_prob(counts_data, n, vocab_size)

    # Get model conditional probs
    p_model = get_model_conditional_probs(model, tokenizer, n, device)

    # Compute per-history KL
    kl_per_history = compute_conditional_kl_per_history(p_data, p_model)

    # Compute history frequencies
    history_freq = compute_history_freq(counts_data, n, vocab_size)

    # Weighted average
    weighted_kl = np.sum(history_freq * kl_per_history)

    return {
        "per_history": kl_per_history,
        "weighted_avg": weighted_kl,
        "p_data": p_data,
        "p_model": p_model,
        "history_freq": history_freq
    }


# ============================================================================
# Markov Order Analysis
# ============================================================================

def conditional_entropy(p_conditional: np.ndarray, history_freq: np.ndarray) -> float:
    """
    H[X_n | X_1...X_{n-1}] = -∑ P(history) ∑ P(x_n|history) log P(x_n|history)

    Args:
        p_conditional: shape (V,)^n, P(x_n | x_1...x_{n-1})
        history_freq: shape (V,)^(n-1), P(x_1...x_{n-1})
    """
    epsilon = 1e-10
    p_safe = np.maximum(p_conditional, epsilon)

    # Entropy per history: -∑_{x_n} P(x_n|history) log P(x_n|history)
    entropy_per_history = -np.sum(p_conditional * np.log2(p_safe), axis=-1)

    # Weight by history frequency
    return np.sum(history_freq * entropy_per_history)


def compute_conditional_entropies(
    data_sequences: List[List[int]],
    vocab_size: int,
    max_n: int
) -> List[float]:
    """
    Compute H[X_n | X_1...X_{n-1}] for n=1 to max_n.

    Returns: list of entropies, one per n
    """
    entropies = []

    for n in range(1, max_n + 1):
        counts = compute_ngram_counts(data_sequences, n)
        p_cond = counts_to_conditional_prob(counts, n, vocab_size)
        history_freq = compute_history_freq(counts, n, vocab_size)

        H = conditional_entropy(p_cond, history_freq)
        entropies.append(H)

    return entropies
