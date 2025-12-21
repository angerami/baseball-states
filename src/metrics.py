"""Evaluation metrics for baseball sequence modeling."""

import numpy as np
import torch
import torch.nn.functional as F

from baseball_states.game_rules import get_forbidden_matrix

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
    runs_scored = compute_runs_scored(predictions, tokenizer)
    return {
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "top5_accuracy": top5_accuracy,
        "perplexity": perplexity,
        "illegal_prob": illegal_prob,
        "runs_scored": runs_scored,
    }