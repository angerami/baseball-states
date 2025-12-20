"""Evaluation metrics for baseball sequence modeling."""

import numpy as np
import torch


def compute_sequence_metrics(eval_pred):
    """
    Compute accuracy metrics and perplexity for sequence prediction.
    
    Args:
        eval_pred: EvalPrediction object with predictions and label_ids
        
    Returns:
        Dictionary of metric names and values for logging
    """
    predictions, labels = eval_pred
    
    # Ensure we're working with numpy arrays
    predictions = predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else predictions
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    
    # predictions shape: (batch_size, seq_len, vocab_size)
    predicted_ids = np.argmax(predictions, axis=-1)
    
    # Mask out padding tokens (label == -100)
    mask = labels != -100
    
    # Top-1 exact match accuracy
    correct = (predicted_ids == labels) & mask
    accuracy = float(correct.sum() / mask.sum())
    
    # Top-3 accuracy
    top3_ids = np.argsort(predictions, axis=-1)[..., -3:]
    labels_expanded = np.expand_dims(labels, axis=-1)
    top3_correct = np.any(top3_ids == labels_expanded, axis=-1) & mask
    top3_accuracy = float(top3_correct.sum() / mask.sum())
    
    # Top-5 accuracy
    top5_ids = np.argsort(predictions, axis=-1)[..., -5:]
    top5_correct = np.any(top5_ids == labels_expanded, axis=-1) & mask
    top5_accuracy = float(top5_correct.sum() / mask.sum())
    
    # Compute perplexity from predictions
    log_probs = predictions - np.log(np.sum(np.exp(predictions), axis=-1, keepdims=True))

    # Get log prob of true labels (only for non-padding tokens)
    # We need to mask first, then gather log probs for valid labels
    batch_size, seq_len, vocab_size = predictions.shape

    # Create a safe copy of labels with -100 replaced by 0 (to avoid index errors)
    # We'll only use the values where mask is True anyway
    labels_safe = np.where(mask, labels, 0)
    true_log_probs = log_probs[np.arange(batch_size)[:, None], np.arange(seq_len), labels_safe]

    # Average over non-padding tokens
    loss = -true_log_probs[mask].mean()
    perplexity = float(np.exp(loss))
    
    return {
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "top5_accuracy": top5_accuracy,
        "perplexity": perplexity,
    }