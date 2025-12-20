"""Utility functions for baseball state prediction"""
import torch
import time


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

def get_unique_name():
    return f"gpt2-train-{time.strftime('%Y-%m%d-%H%M%S')}"