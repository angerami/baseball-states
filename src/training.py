"""Training utilities for baseball state prediction model"""
from dataclasses import dataclass
from pathlib import Path
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_from_disk
import numpy as np

from baseball_states.tokenizer import GameStateTokenizer
from baseball_states.dataset import GameSequenceDataset
from baseball_states.metrics import compute_sequence_metrics
from baseball_states.utils import get_device, get_unique_name


@dataclass
class ModelConfig:
    """Configuration for model architecture and training"""
    # Model architecture
    n_embd: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_positions: int = 32

    # Training
    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 5e-4
    warmup_steps: int = 10
    weight_decay: float = 0.01

    # Data
    data_path: str = "data/tokens_inning"
    val_split: float = 0.1
    max_length: int = 32

    # Checkpointing
    output_dir: str = "checkpoints/baseball_gpt2"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10

    # Other
    seed: int = 42


def create_model(tokenizer: GameStateTokenizer, config: ModelConfig, device=None):
    """Create and initialize a GPT2 model for baseball state prediction

    Args:
        tokenizer: The GameStateTokenizer instance
        config: ModelConfig with architecture parameters
        device: Target device (if None, will auto-detect)

    Returns:
        model: Initialized GPT2LMHeadModel
    """
    if device is None:
        device = get_device()

    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        loss_type="ForCausalLMLoss"
    )

    model = GPT2LMHeadModel(model_config).to(device)

    # Count and display parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    return model


def load_datasets(config: ModelConfig):
    """Load and split dataset into train and eval sets

    Args:
        config: ModelConfig with data parameters

    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    hf_dataset = load_from_disk(config.data_path)
    split = hf_dataset.train_test_split(test_size=config.val_split, seed=config.seed)

    train_dataset = GameSequenceDataset(
        split['train'],
        max_length=config.max_length
    )
    eval_dataset = GameSequenceDataset(
        split['test'],
        max_length=config.max_length
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def create_trainer(model, tokenizer, train_dataset, eval_dataset, config: ModelConfig):
    """Create HuggingFace Trainer instance

    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: ModelConfig with training parameters

    Returns:
        trainer: Configured Trainer instance
    """
    unique_run_name = get_unique_name('gpt2-train')
    
    print(f"Run name: {unique_run_name}")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=unique_run_name,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_dir=f"{config.output_dir}/logs/{unique_run_name}",
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=config.seed,
        # Memory optimization settings
        dataloader_num_workers=0,  # Critical - workers can leak on Mac
        dataloader_pin_memory=False,
        gradient_accumulation_steps=1,  # Explicit
        eval_accumulation_steps=1,  # Don't accumulate eval predictions - key fix!
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_sequence_metrics
    )

    return trainer


def train_model(config: ModelConfig):
    """Main training function

    Args:
        config: ModelConfig with all training parameters

    Returns:
        tuple: (model, tokenizer) - trained model and tokenizer
    """
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = GameStateTokenizer()

    # Load datasets
    train_dataset, eval_dataset = load_datasets(config)

    # Create model
    model = create_model(tokenizer, config, device)

    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, config)

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model and tokenizer
    print(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Final evaluation
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
    print(f"Final perplexity: {np.exp(eval_results['eval_loss']):.4f}")
    print(f"Final accuracy: {eval_results['eval_accuracy']:.4f}")

    return model, tokenizer
