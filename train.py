from dataclasses import dataclass
from pathlib import Path
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_from_disk
from baseball_states.tokenizer import GameStateTokenizer
from baseball_states.dataset import GameSequenceDataset
import numpy as np


@dataclass
class ModelConfig:
    # Model architecture
    n_embd: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_positions: int = 32
    
    # Training
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 5e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Data
    data_path: str = "data/tokens_inning"
    val_split: float = 0.1
    max_length: int = 32
    
    # Checkpointing
    output_dir: str = "checkpoints/baseball_gpt2"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    
    # Other
    seed: int = 42


def compute_metrics(eval_pred):
    """Compute perplexity and token accuracy"""
    predictions, labels = eval_pred
    
    # predictions shape: (batch_size, seq_len, vocab_size)
    # Get predicted token IDs
    predicted_ids = np.argmax(predictions, axis=-1)
    
    # Mask out padding tokens (label == -100)
    mask = labels != -100
    
    # Token accuracy (excluding padding)
    correct = (predicted_ids == labels) & mask
    accuracy = correct.sum() / mask.sum()
    
    # Perplexity (from loss, computed by trainer)
    # We'll let the trainer compute loss, then calculate perplexity from it
    
    return {
        "accuracy": accuracy,
    }


def train_model(config: ModelConfig):
    """Main training function"""
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Load tokenizer
    tokenizer = GameStateTokenizer()
    
    # Load and split dataset
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
    
    # Initialize model
    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        loss_type="ForCausalLMLoss",
    )
    
    model = GPT2LMHeadModel(model_config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=config.seed,
        dataloader_pin_memory=False,
        report_to=["tensorboard"],
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
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


if __name__ == "__main__":
    config = ModelConfig()
    model, tokenizer = train_model(config)