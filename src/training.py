"""Training utilities for baseball state prediction model"""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, TrainerCallback
from datasets import load_from_disk
import numpy as np
import time
from datetime import timedelta
import json
import yaml

from baseball_states.tokenizer import GameStateTokenizer
from baseball_states.dataset import GameSequenceDataset, PackingCollator, collate_fn
from baseball_states.metrics import (
    compute_sequence_metrics,
    compute_ngram_counts,
    compute_joint_distribution_from_data,
    compute_joint_distribution_from_model,
    compute_js_divergence,
    compute_conditional_divergence,
    compute_conditional_entropies,
)
from baseball_states.utils import get_device, get_unique_name
from functools import partial


class TimingCallback(TrainerCallback):
    """Callback to track and display training time and ETA"""

    def __init__(self):
        self.start_time = None
        self.step_times = []
        self.last_log_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the start of training"""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        print(f"\n{'='*60}")
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        current_time = time.time()
        self.step_times.append(current_time)

        # Keep only recent steps for more accurate ETA (last 50 steps)
        if len(self.step_times) > 50:
            self.step_times = self.step_times[-50:]

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if self.start_time is None:
            return

        current_time = time.time()
        elapsed = current_time - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # Calculate ETA based on recent step times
        if len(self.step_times) >= 2 and state.max_steps > 0:
            recent_times = self.step_times[-min(50, len(self.step_times)):]
            avg_step_time = (recent_times[-1] - recent_times[0]) / (len(recent_times) - 1)
            remaining_steps = state.max_steps - state.global_step
            eta_seconds = avg_step_time * remaining_steps
            eta_str = str(timedelta(seconds=int(eta_seconds)))

            # Calculate progress percentage
            progress = (state.global_step / state.max_steps) * 100

            print(f"\n[Step {state.global_step}/{state.max_steps} ({progress:.1f}%)] "
                  f"Elapsed: {elapsed_str} | ETA: {eta_str}")
        else:
            print(f"\n[Step {state.global_step}] Elapsed: {elapsed_str}")

        self.last_log_time = current_time

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        if self.start_time is None:
            return

        total_time = time.time() - self.start_time
        total_str = str(timedelta(seconds=int(total_time)))

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Total time: {total_str}")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")


@dataclass
class ModelConfig:
    """Configuration for model architecture and training

    Attributes:
        Model architecture:
            n_embd: Embedding dimension
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            n_positions: Maximum sequence length

        Training:
            batch_size: Batch size for training and eval
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization

        Data:
            data_path: Path to tokenized dataset
            val_split: Fraction of data to use for validation
            max_length: Maximum sequence length
            train_fraction: Fraction of training data to use (for testing)
            use_packing: If True, use PackingCollator for efficient GPU usage

        Checkpointing:
            output_dir: Directory to save checkpoints and final model
            save_steps: Save checkpoint every N steps
            eval_steps: Run evaluation every N steps
            logging_steps: Log metrics every N steps
            save_initial_checkpoint: If True, save initial untrained model
                                    to {output_dir}/checkpoint-initial (default: True)
            resume_from_checkpoint: Resume training from checkpoint
                                  - "auto": Auto-detect latest checkpoint in output_dir
                                  - True: Same as "auto"
                                  - False/None: Start training from scratch
                                  - str: Path to specific checkpoint directory

        Postprocessing:
            run_postanalysis: If True, run n-gram analysis after training completes
                             Results saved to parallel analysis_output directory (default: False)
            postanalysis_max_n: Maximum n-gram order for analysis (default: 6)

        Other:
            seed: Random seed for reproducibility
    """
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
    train_fraction: float = 1.0
    use_packing: bool = False

    # Checkpointing
    output_dir: str = "checkpoints/baseball_gpt2"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    save_initial_checkpoint: bool = True
    resume_from_checkpoint: str | bool | None = "auto"  # "auto", True, False, or path to checkpoint

    # Postprocessing
    run_postanalysis: bool = False
    postanalysis_max_n: int = 6

    # Other
    seed: int = 42

    @classmethod
    def from_file(cls, path: str) -> 'ModelConfig':
        """Load configuration from a JSON or YAML file

        Args:
            path: Path to config file (.json or .yaml/.yml)

        Returns:
            ModelConfig instance with values from file

        Example:
            config = ModelConfig.from_file('configs/base_config.yaml')
            # Override specific values for this run
            config.num_epochs = 10
            config.output_dir = f'checkpoints/run-{RUN_NUMBER}'
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # Read file based on extension
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}. Use .json, .yaml, or .yml")

        # Filter to only valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        # Warn about ignored fields
        ignored = set(config_dict.keys()) - valid_fields
        if ignored:
            print(f"Warning: Ignoring unrecognized config fields: {ignored}")

        return cls(**filtered_dict)

    def to_file(self, path: str):
        """Save configuration to a JSON or YAML file

        Args:
            path: Path to save config file (.json or .yaml/.yml)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self)

        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}. Use .json, .yaml, or .yml")

        print(f"Config saved to {path}")

    def __repr__(self):
        """Pretty print configuration"""
        lines = ["ModelConfig("]
        for field in fields(self):
            value = getattr(self, field.name)
            lines.append(f"  {field.name}={value!r},")
        lines.append(")")
        return "\n".join(lines)


def get_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint in output_dir

    Args:
        output_dir: Directory to search for checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    # Find all checkpoint directories (format: checkpoint-{step})
    checkpoints = [
        d for d in output_path.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-") and d.name != "checkpoint-initial"
    ]

    if not checkpoints:
        return None

    # Extract step numbers and find the latest
    def get_step(checkpoint_dir):
        try:
            return int(checkpoint_dir.name.split("-")[-1])
        except (ValueError, IndexError):
            return -1

    latest = max(checkpoints, key=get_step)
    return str(latest)


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

    # Optionally use only a fraction of training data (for testing)
    if config.train_fraction < 1.0:
        train_size = int(len(split['train']) * config.train_fraction)
        split['train'] = split['train'].select(range(train_size))
        print(f"Using {config.train_fraction:.1%} of training data ({train_size} samples)")

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

    # Choose collator based on packing configuration
    if config.use_packing:
        data_collator = PackingCollator(tokenizer=tokenizer, max_length=config.max_length)
    else:
        data_collator = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    # Create trainer with timing callback
    timing_callback = TimingCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_sequence_metrics,
        data_collator=data_collator,
        callbacks=[timing_callback],
    )

    return trainer


def run_postprocessing_analysis(model, tokenizer, config: ModelConfig, device=None):
    """Run n-gram analysis on trained model and save results

    Args:
        model: Trained model
        tokenizer: Tokenizer
        config: ModelConfig with data and output paths
        device: Device to run analysis on (default: auto-detect)
    """
    if device is None:
        device = get_device()

    print("\n" + "="*60)
    print("Running post-training n-gram analysis...")
    print("="*60 + "\n")

    # Determine output directory (parallel to output_dir)
    output_path = Path(config.output_dir)
    analysis_dir = output_path.parent / f"analysis_{output_path.name}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analysis output directory: {analysis_dir}")

    # Load training data sequences
    from datasets import load_from_disk
    print(f"Loading dataset from {config.data_path}...")
    dataset = load_from_disk(config.data_path)

    # Get sequences - handle both Dataset and DatasetDict
    # Try different column names: tokens, input_ids, token_ids
    sequences = None

    if hasattr(dataset, 'keys') and 'train' in dataset:
        # It's a DatasetDict
        ds = dataset['train']
    else:
        # It's a Dataset
        ds = dataset

    # Try different column names
    for col_name in ['tokens', 'input_ids', 'token_ids']:
        if col_name in ds.column_names:
            sequences = ds[col_name]
            print(f"Using column: {col_name}")
            break

    if sequences is None:
        print(f"Warning: Could not find sequence column. Available columns: {ds.column_names}")
        print("Skipping analysis")
        return

    print(f"Loaded {len(sequences)} sequences")
    vocab_size = tokenizer.vocab_size

    # Compute basic divergence metrics
    print("\nComputing divergence metrics...")

    # M1: Unigram divergence
    print("  Computing unigram distributions...")
    p_data_unigram = compute_joint_distribution_from_data(sequences, n=1, vocab_size=vocab_size)
    p_model_unigram = compute_joint_distribution_from_model(model, tokenizer, n=1, device=device)
    js_unigram = compute_js_divergence(p_data_unigram, p_model_unigram)
    print(f"    JS divergence (unigram): {js_unigram:.6f}")

    # M2: Bigram divergence
    print("  Computing bigram distributions...")
    p_data_bigram = compute_joint_distribution_from_data(sequences, n=2, vocab_size=vocab_size)
    p_model_bigram = compute_joint_distribution_from_model(model, tokenizer, n=2, device=device)
    js_bigram = compute_js_divergence(p_data_bigram, p_model_bigram)
    print(f"    JS divergence (bigram): {js_bigram:.6f}")

    # M2: Trigram divergence
    print("  Computing trigram distributions...")
    p_data_trigram = compute_joint_distribution_from_data(sequences, n=3, vocab_size=vocab_size)
    p_model_trigram = compute_joint_distribution_from_model(model, tokenizer, n=3, device=device)
    js_trigram = compute_js_divergence(p_data_trigram, p_model_trigram)
    print(f"    JS divergence (trigram): {js_trigram:.6f}")

    # M3: Conditional divergence
    print("  Computing conditional divergence...")
    m3_results = compute_conditional_divergence(
        sequences,
        model=model,
        tokenizer=tokenizer,
        n=3,
        device=device
    )
    weighted_kl = m3_results['weighted_avg']
    print(f"    Weighted KL (trigram): {weighted_kl:.6f}")

    # Conditional entropies
    print(f"\nComputing conditional entropies (n=1 to {config.postanalysis_max_n})...")
    entropies = compute_conditional_entropies(sequences, vocab_size=vocab_size, max_n=config.postanalysis_max_n)
    for i, H in enumerate(entropies, 1):
        print(f"  H[X_{i} | X_1...X_{i-1}] = {H:.4f} bits")

    # Save metadata as JSON
    print("\nSaving analysis results...")
    metadata = {
        "model_path": str(config.output_dir),
        "dataset_path": str(config.data_path),
        "vocab_size": vocab_size,
        "num_sequences": len(sequences),
        "max_n": config.postanalysis_max_n,
        "metrics": {
            "js_unigram": float(js_unigram),
            "js_bigram": float(js_bigram),
            "js_trigram": float(js_trigram),
            "weighted_kl_trigram": float(weighted_kl),
        },
        "entropies": [float(h) for h in entropies],
    }

    metadata_file = analysis_dir / "analysis_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_file}")

    # Save metrics summary
    metrics_file = analysis_dir / "metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("N-gram Analysis Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {config.output_dir}\n")
        f.write(f"Dataset: {config.data_path}\n")
        f.write(f"Vocabulary size: {vocab_size}\n")
        f.write(f"Number of sequences: {len(sequences)}\n\n")
        f.write("Divergence Metrics:\n")
        f.write(f"  JS divergence (unigram):  {js_unigram:.6f}\n")
        f.write(f"  JS divergence (bigram):   {js_bigram:.6f}\n")
        f.write(f"  JS divergence (trigram):  {js_trigram:.6f}\n")
        f.write(f"  Weighted KL (trigram):    {weighted_kl:.6f}\n\n")
        f.write("Conditional Entropies:\n")
        for i, H in enumerate(entropies, 1):
            f.write(f"  H[X_{i} | X_1...X_{i-1}] = {H:.4f} bits\n")
    print(f"  Saved metrics to {metrics_file}")

    # Save numpy arrays for dashboard
    arrays_file = analysis_dir / "analysis_arrays.npz"

    # Compute conditional probabilities for dashboard
    from baseball_states.metrics import get_model_conditional_probs, counts_to_conditional_prob
    bigram_counts = compute_ngram_counts(sequences, n=2)
    p_data_bigram_cond = counts_to_conditional_prob(bigram_counts, n=2, vocab_size=vocab_size)
    p_model_bigram_cond = get_model_conditional_probs(model, tokenizer, n=2, device=device)

    # Flatten trigram conditionals for heatmaps
    from baseball_states.utils import flatten_conditional_for_heatmap
    p_data_flat = flatten_conditional_for_heatmap(m3_results["p_data"])
    p_model_flat = flatten_conditional_for_heatmap(m3_results["p_model"])

    np.savez_compressed(
        arrays_file,
        p_data_unigram=p_data_unigram,
        p_model_unigram=p_model_unigram,
        p_data_bigram=p_data_bigram,
        p_model_bigram=p_model_bigram,
        p_data_bigram_cond=p_data_bigram_cond,
        p_model_bigram_cond=p_model_bigram_cond,
        p_data_trigram=p_data_trigram,
        p_model_trigram=p_model_trigram,
        m3_p_data=m3_results["p_data"],
        m3_p_model=m3_results["p_model"],
        m3_per_history=m3_results["per_history"],
        m3_history_freq=m3_results["history_freq"],
        p_data_flat=p_data_flat,
        p_model_flat=p_model_flat,
    )
    print(f"  Saved arrays to {arrays_file}")

    print("\n" + "="*60)
    print("Post-training analysis complete!")
    print(f"Results saved to: {analysis_dir}")
    print("="*60 + "\n")


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

    # Save initial checkpoint if requested
    if config.save_initial_checkpoint:
        from pathlib import Path
        initial_checkpoint_dir = Path(config.output_dir) / "checkpoint-initial"
        initial_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving initial checkpoint to {initial_checkpoint_dir}")
        model.save_pretrained(initial_checkpoint_dir)
        tokenizer.save_pretrained(initial_checkpoint_dir)
        print("Initial checkpoint saved!")

    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, config)

    # Determine checkpoint to resume from
    resume_checkpoint = None
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint == "auto" or config.resume_from_checkpoint is True:
            # Auto-detect latest checkpoint
            resume_checkpoint = get_latest_checkpoint(config.output_dir)
            if resume_checkpoint:
                print(f"Found checkpoint to resume from: {resume_checkpoint}")
            else:
                print("No checkpoint found, starting training from scratch")
        elif isinstance(config.resume_from_checkpoint, str) and config.resume_from_checkpoint != "auto":
            # Specific checkpoint path provided
            resume_checkpoint = config.resume_from_checkpoint
            if Path(resume_checkpoint).exists():
                print(f"Resuming from checkpoint: {resume_checkpoint}")
            else:
                print(f"Warning: Checkpoint not found at {resume_checkpoint}, starting from scratch")
                resume_checkpoint = None
    else:
        print("Starting training from scratch (resume_from_checkpoint=False)")

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

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

    # Run postprocessing analysis if enabled
    if config.run_postanalysis:
        try:
            run_postprocessing_analysis(model, tokenizer, config, device)
        except Exception as e:
            print(f"\nWarning: Postprocessing analysis failed with error: {e}")
            print("Training completed successfully, but analysis could not be completed.")

    return model, tokenizer
