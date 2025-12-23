# Training Configuration Guide

This directory contains configuration templates for training the baseball state prediction model.

## Configuration Files

- **`base_config.yaml`** - Base configuration for local training
- **`colab_config.yaml`** - Optimized for Google Colab (smaller batch size, adjusted for GPU memory)

## Usage

### Loading a config file

```python
from baseball_states.training import ModelConfig, train_model

# Load from YAML or JSON
config = ModelConfig.from_file('configs/base_config.yaml')

# Override specific values for this run
config.num_epochs = 20
config.output_dir = 'checkpoints/my-experiment'

# Train
model, tokenizer = train_model(config)
```

### Saving a config file

```python
# Save the config used for reproducibility
config.to_file('checkpoints/my-experiment/config_used.yaml')
```

## Checkpoint Resume Behavior

The `resume_from_checkpoint` parameter controls whether training resumes from a previous checkpoint:

### Options:

1. **`"auto"`** (default) - Smart resume
   - Automatically looks for the latest checkpoint in `output_dir`
   - If found: resumes training from that checkpoint
   - If not found: starts training from scratch
   - **Use case**: Interrupted training, want to continue where you left off

2. **`true`** - Same as `"auto"`

3. **`false`** or `null` - Always start fresh
   - Ignores any existing checkpoints
   - Creates a new randomly initialized model
   - **Use case**: New experiment, or re-running with different hyperparameters

4. **`"path/to/checkpoint-1000"`** - Resume from specific checkpoint
   - Loads the exact checkpoint you specify
   - **Use case**: Want to continue from a specific point, not necessarily the latest

### Examples:

#### Example 1: Auto-resume (default)
```yaml
# config.yaml
resume_from_checkpoint: auto
output_dir: checkpoints/run-001
```

**First run:** No checkpoints exist, trains from scratch
```
Starting training from scratch
Training... saves checkpoint-500, checkpoint-1000, checkpoint-1500
```

**Second run (interrupted at step 1200):** Resumes from latest checkpoint
```
Found checkpoint to resume from: checkpoints/run-001/checkpoint-1000
Resuming from step 1000...
```

#### Example 2: Always start fresh
```yaml
resume_from_checkpoint: false
output_dir: checkpoints/run-002
```
Ignores any existing checkpoints, starts from epoch 0.

#### Example 3: Resume from specific checkpoint
```python
config = ModelConfig.from_file('configs/base_config.yaml')
config.resume_from_checkpoint = 'checkpoints/run-001/checkpoint-500'
config.output_dir = 'checkpoints/run-001-continued'
```

## Directory Structure After Training

```
checkpoints/
└── my-experiment/
    ├── checkpoint-initial/      # Initial untrained model (if save_initial_checkpoint=true)
    ├── checkpoint-500/           # Intermediate checkpoint at step 500
    ├── checkpoint-1000/          # Intermediate checkpoint at step 1000
    ├── checkpoint-1500/          # Intermediate checkpoint at step 1500
    ├── config_used.yaml          # Actual config used for this run
    ├── pytorch_model.bin         # Final trained model
    ├── config.json               # Model architecture config
    └── logs/
        └── gpt2-train-2025-1223-143022/  # TensorBoard logs
```

## Best Practices

1. **Use unique `output_dir` for each experiment**
   ```python
   config.output_dir = f'checkpoints/experiment-{experiment_name}'
   ```

2. **Save the config after training**
   ```python
   config.to_file(f'{config.output_dir}/config_used.yaml')
   ```
   This creates a record of exactly what parameters were used.

3. **For long-running jobs, use `resume_from_checkpoint="auto"`**
   - Protects against interruptions
   - Can safely re-run the same script if it crashes

4. **For new experiments, explicitly set `resume_from_checkpoint=false`**
   - Ensures you don't accidentally continue from an old checkpoint
   - Or use a new `output_dir` for each experiment

## Common Patterns

### Pattern 1: Development iteration (quick experiments)
```yaml
# dev_config.yaml
num_epochs: 1
train_fraction: 0.1  # Use only 10% of data
resume_from_checkpoint: false  # Always start fresh
```

### Pattern 2: Production training (long-running, resumable)
```yaml
# prod_config.yaml
num_epochs: 50
train_fraction: 1.0
resume_from_checkpoint: auto  # Can resume if interrupted
save_steps: 500  # Save frequently
```

### Pattern 3: Fine-tuning from a checkpoint
```python
config = ModelConfig.from_file('configs/base_config.yaml')
config.resume_from_checkpoint = 'checkpoints/pretrained/checkpoint-10000'
config.output_dir = 'checkpoints/finetuned'
config.learning_rate = 1e-5  # Lower learning rate for fine-tuning
config.num_epochs = 5
```
