"""
Test script to verify model initialization and forward pass without training.
Run this before training to catch any issues.
"""
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from datasets import load_from_disk
from baseball_states.tokenizer import GameStateTokenizer
from baseball_states.dataset import GameSequenceDataset


def test_model_init():
    """Test model initialization with random input"""
    print("=" * 60)
    print("TEST 1: Model Initialization with Random Input")
    print("=" * 60)
    
    tokenizer = GameStateTokenizer()
    
    # Model config
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=32,
        n_embd=64,
        n_layer=4,
        n_head=4,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    model = GPT2LMHeadModel(config)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Create random input
    batch_size = 4
    seq_len = 10
    random_input = torch.randint(0, len(tokenizer), (batch_size, seq_len))
    
    print(f"\nRandom input shape: {random_input.shape}")
    print(f"Sample input IDs: {random_input[0].tolist()}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=random_input, labels=random_input)
    
    print(f"\nLogits shape: {outputs.logits.shape}")
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Expected logits shape: ({batch_size}, {seq_len}, {len(tokenizer)})")
    
    # Check that loss is reasonable (not NaN or inf)
    assert not torch.isnan(outputs.loss), "Loss is NaN!"
    assert not torch.isinf(outputs.loss), "Loss is inf!"
    
    print("\n✓ Random input test passed")
    return model, tokenizer


def test_real_data():
    """Test forward pass with real data from dataset"""
    print("\n" + "=" * 60)
    print("TEST 2: Forward Pass with Real Data")
    print("=" * 60)
    
    # Load tokenizer and model
    tokenizer = GameStateTokenizer()
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=32,
        n_embd=64,
        n_layer=4,
        n_head=4,
        pad_token_id=tokenizer.token_to_id['<PAD>'],
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    
    # Load dataset
    try:
        hf_dataset = load_from_disk("data/processed/sequences_inning")
        dataset = GameSequenceDataset(hf_dataset, tokenizer, max_length=32)
        
        print(f"Dataset size: {len(dataset)}")
        
        # Get a few samples
        sample = dataset[0]
        print(f"\nSample input shape: {sample['input_ids'].shape}")
        print(f"Sample labels shape: {sample['labels'].shape}")
        
        # Decode to verify
        decoded_input = tokenizer.decode(sample['input_ids'].tolist())
        decoded_labels = tokenizer.decode(
            [t for t in sample['labels'].tolist() if t != -100]
        )
        
        print(f"\nDecoded input: {decoded_input[:10]}...")
        print(f"Decoded labels: {decoded_labels[:10]}...")
        
        # Create batch
        batch_inputs = torch.stack([dataset[i]['input_ids'] for i in range(4)])
        batch_labels = torch.stack([dataset[i]['labels'] for i in range(4)])
        
        print(f"\nBatch input shape: {batch_inputs.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=batch_inputs, labels=batch_labels)
        
        print(f"\nLogits shape: {outputs.logits.shape}")
        print(f"Loss: {outputs.loss.item():.4f}")
        
        # Check predictions
        predictions = torch.argmax(outputs.logits, dim=-1)
        print(f"Predictions shape: {predictions.shape}")
        
        # Sample prediction
        print(f"\nSample prediction (first sequence, first 5 tokens):")
        print(f"Input:  {batch_inputs[0, :5].tolist()}")
        print(f"Pred:   {predictions[0, :5].tolist()}")
        print(f"Label:  {batch_labels[0, :5].tolist()}")
        
        print("\n✓ Real data test passed")
        
    except FileNotFoundError:
        print("\n⚠ Dataset not found at data/processed/sequences_inning")
        print("Run the pipeline first to generate sequences")


def test_generation():
    """Test autoregressive generation"""
    print("\n" + "=" * 60)
    print("TEST 3: Autoregressive Generation")
    print("=" * 60)
    
    tokenizer = GameStateTokenizer()
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=32,
        n_embd=64,
        n_layer=4,
        n_head=4,
        pad_token_id=tokenizer.token_to_id['<PAD>'],
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    
    # Start with <START_INNING>
    start_token_id = tokenizer.bos_token_id
    input_ids = torch.tensor([[start_token_id]])
    
    print(f"Starting generation with: {tokenizer.id_to_token[start_token_id]}")
    
    # Generate 10 tokens
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=10,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_tokens = [tokenizer.id_to_token[i] for i in generated[0].tolist()]
    print(f"Generated sequence: {generated_tokens}")
    
    print("\n✓ Generation test passed")


if __name__ == "__main__":
    # Run all tests
    model, tokenizer = test_model_init()
    test_real_data()
    test_generation()
    
    print("\n" + "=" * 60)
    print("All tests passed! Ready to train.")
    print("=" * 60)