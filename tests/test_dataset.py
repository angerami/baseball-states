from baseball_states.dataset import GameSequenceDataset
from torch.utils.data import DataLoader


def test_dataset():
    train_dataset = GameSequenceDataset(dataset, max_length = 22)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    #####
    # Check one batch
    batch = next(iter(train_loader))
    print('Input IDs')
    print(batch['input_ids'].shape)  # Should be [batch_size, seq_len-1]
    print('Labels')
    print(batch['labels'].shape)     # Should be [batch_size, seq_len-1]

    # Decode first sequence to verify
    print(tokenizer.decode(batch['input_ids'][0]))
    print(tokenizer.decode(batch['labels'][0]))

    # Check for pad tokens
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"Pad tokens in batch: {(batch['input_ids'] == tokenizer.pad_token_id).sum()}")

    # Verify labels are shifted inputs
    print(f"Input[0]: {batch['input_ids'][0]}")
    print(f"Label[0]: {batch['labels'][0]}")
