import torch
from torch.utils.data import Dataset

class GameSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapper for HuggingFace datasets produced by sequencer.py.

    The HF dataset contains:
        - 'sequence': List[str] - token sequences like ['<START_INNING>', '0_Empty', '1_1st', ...]
        - 'length': int - length of each sequence

    This dataset converts token sequences to IDs using the tokenizer and prepares
    them for causal language modeling (next-token prediction).
    """

    def __init__(self, hf_dataset, tokenizer, max_length: int = 512, pad_to_max: bool = True):
        """
        Args:
            hf_dataset: HuggingFace Dataset from sequencer.py (with 'sequence' column)
            tokenizer: GameStateTokenizer instance
            max_length: Maximum sequence length (truncate longer sequences)
            pad_to_max: If True, pad all sequences to max_length; if False, only pad to batch max
        """
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single training example.

        Returns:
            dict with:
                - 'input_ids': tensor of input token IDs (for predicting next token)
                - 'labels': tensor of target token IDs (shifted by 1)
                - 'attention_mask': tensor indicating which positions are padding
        """
        # Get token sequence (list of strings)
        token_sequence = self.data[idx]['sequence']

        # Convert list of tokens to space-separated string for tokenizer
        # The tokenizer's _tokenize method expects a string and splits on spaces
        token_string = ' '.join(token_sequence)

        # Use the public encode method
        ids = self.tokenizer.encode(token_string, add_special_tokens=False)

        # Truncate if necessary
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]

        # For causal LM: input predicts next token
        # input_ids: [token_0, token_1, ..., token_n-1]
        # labels:    [token_1, token_2, ..., token_n]
        input_ids = ids[:-1] if len(ids) > 1 else ids
        labels = ids[1:] if len(ids) > 1 else ids

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)

        # Pad to max_length if requested
        if self.pad_to_max:
            pad_length = self.max_length - 1 - len(input_ids)  # -1 because we removed last token
            if pad_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
                labels = labels + [-100] * pad_length  # -100 is ignored by CrossEntropyLoss
                attention_mask = attention_mask + [0] * pad_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


def collate_fn(batch, pad_token_id: int):
    """
    Custom collate function for DataLoader when pad_to_max=False.
    Pads sequences to the maximum length in the batch.

    Args:
        batch: List of dicts from __getitem__
        pad_token_id: ID to use for padding

    Returns:
        Batched tensors
    """
    # Find max length in this batch
    max_len = max(len(item['input_ids']) for item in batch)

    # Pad each sequence to max_len
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len

        # Pad input_ids
        padded_input = torch.cat([
            item['input_ids'],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        input_ids.append(padded_input)

        # Pad labels with -100 (ignore index)
        padded_labels = torch.cat([
            item['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        labels.append(padded_labels)

        # Pad attention mask
        padded_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        attention_mask.append(padded_mask)

    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask)
    }