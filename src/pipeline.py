DATA_DIR = "data"
INPUT_DS = f'{DATA_DIR}/plate_appearances'
SEQUENCE_DS = f'{DATA_DIR}/sequences_inning'
TOKENIZED_DS = f'{DATA_DIR}/tokens_inning'

import pandas as pd
from datasets import load_from_disk, Dataset
from baseball_states.sequencer import GameStateSequencer, SequenceConfig
from baseball_states.tokenizer import GameStateTokenizer

def run_pipeline(input_path = INPUT_DS, seq_path = SEQUENCE_DS, tokenized_path=TOKENIZED_DS, inning=True, run_sequencer=True, save_sequences=True):

    #Generate or load the sequences of tokens
    if run_sequencer:
        sequencer = GameStateSequencer(data_path=input_path)
        config = SequenceConfig(level='inning' if inning else 'game')
        sequences = sequencer.create_sequences(config)
        df = sequencer.to_pandas(sequences)
        if save_sequences:
            sequencer.save_sequences(df, seq_path)
    else:
        hf_dataset = load_from_disk(seq_path)
        df = hf_dataset.to_pandas()

    tokenizer = GameStateTokenizer()
    tokens = df['sequence']
    ids = [tokenizer.encode(token_list, add_special_tokens=False) for token_list in tokens]
    df_tokenized = pd.DataFrame({'tokens': ids, 'length': [len(x) for x in ids]})
    dataset = Dataset.from_pandas(df_tokenized)
    dataset.save_to_disk(tokenized_path)
    print(f"Saved {len(tokens)} tokens to {tokenized_path}")


    from baseball_states.dataset import GameSequenceDataset
    from torch.utils.data import DataLoader
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

if __name__ == "__main__":
    run_pipeline()