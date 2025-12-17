from transformers import PreTrainedTokenizer
import json
import os

class GameStateTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}
    
    def __init__(self, vocab_file=None, **kwargs):
        # If loading from file
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                self.token_to_id = json.load(f)
            self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
        else:
            # Build vocab from scratch
            self.special_tokens = ['<PAD>', '<START_GAME>', '<END_GAME>', 
                                   '<START_INNING>', '<END_INNING>']
            self.state_tokens = [f'OUT{o}_BASE{b}' for o in range(3) for b in range(8)]
            self.outcome_tokens = ['SINGLE', 'DOUBLE', ...]
            
            vocab = self.special_tokens + self.state_tokens + self.outcome_tokens
            self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
            self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        super().__init__(
            pad_token='<PAD>',
            bos_token='<START_GAME>',
            eos_token='<END_GAME>',
            **kwargs
        )    
    def _tokenize(self, text):
        # Already tokenized, just return as-is
        return text.split()
    
    def _convert_token_to_id(self, token):
        return self.token_to_id.get(token, self.token_to_id['<PAD>'])
    
    def _convert_id_to_token(self, index):
        return self.id_to_token.get(index, '<PAD>')
    
    def get_vocab(self):
        return self.token_to_id.copy()
    
    @property
    def vocab_size(self):
        return len(self.token_to_id)
    
    def save_vocabulary(self, save_directory, filename_prefix=None):
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        with open(vocab_file, 'w') as f:
            json.dump(self.token_to_id, f, indent=2)
        return (vocab_file,)
    
# Usage
# Create and save
# tokenizer = GameStateTokenizer()
# tokenizer.save_pretrained("./model_dir")

# # Load from disk
# tokenizer = GameStateTokenizer.from_pretrained("./model_dir")

# # Or from HF Hub
# tokenizer = AutoTokenizer.from_pretrained("username/baseball-model")