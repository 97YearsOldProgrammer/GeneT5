import json
from pathlib import Path
from tokenizers import Tokenizer
import torch


class GeneTokenizer:
    """Lightweight tokenizer without transformers dependency."""
    
    def __init__(self, tokenizer_dir):
        path = Path(tokenizer_dir)
        
        # Load core tokenizer
        tokenizer_file = path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_file}")
        
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        
        # Read tokenizer.json to get special tokens from added_tokens
        with open(tokenizer_file) as f:
            tokenizer_data = json.load(f)
        
        # Build a map of special tokens from added_tokens
        added_tokens = tokenizer_data.get("added_tokens", [])
        special_tokens = {
            token["content"]: token["id"] 
            for token in added_tokens 
            if token.get("special", False)
        }
        
        # Find special tokens by their content
        self.pad_token = self._find_special_token(special_tokens, ["[PAD]", "<pad>"], "[PAD]")
        self.cls_token = self._find_special_token(special_tokens, ["[CLS]", "<cls>", "<s>"], "[CLS]")
        self.sep_token = self._find_special_token(special_tokens, ["[SEP]", "<sep>", "</s>"], "[SEP]")
        self.unk_token = self._find_special_token(special_tokens, ["[UNK]", "<unk>"], "[UNK]")
        self.mask_token = self._find_special_token(special_tokens, ["[MASK]", "<mask>"], "[MASK]")
        
        # Get IDs
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)
        self.cls_token_id = self.tokenizer.token_to_id(self.cls_token)
        self.sep_token_id = self.tokenizer.token_to_id(self.sep_token)
        self.unk_token_id = self.tokenizer.token_to_id(self.unk_token)
        
        # Validate that we found the tokens
        if self.pad_token_id is None:
            raise ValueError(f"Could not find pad_token '{self.pad_token}' in vocabulary")
        if self.cls_token_id is None:
            raise ValueError(f"Could not find cls_token '{self.cls_token}' in vocabulary")
        if self.sep_token_id is None:
            raise ValueError(f"Could not find sep_token '{self.sep_token}' in vocabulary")
        
        # BOS/EOS (often same as CLS/SEP)
        self.bos_token_id = self.cls_token_id
        self.eos_token_id = self.sep_token_id
    
    def _find_special_token(self, special_tokens, candidates, default):
        """Find first matching token from candidates in special tokens map."""
        for candidate in candidates:
            if candidate in special_tokens:
                return candidate
        # Return default if nothing found
        return default
    
    def __len__(self):
        return self.tokenizer.get_vocab_size()
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs."""
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids
    
    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs to text."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def __call__(self, text, max_length=None, truncation=True, padding=False, return_tensors=None):
        """HF-style interface for compatibility."""
        ids = self.encode(text, add_special_tokens=True)
        
        # Truncate
        if max_length and truncation and len(ids) > max_length:
            ids = ids[:max_length]
        
        # Attention mask (1 for real tokens)
        attention_mask = [1] * len(ids)
        
        # Padding
        if padding and max_length:
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
        
        result = {"input_ids": ids, "attention_mask": attention_mask}
        
        if return_tensors == "pt":
            result = {k: torch.tensor([v]) for k, v in result.items()}
        
        return result
    
    def save_pretrained(self, save_dir):
        """Save tokenizer files."""
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path / "tokenizer.json"))