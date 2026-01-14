
import json
from pathlib    import Path
from tokenizers import Tokenizer
import torch


class GeneTokenizer:
    """Lightweight tokenizer without transformers dependency."""
    
    def __init__(self, tokenizer_dir):
        path = Path(tokenizer_dir)
        
        # Load core tokenizer
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        
        # Load special tokens
        with open(path / "special_tokens_map.json") as f:
            special = json.load(f)
        
        self.pad_token = special.get("pad_token", {}).get("content", "[PAD]")
        self.cls_token = special.get("cls_token", {}).get("content", "[CLS]")
        self.sep_token = special.get("sep_token", {}).get("content", "[SEP]")
        self.unk_token = special.get("unk_token", {}).get("content", "[UNK]")
        self.mask_token = special.get("mask_token", {}).get("content", "[MASK]")
        
        # Get IDs
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)
        self.cls_token_id = self.tokenizer.token_to_id(self.cls_token)
        self.sep_token_id = self.tokenizer.token_to_id(self.sep_token)
        self.unk_token_id = self.tokenizer.token_to_id(self.unk_token)
        
        # BOS/EOS (often same as CLS/SEP)
        self.bos_token_id = self.cls_token_id
        self.eos_token_id = self.sep_token_id
    
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