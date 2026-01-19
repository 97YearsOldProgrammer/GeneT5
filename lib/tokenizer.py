import json
from pathlib import Path

# optional import - only needed for GeneTokenizer class
try:
    from tokenizers import Tokenizer
    import torch
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


class GeneTokenizer:
    """Lightweight tokenizer without transformers dependency."""
    
    def __init__(self, tokenizer_dir):
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("GeneTokenizer requires 'tokenizers' and 'torch' packages")
        
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


##############################
#####  Token Management  #####
##############################


def load_tokenizer_config(tokenizer_path):
    """Load tokenizer.json config file."""
    path = Path(tokenizer_path)
    
    if path.is_dir():
        config_file = path / "tokenizer.json"
    else:
        config_file = path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Tokenizer not found: {config_file}")
    
    with open(config_file, 'r') as f:
        return json.load(f), config_file


def get_existing_tokens(config):
    """Extract all existing tokens from tokenizer config."""
    tokens = set()
    
    if "model" in config and "vocab" in config["model"]:
        vocab = config["model"]["vocab"]
        if isinstance(vocab, dict):
            tokens.update(vocab.keys())
        elif isinstance(vocab, list):
            tokens.update(vocab)
    
    if "added_tokens" in config:
        for token_info in config["added_tokens"]:
            if isinstance(token_info, dict) and "content" in token_info:
                tokens.add(token_info["content"])
            elif isinstance(token_info, str):
                tokens.add(token_info)
    
    return tokens


def load_tokens_from_txt(txt_path):
    """Load tokens from a text file (one per line)."""
    tokens = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
    
    return tokens


def save_tokens_to_txt(tokens, txt_path):
    """Save tokens to a text file (one per line)."""
    path = Path(txt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        for token in tokens:
            f.write(f"{token}\n")
    
    return path


def append_tokens_to_txt(tokens, txt_path):
    """Append tokens to existing txt file, avoiding duplicates."""
    path = Path(txt_path)
    
    existing = set()
    if path.exists():
        existing = set(load_tokens_from_txt(txt_path))
    
    new_tokens = [t for t in tokens if t not in existing]
    
    if new_tokens:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a') as f:
            for token in new_tokens:
                f.write(f"{token}\n")
    
    return new_tokens


def append_tokens_to_config(config, new_tokens):
    """Append new tokens to tokenizer config."""
    if not new_tokens:
        return config, []
    
    max_id = 0
    
    if "added_tokens" in config:
        for token_info in config["added_tokens"]:
            if isinstance(token_info, dict) and "id" in token_info:
                max_id = max(max_id, token_info["id"])
    
    if "model" in config and "vocab" in config["model"]:
        vocab = config["model"]["vocab"]
        if isinstance(vocab, dict):
            max_id = max(max_id, max(vocab.values()) if vocab else 0)
    
    if "added_tokens" not in config:
        config["added_tokens"] = []
    
    added = []
    for i, token in enumerate(new_tokens):
        new_id      = max_id + 1 + i
        token_entry = {
            "id":         new_id,
            "content":    token,
            "single_word": False,
            "lstrip":     False,
            "rstrip":     False,
            "normalized": False,
            "special":    False
        }
        config["added_tokens"].append(token_entry)
        added.append((token, new_id))
    
    return config, added


def save_tokenizer_config(config, output_path):
    """Save tokenizer config to file."""
    output_file = Path(output_path)
    
    if output_file.is_dir():
        output_file = output_file / "tokenizer.json"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_file


def find_missing_tokens(config, tokens):
    """Find tokens not present in tokenizer config."""
    existing = get_existing_tokens(config)
    return [t for t in tokens if t not in existing]


def update_tokenizer_from_txt(tokenizer_path, txt_path, output_path=None):
    """
    Update tokenizer with tokens from txt file.
    
    Returns:
        tuple: (config, added_tokens, output_path)
    """
    config, config_path = load_tokenizer_config(tokenizer_path)
    existing            = get_existing_tokens(config)
    txt_tokens          = load_tokens_from_txt(txt_path)
    missing             = [t for t in txt_tokens if t not in existing]
    
    if missing:
        config, added = append_tokens_to_config(config, missing)
        out_path      = output_path or config_path
        saved         = save_tokenizer_config(config, out_path)
        return config, added, saved
    
    return config, [], config_path