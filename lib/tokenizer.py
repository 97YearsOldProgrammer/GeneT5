import json
from pathlib import Path
from transformers import AutoTokenizer


#####################
##### Auxilary  #####
#####################


def load_tokenizer_config(tokenizer_path):
    """Load tokenizer.json config file"""
    
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
    """Extract all existing tokens from tokenizer config"""
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
    """Load tokens from a text file (one per line)"""
    tokens = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
    
    return tokens


def append_tokens_to_config(config, new_tokens):
    """Append new tokens to tokenizer config"""
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
    """Save tokenizer config to file"""
    
    output_file = Path(output_path)
    
    if output_file.is_dir():
        output_file = output_file / "tokenizer.json"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_file


def find_missing_tokens(config, tokens):
    """Find tokens not present in tokenizer config"""
    
    existing = get_existing_tokens(config)
    return [t for t in tokens if t not in existing]


#######################
#####  Tokenizer  #####
#######################


class GeneTokenizer:
    """Wrapper around HuggingFace tokenizer for GeneT5 models."""
    
    def __init__(self, tokenizer_path, trust_remote_code=True):
        """
        Load tokenizer from path.
        
        Args:
            tokenizer_path: Path to tokenizer directory or tokenizer.json
            trust_remote_code: Whether to trust remote code (default: True)
        """
        self.tokenizer_path = Path(tokenizer_path)
        
        # Try loading with AutoTokenizer first (handles most cases)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), 
                trust_remote_code=trust_remote_code
            )
        except Exception as e:
            # Fallback: try loading tokenizer.json directly
            config, _ = load_tokenizer_config(tokenizer_path)
            self._vocab = self._extract_vocab(config)
            self._tokenizer = None
    
    def _extract_vocab(self, config):
        """Extract vocabulary from tokenizer config."""
        vocab = {}
        
        # Get vocab from model section
        if "model" in config and "vocab" in config["model"]:
            model_vocab = config["model"]["vocab"]
            if isinstance(model_vocab, dict):
                vocab.update(model_vocab)
            elif isinstance(model_vocab, list):
                vocab = {token: i for i, token in enumerate(model_vocab)}
        
        # Add tokens from added_tokens section
        if "added_tokens" in config:
            for token_info in config["added_tokens"]:
                if isinstance(token_info, dict):
                    if "content" in token_info and "id" in token_info:
                        vocab[token_info["content"]] = token_info["id"]
                elif isinstance(token_info, str):
                    if token_info not in vocab:
                        vocab[token_info] = len(vocab)
        
        return vocab
    
    def __len__(self):
        """Return vocabulary size."""
        if self._tokenizer is not None:
            return len(self._tokenizer)
        return len(self._vocab)
    
    def __call__(self, text, **kwargs):
        """Tokenize text."""
        if self._tokenizer is not None:
            return self._tokenizer(text, **kwargs)
        raise NotImplementedError("Direct tokenization requires HuggingFace tokenizer")
    
    def encode(self, text, **kwargs):
        """Encode text to token ids."""
        if self._tokenizer is not None:
            return self._tokenizer.encode(text, **kwargs)
        raise NotImplementedError("Encoding requires HuggingFace tokenizer")
    
    def decode(self, token_ids, **kwargs):
        """Decode token ids to text."""
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids, **kwargs)
        raise NotImplementedError("Decoding requires HuggingFace tokenizer")
    
    @property
    def vocab_size(self):
        """Return vocabulary size."""
        return len(self)
    
    def get_vocab(self):
        """Return vocabulary dictionary."""
        if self._tokenizer is not None:
            return self._tokenizer.get_vocab()
        return self._vocab.copy()
    
    def save_pretrained(self, save_path):
        """Save tokenizer to directory."""
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(save_path)
        else:
            raise NotImplementedError("Saving requires HuggingFace tokenizer")