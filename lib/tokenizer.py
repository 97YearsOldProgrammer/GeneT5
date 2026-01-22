import json
from pathlib import Path


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


def save_tokens_to_txt(tokens, txt_path):
    """Save tokens to a text file (one per line)"""
    path = Path(txt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        for token in tokens:
            f.write(f"{token}\n")
    
    return path


def append_tokens_to_txt(tokens, txt_path):
    """Append tokens to existing txt file, avoiding duplicates"""
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


def update_tokenizer_from_txt(tokenizer_path, txt_path, output_path=None):
    """Update tokenizer with tokens from txt file"""
    
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