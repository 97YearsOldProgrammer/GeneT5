import argparse
import json
from pathlib import Path


def load_tokenizer(tokenizer_path):
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
    tokens = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
    
    return tokens


def append_tokens(config, new_tokens):
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


def save_tokenizer(config, output_path):
    output_file = Path(output_path)
    
    if output_file.is_dir():
        output_file = output_file / "tokenizer.json"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Append missing tokens from txt file to tokenizer.json")
    parser.add_argument("txt_file",
        help="Path to txt file with tokens (one per line).")
    parser.add_argument("tokenizer",
        help="Path to tokenizer.json or directory containing it.")
    parser.add_argument("--output", default=None,
        help="Output path for updated tokenizer (default: overwrite input).")
    parser.add_argument("--dry_run", action="store_true",
        help="Show what would be added without saving.")
    args = parser.parse_args()
    
    print(f"\n{' Token Appender ':=^60}")
    
    # load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer}")
    config, config_path = load_tokenizer(args.tokenizer)
    existing            = get_existing_tokens(config)
    print(f"  Found {len(existing)} existing tokens")
    
    # load tokens from txt
    print(f"\nLoading tokens from {args.txt_file}")
    txt_tokens = load_tokens_from_txt(args.txt_file)
    print(f"  Found {len(txt_tokens)} tokens in txt file")
    
    # find missing
    missing = [t for t in txt_tokens if t not in existing]
    print(f"  Missing tokens: {len(missing)}")
    
    if not missing:
        print("\n  All tokens already present. Nothing to do.")
        return
    
    # show what will be added
    print(f"\n  Tokens to add:")
    for i, token in enumerate(missing[:20]):
        print(f"    {token}")
    if len(missing) > 20:
        print(f"    ... and {len(missing) - 20} more")
    
    if args.dry_run:
        print("\n  [DRY RUN] No changes made.")
        return
    
    # append tokens
    config, added = append_tokens(config, missing)
    
    # save
    output_path = args.output or config_path
    saved_path  = save_tokenizer(config, output_path)
    
    print(f"\n{'=' * 60}")
    print(f"SUCCESS: Added {len(added)} tokens")
    print(f"  Saved to: {saved_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()