import sys
import argparse
from lib import tokenizer as tk



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
config, config_path = tk.load_tokenizer_config(args.tokenizer)
existing            = tk.get_existing_tokens(config)
print(f"  Found {len(existing)} existing tokens")

# load tokens from txt
print(f"\nLoading tokens from {args.txt_file}")
txt_tokens = tk.load_tokens_from_txt(args.txt_file)
print(f"  Found {len(txt_tokens)} tokens in txt file")

# find missing
missing = tk.find_missing_tokens(config, txt_tokens)
print(f"  Missing tokens: {len(missing)}")

if not missing:
    print("\n  All tokens already present. Nothing to do.")
    sys.exit(0)

# show what will be added
print(f"\n  Tokens to add:")
for i, token in enumerate(missing[:20]):
    print(f"    {token}")
if len(missing) > 20:
    print(f"    ... and {len(missing) - 20} more")

if args.dry_run:
    print("\n  [DRY RUN] No changes made.")
    sys.exit(0)

# append tokens
config, added = tk.append_tokens_to_config(config, missing)

# save
output_path = args.output or config_path
saved_path  = tk.save_tokenizer_config(config, output_path)

print(f"\n{'=' * 60}")
print(f"SUCCESS: Added {len(added)} tokens")
print(f"  Saved to: {saved_path}")
print(f"{'=' * 60}")