import argparse
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Generate extended tokenizer file with special tokens.")
parser.add_argument("--output", type=str, default="./extended_tokens.txt",
    help="Output file path for the token list.")
args = parser.parse_args()


# Special Tokens
SPECIAL_TOKENS = [
    "<bos>",            # Beginning of Sentence
    "<eos>",            # End of Sentence
    "<pad>",            # Padding
    "<sep>",            # Separator
    "<mask>",           # Masking
    "<unk>",
]

# Type Tokens
TYPE_TOKENS = [
    # Prokaryotic
    "mobile_genetic_element",
    "origin_of_replication",
    
    # Casual
    "gene",
    "exon",
    "intron",
    "cds",
    "utr5",
    
    # RNA
    "ncRNA",            # Non-coding RNA
    "rRNA",             # Ribosomal  RNA
    "tRNA",             # Transfer   RNA
]

# Task Specific tokens
TASK_TOKENS = [
    "<translate>",
    "<transcribe>",
    "<annotate>",
    "<generate>",
]

# Strand tokens
STRAND_TOKENS = ["+", "-"]

# Build token list
tokens = []
tokens.extend(SPECIAL_TOKENS)
tokens.extend(TYPE_TOKENS)
tokens.extend(TASK_TOKENS)
tokens.extend(STRAND_TOKENS)

# Add integer tokens (0-999)
for i in range(1000):
    tokens.append(str(i))

# Write to file
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    for token in tokens:
        f.write(f"{token}\n")

print(f"Generated {len(tokens)} tokens -> {output_path}")