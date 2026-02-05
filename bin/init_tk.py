import argparse
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Generate extended tokenizer file with special tokens.")
parser.add_argument("output", type=str, default="./extended_tokens.txt",
    help="Output file path for the token list.")
args = parser.parse_args()


# Special Tokens
# Original DNABERT-v2 have already contain following tokens
# [UNK] [CLS] [SEP] [PAD] [MASK]
# [CLS] --> Classification


SPECIAL_TOKENS = [
    "<bos>",
    "<eos>",
]

# Type Tokens
TYPE_TOKENS = [
    
    # Prokaryote
    "origin_of_replication", 
    "mobile_genetic_element",
    
    "pseudogene", 
    "ncRNA", "ncrna", 
    "tRNA", "trna", 
    "rRNA", "rrna", 
    "tmRNA", "tmrna", 
    "sRNA", "srna",
    "misc_RNA", "misc_rna", 
    "antisense_RNA", "antisense_rna",
    "RNase_P_RNA", "rnase_p_rna", 
    "SRP_RNA", "srp_rna",
    "snoRNA", "snorna", 
    "snRNA", "snrna",
    "transposable_element", 
]

# Task Specific Tokens
TASK_TOKENS = [
    "[ATT]",
    "[HIT]",
]

# Strand tokens
STRAND_TOKENS = ["+", "-"]

# Text Specific Tokens
TEXT_TOKENS = [
    r"[\t]",
    r"[\n]",
    ".",
]

# Build token list
tokens = []
tokens.extend(SPECIAL_TOKENS)
tokens.extend(TYPE_TOKENS)
tokens.extend(TASK_TOKENS)
tokens.extend(STRAND_TOKENS)
tokens.extend(TEXT_TOKENS)

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