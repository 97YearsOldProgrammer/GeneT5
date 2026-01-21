import argparse
import sys
from pathlib import Path

from lib.build_model import build_gt5


parser = argparse.ArgumentParser(
    description="Initialize GeneT5 weights from DNABERT-2 and save to disk.")
parser.add_argument("--save_dir", type=str, default="./checkpoints/genet5_init", 
    help="Directory to save the initialized model weights and config.")
parser.add_argument("--dnabert_path", type=str, default="zhihan1996/DNABERT-2-117M",
    help="HuggingFace model ID or local path to DNABERT-2.")
parser.add_argument("--vocab_size", type=int, default=None, 
    help="Vocabulary size. If None, auto-detect from tokenizer.")
parser.add_argument("--new_tokens", type=str, nargs="*", default=None,
    help="New tokens to add to tokenizer.")
parser.add_argument("--new_tokens_file", type=str, default=None,
    help="Path to file containing new tokens (one per line).")
parser.add_argument("--block_size", type=int, default=64, 
    help="Block size for BigBird sparse attention.")
parser.add_argument("--decoder_layers", type=int, default=None, 
    help="Number of decoder layers. Defaults to matching encoder (12).")
parser.add_argument("--decoder_heads", type=int, default=None,
    help="Number of decoder heads. Defaults to matching encoder.")
parser.add_argument("--tie_weights", action="store_true", default=True,
    help="Tie input/output embeddings.")
parser.add_argument("--no_tie_weights", action="store_false", dest="tie_weights",
    help="Do not tie input/output embeddings.")
parser.add_argument("--use_moe", action="store_true", default=True,
    help="Enable Mixture of Experts in Decoder (default: True).")
parser.add_argument("--no_moe", action="store_false", dest="use_moe",
    help="Disable Mixture of Experts in Decoder.")
parser.add_argument("--num_experts", type=int, default=8,
    help="Number of experts for MoE (default: 8).")
parser.add_argument("--moe_top_k", type=int, default=2,
    help="Top-K routing for MoE (default: 2).")

# Initialization Parameters (DeepSeek-style)
parser.add_argument("--init_std", type=float, default=0.02,
    help="Default standard deviation for random parameter initialization. "
         "DeepSeek uses 0.006, standard transformer uses 0.02.")
parser.add_argument("--init_embed_std", type=float, default=None,
    help="Std for embedding layers (defaults to --init_std).")
parser.add_argument("--init_ffn_std", type=float, default=None,
    help="Std for FFN layers (defaults to --init_std).")
parser.add_argument("--init_attn_std", type=float, default=None,
    help="Std for attention layers (defaults to --init_std).")
parser.add_argument("--init_moe_router_std", type=float, default=None,
    help="Std for MoE router (defaults to --init_std).")

args = parser.parse_args()


# Load tokens from file if provided
new_tokens_list = args.new_tokens or []
if args.new_tokens_file:
    with open(args.new_tokens_file, "r") as f:
        file_tokens = [line.strip() for line in f if line.strip()]
    new_tokens_list.extend(file_tokens)

# Visual separator
print(f"\n{' GeneT5 Initialization ':=^60}")
    
# Run the build function
saved_path = build_gt5(
    dnabert_model_name  = args.dnabert_path,
    save_dir            = args.save_dir,
    block_size          = args.block_size,
    decoder_num_layers  = args.decoder_layers,
    decoder_num_heads   = args.decoder_heads,
    decoder_use_moe     = args.use_moe,
    decoder_num_experts = args.num_experts,
    decoder_moe_top_k   = args.moe_top_k,
    vocab_size          = args.vocab_size,
    tie_weights         = args.tie_weights,
    new_tokens_list     = new_tokens_list if new_tokens_list else None,
    init_std            = args.init_std,
    init_embed_std      = args.init_embed_std,
    init_ffn_std        = args.init_ffn_std,
    init_attn_std       = args.init_attn_std,
    init_moe_router_std = args.init_moe_router_std,
)
        
print(f"\nSUCCESS: Model initialized and saved to:")
print(f"  -> {saved_path}")
print(f"  -> {saved_path}/config.json")
print(f"  -> {saved_path}/pytorch_model.bin")