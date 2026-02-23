import argparse

from lib.model.build import build_gt5


parser = argparse.ArgumentParser(
    description="Initialize GeneT5 weights from DNABERT-2 and save to disk.")
parser.add_argument("--save_dir", type=str, default="/workspace/model/GeneT5/init/genet5_init",
    help="Directory to save the initialized model weights and config.")
parser.add_argument("--dnabert_path", type=str, default="zhihan1996/DNABERT-2-117M",
    help="HuggingFace model ID or local path to DNABERT-2.")
parser.add_argument("--layers", type=int, default=None,
    help="Number of decoder layers. Defaults to matching DNABERT-2.")
parser.add_argument("--heads", type=int, default=None,
    help="Number of attention heads. Defaults to matching DNABERT-2.")
parser.add_argument("--ff_dim", type=int, default=None,
    help="Feed-forward dimension. Defaults to matching DNABERT-2.")
parser.add_argument("--dropout", type=float, default=0.1,
    help="Dropout rate.")
parser.add_argument("--tie_weights", action="store_true", default=True,
    help="Tie input/output embeddings.")
parser.add_argument("--no_tie_weights", action="store_false", dest="tie_weights",
    help="Do not tie input/output embeddings.")
parser.add_argument("--use_moe", action="store_true", default=True,
    help="Enable Mixture of Experts.")
parser.add_argument("--no_moe", action="store_false", dest="use_moe",
    help="Disable Mixture of Experts.")
parser.add_argument("--num_experts", type=int, default=16,
    help="Number of experts for MoE.")
parser.add_argument("--moe_top_k", type=int, default=2,
    help="Top-K routing for MoE.")
parser.add_argument("--init_std", type=float, default=0.02,
    help="Default standard deviation for random parameter initialization.")
parser.add_argument("--init_embed_std", type=float, default=None,
    help="Std for embedding layers (defaults to --init_std).")
parser.add_argument("--init_ffn_std", type=float, default=None,
    help="Std for FFN layers (defaults to --init_std).")
parser.add_argument("--init_attn_std", type=float, default=None,
    help="Std for attention layers (defaults to --init_std).")
parser.add_argument("--init_moe_router_std", type=float, default=0.006,
    help="Std for MoE router.")

args = parser.parse_args()

print(f"\n{' GeneT5 Initialization ':=^60}")

saved_path = build_gt5(
    dnabert_model_name  = args.dnabert_path,
    save_dir            = args.save_dir,
    num_layers          = args.layers,
    num_heads           = args.heads,
    ff_dim              = args.ff_dim,
    dropout             = args.dropout,
    use_alibi           = True,
    use_moe             = args.use_moe,
    num_experts         = args.num_experts,
    moe_top_k           = args.moe_top_k,
    tie_weights         = args.tie_weights,
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
