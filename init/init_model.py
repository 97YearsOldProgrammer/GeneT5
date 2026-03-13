import argparse

from lib.model.build import build_gt5


parser = argparse.ArgumentParser(
    description="Initialize GeneT5 weights from DNABERT-2 and save to disk.")
parser.add_argument("--save_dir", type=str, default="model/init_dense_24L",
    help="Directory to save the initialized model weights and config.")
parser.add_argument("--dnabert_path", type=str, default="zhihan1996/DNABERT-2-117M",
    help="HuggingFace model ID or local path to DNABERT-2.")
parser.add_argument("--layers", type=int, default=24,
    help="Number of encoder layers (>12 triggers G_stack from DNABERT-2 12L).")
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
parser.add_argument("--depth_scaling", action="store_true", default=True,
    help="LayerNorm Scaling: residual *= 1/sqrt(depth) to prevent curse of depth.")
parser.add_argument("--no_depth_scaling", action="store_false", dest="depth_scaling",
    help="Disable LayerNorm Scaling.")
parser.add_argument("--init_std", type=float, default=0.006,
    help="Standard deviation for random init.")

args = parser.parse_args()

print(f"\n{' GeneT5 Initialization ':=^60}")

saved_path = build_gt5(
    dnabert_model_name  = args.dnabert_path,
    save_dir            = args.save_dir,
    num_layers          = args.layers,
    num_heads           = args.heads,
    ff_dim              = args.ff_dim,
    dropout             = args.dropout,
    tie_weights         = args.tie_weights,
    init_std            = args.init_std,
    depth_scaling       = args.depth_scaling,
)

print(f"\nSUCCESS: Model initialized and saved to:")
print(f"  -> {saved_path}")
print(f"  -> {saved_path}/config.json")
print(f"  -> {saved_path}/pytorch_model.bin")
