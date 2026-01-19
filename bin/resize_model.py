import argparse
import torch
import json
import sys
from pathlib import Path

from lib.model      import GeneT5
from lib.tokenizer  import GeneTokenizer


parser = argparse.ArgumentParser(
    description="Resize GeneT5 model embeddings to match tokenizer vocabulary.")
parser.add_argument("model_path", type=str,
    help="Path to the model directory (containing config.json and pytorch_model.bin)")
parser.add_argument("tokenizer_path", type=str,
    help="Path to tokenizer directory. Defaults to model_path if not provided.")
args = parser.parse_args()


# 1. Load Tokenizer
tokenizer = GeneTokenizer(args.tokenizer_path)
new_vocab_size = len(tokenizer)
print(f"Tokenizer Size: {new_vocab_size}")

# 2. Load Model
model = GeneT5.from_pretrained(args.model_path, device="cpu")
print(f"Old Model Size: {model.encoder_embed.weight.shape[0]}")

# 3. Resize Embeddings
if model.encoder_embed.weight.shape[0] != new_vocab_size:
    print(f"Resizing model from {model.vocab_size} to {new_vocab_size}...")
    
    # Create new embedding layers
    old_enc = model.encoder_embed
    old_dec = model.decoder_embed
    old_head = model.lm_head
    
    # New layers
    model.encoder_embed = torch.nn.Embedding(new_vocab_size, model.embed_dim)
    model.decoder_embed = torch.nn.Embedding(new_vocab_size, model.embed_dim)
    model.lm_head = torch.nn.Linear(model.embed_dim, new_vocab_size, bias=False)
    
    # Copy old weights
    n_copy = min(old_enc.weight.shape[0], new_vocab_size)
    with torch.no_grad():
        model.encoder_embed.weight[:n_copy] = old_enc.weight[:n_copy]
        model.decoder_embed.weight[:n_copy] = old_dec.weight[:n_copy]
        model.lm_head.weight[:n_copy] = old_head.weight[:n_copy]
        
    # Update config
    model.vocab_size = new_vocab_size
    
    # Save the fixed model
    model.save(args.model_path + "/pytorch_model.bin")
    
    # Update config.json
    import json
    with open(args.model_path + "/config.json", "r") as f:
        config = json.load(f)
    config["vocab_size"] = new_vocab_size
    with open(args.model_path + "/config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print("✓ Model resized and saved. You can now run finet.py")
else:
    print("✓ Model size already matches tokenizer. No changes needed.")