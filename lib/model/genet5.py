import torch
import torch.nn             as nn
import json

from pathlib    import Path
from lib.blocks import Encoder


class GeneT5(nn.Module):

    def __init__(
        self,
        embed_dim      = 768,
        num_layers     = 12,
        num_heads      = 12,
        ff_dim         = 3072,
        dropout        = 0.1,
        num_kv_heads   = None,
        vocab_size     = 4096,
        tie_weights    = True,
        fused_add_norm = False,
        depth_scaling  = False,
    ):
        super().__init__()

        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size

        self.embed         = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.encoder = Encoder(
            num_layers     = num_layers,
            embed_dim      = embed_dim,
            num_heads      = num_heads,
            ff_dim         = ff_dim,
            dropout        = dropout,
            attn_dropout   = dropout,
            num_kv_heads   = num_kv_heads,
            fused_add_norm = fused_add_norm,
            depth_scaling  = depth_scaling,
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, cu_seqlens=None, max_seqlen=None, return_hidden=False):

        x = self.embed(input_ids)
        x = self.embed_dropout(x)
        x = self.encoder(x, cu_seqlens, max_seqlen)

        if return_hidden:
            return {"hidden": x}

        logits = self.lm_head(x)
        return {"logits": logits}

    def get_param_stats(self):

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_params":    total,
            "total_trainable": trainable,
            "total_frozen":    total - trainable,
        }

    def save(self, save_path):

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "embed":   self.embed.state_dict(),
            "encoder": self.encoder.state_dict(),
            "lm_head": self.lm_head.state_dict(),
        }

        torch.save(ckpt, path)
        print(f"Saved to {path}")

    def load(self, checkpoint_path, strict=False):

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        enc_key = "encoder" if "encoder" in ckpt else "decoder"

        self.embed.load_state_dict(ckpt["embed"], strict=strict)
        self.encoder.load_state_dict(ckpt[enc_key], strict=strict)
        self.lm_head.load_state_dict(ckpt["lm_head"], strict=strict)
        print(f"Loaded from {checkpoint_path}")

    @classmethod
    def from_pretrained(cls, checkpoint_dir, device="cpu", dtype=torch.float32,
                        fused_add_norm=False):
        """Load model from checkpoint directory"""

        path = Path(checkpoint_dir)

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        model = cls(
            embed_dim      = config["embed_dim"],
            num_layers     = config["num_layers"],
            num_heads      = config["num_heads"],
            ff_dim         = config["ff_dim"],
            dropout        = config.get("dropout", 0.1),
            num_kv_heads   = config.get("num_kv_heads"),
            vocab_size     = config["vocab_size"],
            tie_weights    = config["tie_weights"],
            fused_add_norm = fused_add_norm,
            depth_scaling  = config.get("depth_scaling", False),
        )

        model.load(path / "pytorch_model.bin")
        model = model.to(device=device, dtype=dtype)

        stats = model.get_param_stats()
        print(f"Loaded GeneT5 from {path} ({stats['total_params']:,} params, dtype={dtype})")

        return model
