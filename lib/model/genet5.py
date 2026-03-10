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
        ff_dim         = 1536,
        dense_ff_dim   = None,
        dropout        = 0.1,
        use_moe        = True,
        moe_layers     = None,
        num_experts    = 8,
        moe_top_k      = 2,
        num_kv_heads   = None,
        vocab_size     = 4096,
        tie_weights    = True,
        fused_add_norm = False,
    ):
        super().__init__()

        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size

        # Single embedding layer
        self.embed         = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Bidirectional transformer stack
        self.encoder = Encoder(
            num_layers     = num_layers,
            embed_dim      = embed_dim,
            num_heads      = num_heads,
            ff_dim         = ff_dim,
            dense_ff_dim   = dense_ff_dim,
            dropout        = dropout,
            attn_dropout   = dropout,
            use_moe        = use_moe,
            moe_layers     = moe_layers,
            num_experts    = num_experts,
            moe_top_k      = moe_top_k,
            num_kv_heads   = num_kv_heads,
            fused_add_norm = fused_add_norm,
        )

        # LM Head (predicts unmasked tokens for diffusion)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, cu_seqlens=None, max_seqlen=None, return_hidden=False):

        x = self.embed(input_ids)
        x = self.embed_dropout(x)

        x, moe = self.encoder(x, cu_seqlens, max_seqlen)

        if return_hidden:
            return {"hidden": x, "moe_loss": moe}

        logits = self.lm_head(x)
        return {"logits": logits, "moe_loss": moe}

    def get_expert_frac(self):
        """Average per-expert load fraction across all MoE layers"""

        fracs = []
        for layer in self.encoder.layers:
            if hasattr(layer.ff, '_last_expert_frac'):
                fracs.append(layer.ff._last_expert_frac)
        if not fracs:
            return None
        return torch.stack(fracs).mean(dim=0)

    def freeze_base(self):
        """Freeze pretrained base, keep MoE experts/routers + embed/head trainable"""

        for p in self.parameters():
            p.requires_grad = False

        # Unfreeze MoE expert weights and routers
        for layer in self.encoder.layers:
            if layer.use_moe:
                for p in layer.ff.parameters():
                    p.requires_grad = True

        # Unfreeze embed + LM head (needed for new GFF tokens)
        for p in self.embed.parameters():
            p.requires_grad = True
        for p in self.lm_head.parameters():
            p.requires_grad = True

    def get_param_stats(self):

        total     = sum(p.numel() for p in self.parameters())
        emb_train = sum(p.numel() for p in self.embed.parameters() if p.requires_grad)
        enc_train = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_train = sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)
        trainable = emb_train + enc_train + head_train

        return {
            "embed_trainable":   emb_train,
            "encoder_trainable": enc_train,
            "head_trainable":    head_train,
            "total_trainable":   trainable,
            "total_frozen":      total - trainable,
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

        # Backward compat: old checkpoints used "decoder" key
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
            dense_ff_dim   = config.get("dense_ff_dim"),
            dropout        = config.get("dropout", 0.1),
            use_moe        = config.get("use_moe", True),
            moe_layers     = config.get("moe_layers"),
            num_experts    = config.get("num_experts", 8),
            moe_top_k      = config.get("moe_top_k", 2),
            num_kv_heads   = config.get("num_kv_heads"),
            vocab_size     = config["vocab_size"],
            tie_weights    = config["tie_weights"],
            fused_add_norm = fused_add_norm,
        )

        model.load(path / "pytorch_model.bin")
        model = model.to(device=device, dtype=dtype)

        print(f"Loaded GeneT5 from {path} (dtype={dtype})")
        stats = model.get_param_stats()
        total = stats['total_trainable'] + stats['total_frozen']
        print(f"  Total params: {total:,}")

        return model
