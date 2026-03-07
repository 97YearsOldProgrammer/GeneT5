import torch
import torch.nn             as nn
import json

from pathlib    import Path
from lib.blocks import Encoder


class GeneT5(nn.Module):

    def __init__(
        self,
        embed_dim   = 768,
        num_layers  = 12,
        num_heads   = 12,
        ff_dim      = 384,
        dropout     = 0.1,
        use_alibi   = True,
        use_moe     = True,
        num_experts = 16,
        moe_top_k   = 2,
        num_kv_heads = None,
        vocab_size  = 4096,
        tie_weights = True,
    ):
        super().__init__()

        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size

        # Single embedding layer
        self.embed         = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Bidirectional transformer stack
        self.encoder = Encoder(
            num_layers   = num_layers,
            embed_dim    = embed_dim,
            num_heads    = num_heads,
            ff_dim       = ff_dim,
            dropout      = dropout,
            attn_dropout = dropout,
            use_alibi    = use_alibi,
            use_moe      = use_moe,
            num_experts  = num_experts,
            moe_top_k    = moe_top_k,
            num_kv_heads = num_kv_heads,
        )

        # LM Head (predicts unmasked tokens for diffusion)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, cu_seqlens=None, max_seqlen=None):

        x = self.embed(input_ids)
        x = self.embed_dropout(x)

        x, moe = self.encoder(x, cu_seqlens, max_seqlen)
        logits = self.lm_head(x)

        return {"logits": logits, "moe_loss": moe}

    def get_param_stats(self):

        emb_train     = sum(p.numel() for p in self.embed.parameters() if p.requires_grad)
        enc_train     = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_train    = sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)

        return {
            "embed_trainable":   emb_train,
            "encoder_trainable": enc_train,
            "head_trainable":    head_train,
            "total_trainable":   emb_train + enc_train + head_train,
            "total_frozen":      0,
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
    def from_pretrained(cls, checkpoint_dir, device="cpu", dtype=torch.float32):
        """Load model from checkpoint directory"""

        path = Path(checkpoint_dir)

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        model = cls(
            embed_dim    = config["embed_dim"],
            num_layers   = config["num_layers"],
            num_heads    = config["num_heads"],
            ff_dim       = config["ff_dim"],
            dropout      = config.get("dropout", 0.1),
            use_alibi    = config.get("use_alibi", True),
            use_moe      = config.get("use_moe", True),
            num_experts  = config.get("num_experts", 8),
            moe_top_k    = config.get("moe_top_k", 2),
            num_kv_heads = config.get("num_kv_heads"),
            vocab_size   = config["vocab_size"],
            tie_weights  = config["tie_weights"],
        )

        model.load(path / "pytorch_model.bin")
        model = model.to(device=device, dtype=dtype)

        print(f"Loaded GeneT5 from {path} (dtype={dtype})")
        stats = model.get_param_stats()
        print(f"  Total params: {stats['total_trainable']:,}")

        return model
