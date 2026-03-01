import torch
import torch.nn             as nn
import json

from pathlib    import Path
from lib.blocks import Decoder

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss


class GeneT5(nn.Module):

    def __init__(
        self,
        embed_dim   = 768,
        num_layers  = 12,
        num_heads   = 12,
        ff_dim      = 3072,
        dropout     = 0.1,
        use_alibi   = True,
        use_moe     = True,
        num_experts = 8,
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

        # Bidirectional encoder stack
        self.decoder = Decoder(
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

        # LM Head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.embed.weight

        # Loss function
        self.loss_fct = LigerCrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None):

        x      = self.embed(input_ids)
        x      = self.embed_dropout(x)
        x, moe = self.decoder(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
            if moe is not None:
                loss = loss + moe

        return {"logits": logits, "loss": loss, "moe_loss": moe}

    def get_param_stats(self):

        emb_train     = sum(p.numel() for p in self.embed.parameters() if p.requires_grad)
        dec_train     = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        head_train    = sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)

        return {
            "embed_trainable":   emb_train,
            "decoder_trainable": dec_train,
            "head_trainable":    head_train,
            "total_trainable":   emb_train + dec_train + head_train,
            "total_frozen":      0,
        }

    def save(self, save_path):

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "embed":   self.embed.state_dict(),
            "decoder": self.decoder.state_dict(),
            "lm_head": self.lm_head.state_dict(),
        }

        torch.save(ckpt, path)
        print(f"Saved to {path}")

    def load(self, checkpoint_path, strict=False):

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.embed.load_state_dict(ckpt["embed"], strict=strict)
        self.decoder.load_state_dict(ckpt["decoder"], strict=strict)
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
