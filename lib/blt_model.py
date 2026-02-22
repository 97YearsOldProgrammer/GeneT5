import json

import torch
import torch._dynamo
import torch.nn             as nn
import torch.nn.functional  as F

from pathlib import Path

from lib.blocks                import Decoder
from lib.blocks.local_encoder  import LocalEncoder
from lib.blocks.local_decoder  import LocalDecoder
from lib.blocks._scatter_ops   import patch_ids_from_boundaries

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss


class GeneBLT(nn.Module):

    def __init__(
        self,
        byte_vocab_size   = 14,
        local_dim         = 256,
        global_dim        = 768,
        local_num_layers  = 4,
        local_num_heads   = 4,
        local_ff_dim      = 1024,
        global_num_layers = 12,
        global_num_heads  = 12,
        global_ff_dim     = 3072,
        patch_size        = 8,
        enc_window_size   = (256, 256),
        dropout           = 0.1,
        use_moe           = True,
        num_experts       = 8,
        moe_top_k         = 2,
        num_kv_heads      = None,
        ngram_sizes       = tuple(range(3, 21)),
        hash_table_size   = 4096,
        min_patch_size    = 4,
        max_patch_size    = 32,
        target_avg_patch_size = 8,
    ):
        super().__init__()

        self.byte_vocab_size      = byte_vocab_size
        self.global_dim           = global_dim
        self.target_avg_patch_size = target_avg_patch_size

        self.local_encoder = LocalEncoder(
            byte_vocab_size = byte_vocab_size,
            local_dim       = local_dim,
            global_dim      = global_dim,
            num_layers      = local_num_layers,
            num_heads       = local_num_heads,
            ff_dim          = local_ff_dim,
            patch_size      = patch_size,
            window_size     = enc_window_size,
            dropout         = dropout,
            ngram_sizes     = ngram_sizes,
            hash_table_size = hash_table_size,
            min_patch_size  = min_patch_size,
            max_patch_size  = max_patch_size,
        )

        self.global_transformer = Decoder(
            num_layers   = global_num_layers,
            embed_dim    = global_dim,
            num_heads    = global_num_heads,
            ff_dim       = global_ff_dim,
            dropout      = dropout,
            attn_dropout = dropout,
            use_alibi    = True,
            use_moe      = use_moe,
            num_experts  = num_experts,
            moe_top_k    = moe_top_k,
            num_kv_heads = num_kv_heads,
        )

        self.local_decoder = LocalDecoder(
            byte_vocab_size = byte_vocab_size,
            local_dim       = local_dim,
            global_dim      = global_dim,
            num_layers      = local_num_layers,
            num_heads       = local_num_heads,
            ff_dim          = local_ff_dim,
            patch_size      = patch_size,
            dropout         = dropout,
        )

        self.loss_fct = LigerCrossEntropyLoss(ignore_index=-100)

    def forward(
        self, input_bytes, target_bytes, labels=None,
        input_patch_ids=None, target_patch_ids=None,
    ):
        """
        input_bytes:      [B, L_in]  byte IDs
        target_bytes:     [B, L_out] byte IDs
        labels:           [B, L_out - 1] byte-level labels (-100 on padding)
        input_patch_ids:  [B, L_in]  optional precomputed patch assignments
        target_patch_ids: [B, L_out] optional precomputed patch assignments
        """

        input_patches, input_pids, input_blog = self.local_encoder(
            input_bytes, patch_ids=input_patch_ids,
        )
        target_patches, target_pids, target_blog = self.local_encoder(
            target_bytes, patch_ids=target_patch_ids,
        )

        P_in = input_patches.shape[1]

        all_patches = torch.cat([input_patches, target_patches], dim=1)
        torch._dynamo.mark_dynamic(all_patches, 0)
        torch._dynamo.mark_dynamic(all_patches, 1)

        global_out, moe_loss = self.global_transformer(
            all_patches, prefix_len=P_in,
        )

        target_out = global_out[:, P_in:]

        target_dec_pids = target_pids[:, :-1]
        logits = self.local_decoder(target_out, target_bytes[:, :-1], target_dec_pids)

        loss = None
        if labels is not None:
            loss = self.loss_fct(
                logits.reshape(-1, self.byte_vocab_size),
                labels.reshape(-1),
            )
            if moe_loss is not None:
                loss = loss + moe_loss

        return {
            "logits":                  logits,
            "loss":                    loss,
            "moe_loss":                moe_loss,
            "input_patch_ids":         input_pids,
            "target_patch_ids":        target_pids,
            "input_boundary_logits":   input_blog,
            "target_boundary_logits":  target_blog,
        }

    @torch.no_grad()
    def generate(
        self,
        input_bytes,
        max_length   = 5000,
        temperature  = 1.0,
        top_k        = 50,
        top_p        = 0.9,
        eos_token_id = 7,
        pad_token_id = 0,
    ):
        """Two-level autoregressive generation with dynamic patches"""

        was_training = self.training
        self.eval()

        try:
            device = input_bytes.device
            B      = input_bytes.size(0)

            input_patches, input_pids, _ = self.local_encoder(input_bytes)
            P_in = input_patches.shape[1]

            generated = []
            finished  = torch.zeros(B, dtype=torch.bool, device=device)

            target_start     = torch.full((B, 1), pad_token_id, dtype=torch.long, device=device)
            all_target_bytes = target_start

            while len(generated) < max_length:
                if finished.all():
                    break

                target_patches, target_pids, _ = self.local_encoder(all_target_bytes)
                all_patches = torch.cat([input_patches, target_patches], dim=1)

                global_out, _ = self.global_transformer(
                    all_patches, prefix_len=P_in,
                )

                target_out = global_out[:, P_in:]

                last_pids = target_pids[:, -1:]
                logits    = self.local_decoder(target_out, all_target_bytes[:, -1:], last_pids)
                next_logit = logits[:, -1, :]

                if temperature != 1.0:
                    next_logit = next_logit / temperature

                if top_k > 0:
                    k      = min(top_k, next_logit.size(-1))
                    thresh = torch.topk(next_logit, k)[0][..., -1, None]
                    next_logit[next_logit < thresh] = float('-inf')

                probs      = F.softmax(next_logit, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                next_token[finished] = pad_token_id
                generated.append(next_token)
                all_target_bytes = torch.cat([all_target_bytes, next_token], dim=1)

                finished = finished | (next_token.squeeze(-1) == eos_token_id)

            if generated:
                return torch.cat(generated, dim=1)
            return torch.empty(B, 0, dtype=torch.long, device=device)

        finally:
            self.train(was_training)

    def get_param_stats(self):

        enc_train    = sum(p.numel() for p in self.local_encoder.parameters()      if p.requires_grad)
        global_train = sum(p.numel() for p in self.global_transformer.parameters() if p.requires_grad)
        dec_train    = sum(p.numel() for p in self.local_decoder.parameters()       if p.requires_grad)
        total_train  = enc_train + global_train + dec_train

        enc_frozen    = sum(p.numel() for p in self.local_encoder.parameters()      if not p.requires_grad)
        global_frozen = sum(p.numel() for p in self.global_transformer.parameters() if not p.requires_grad)
        dec_frozen    = sum(p.numel() for p in self.local_decoder.parameters()       if not p.requires_grad)
        total_frozen  = enc_frozen + global_frozen + dec_frozen

        return {
            "local_encoder_trainable":      enc_train,
            "global_transformer_trainable": global_train,
            "local_decoder_trainable":      dec_train,
            "total_trainable":              total_train,
            "total_frozen":                 total_frozen,
        }

    def save(self, save_path):

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "local_encoder":      self.local_encoder.state_dict(),
            "global_transformer": self.global_transformer.state_dict(),
            "local_decoder":      self.local_decoder.state_dict(),
        }

        torch.save(ckpt, path)
        print(f"Saved GeneBLT to {path}")

    def load(self, checkpoint_path, strict=False):

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.local_encoder.load_state_dict(ckpt["local_encoder"], strict=strict)
        self.global_transformer.load_state_dict(ckpt["global_transformer"], strict=strict)
        self.local_decoder.load_state_dict(ckpt["local_decoder"], strict=strict)
        print(f"Loaded GeneBLT from {checkpoint_path}")

    @classmethod
    def from_pretrained(cls, checkpoint_dir, device="cpu", dtype=torch.float32):
        """Load GeneBLT from checkpoint directory with blt_config.json"""

        path = Path(checkpoint_dir)

        with open(path / "blt_config.json", "r") as f:
            config = json.load(f)

        model = cls(
            byte_vocab_size       = config.get("byte_vocab_size", 14),
            local_dim             = config.get("local_dim", 256),
            global_dim            = config.get("global_dim", 768),
            local_num_layers      = config.get("local_num_layers", 4),
            local_num_heads       = config.get("local_num_heads", 4),
            local_ff_dim          = config.get("local_ff_dim", 1024),
            global_num_layers     = config.get("global_num_layers", 12),
            global_num_heads      = config.get("global_num_heads", 12),
            global_ff_dim         = config.get("global_ff_dim", 3072),
            patch_size            = config.get("patch_size", 8),
            enc_window_size       = tuple(config.get("enc_window_size", [256, 256])),
            dropout               = config.get("dropout", 0.1),
            use_moe               = config.get("use_moe", True),
            num_experts           = config.get("num_experts", 8),
            moe_top_k             = config.get("moe_top_k", 2),
            num_kv_heads          = config.get("num_kv_heads"),
            ngram_sizes           = tuple(config.get("ngram_sizes", list(range(3, 21)))),
            hash_table_size       = config.get("hash_table_size", 4096),
            min_patch_size        = config.get("min_patch_size", 4),
            max_patch_size        = config.get("max_patch_size", 32),
            target_avg_patch_size = config.get("target_avg_patch_size", 8),
        )

        model.load(path / "pytorch_model.bin")
        model = model.to(device=device, dtype=dtype)

        print(f"Loaded GeneBLT from {path} (dtype={dtype})")
        stats = model.get_param_stats()
        print(f"  Total params: {stats['total_trainable'] + stats['total_frozen']:,}")
        print(f"  Trainable:    {stats['total_trainable']:,}")

        return model
