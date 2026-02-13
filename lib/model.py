import torch
import torch.nn             as nn
import torch.nn.functional  as F
import json

from pathlib    import Path
from lib.blocks import Encoder, Decoder

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss


class GeneT5(nn.Module):

    def __init__(
        self,
        embed_dim            = 768,
        encoder_num_layers   = 12,
        encoder_num_heads    = 12,
        encoder_ff_dim       = 3072,
        decoder_num_layers   = 12,
        decoder_num_heads    = 12,
        decoder_ff_dim       = 3072,
        decoder_dropout      = 0.1,
        decoder_use_alibi    = True,
        decoder_use_moe      = True,
        decoder_num_experts  = 8,
        decoder_moe_top_k    = 2,
        decoder_num_kv_heads = None,
        vocab_size           = 4096,
        tie_weights          = True,
        encoder_window_size  = 1024,
        decoder_window_size  = 256,
    ):
        super().__init__()

        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size

        if decoder_num_kv_heads is None:
            decoder_num_kv_heads = max(1, decoder_num_heads // 4)

        self.encoder_window_size = encoder_window_size
        self.decoder_window_size = decoder_window_size

        # Encoder embedding
        self.encoder_embed         = nn.Embedding(vocab_size, embed_dim)
        self.encoder_embed_dropout = nn.Dropout(decoder_dropout)

        # Encoder (flash_attn sliding window)
        self.encoder = Encoder(
            num_layers   = encoder_num_layers,
            embed_dim    = embed_dim,
            num_heads    = encoder_num_heads,
            ff_dim       = encoder_ff_dim,
            dropout      = decoder_dropout,
            attn_dropout = decoder_dropout,
            use_alibi    = True,
            window_size  = encoder_window_size,
        )

        # Decoder
        self.decoder = Decoder(
            num_layers   = decoder_num_layers,
            embed_dim    = embed_dim,
            num_heads    = decoder_num_heads,
            ff_dim       = decoder_ff_dim,
            dropout      = decoder_dropout,
            attn_dropout = decoder_dropout,
            use_alibi    = decoder_use_alibi,
            use_moe      = decoder_use_moe,
            num_experts  = decoder_num_experts,
            moe_top_k    = decoder_moe_top_k,
            num_kv_heads = decoder_num_kv_heads,
            window_size  = decoder_window_size,
        )

        # Decoder embeddings
        self.decoder_embed         = nn.Embedding(vocab_size, embed_dim)
        self.decoder_embed_dropout = nn.Dropout(decoder_dropout)

        # LM Head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.decoder_embed.weight

        # Loss function (reused across forward calls)
        self.loss_fct = LigerCrossEntropyLoss(ignore_index=-100)

    def encode(self, encoder_input_ids):
        """Encode input sequence"""

        encoder_embeds = self.encoder_embed(encoder_input_ids)
        encoder_embeds = self.encoder_embed_dropout(encoder_embeds)
        encoder_hidden = self.encoder(encoder_embeds)

        return encoder_hidden

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        labels                 = None,
        decoder_attention_mask = None,
        encoder_hidden_states  = None,
    ):
        # Encode (or use cached encoder output)
        if encoder_hidden_states is None:
            encoder_hidden_states = self.encode(encoder_input_ids)

        # Create decoder attention mask from padding if not provided
        if decoder_attention_mask is None and decoder_input_ids is not None:
            decoder_attention_mask = (decoder_input_ids >= 0) & (decoder_input_ids < self.vocab_size)
            decoder_attention_mask = decoder_attention_mask.long()

        # Replace invalid indices with pad token (0) for embedding lookup
        decoder_input_ids_safe = decoder_input_ids.clamp(0, self.vocab_size - 1)

        # Embed decoder inputs
        decoder_embeds = self.decoder_embed(decoder_input_ids_safe)
        decoder_embeds = self.decoder_embed_dropout(decoder_embeds)

        # Decode with cross-attention to encoder output
        decoder_output, moe_loss = self.decoder(
            hidden_states          = decoder_embeds,
            encoder_hidden_states  = encoder_hidden_states,
            attention_mask         = decoder_attention_mask,
            encoder_attention_mask = None,
        )

        # Project to vocab
        logits = self.lm_head(decoder_output)

        # Compute loss (Liger fused kernel: 2x faster)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
            if moe_loss is not None:
                loss = loss + moe_loss

        outputs = {
            "logits":         logits,
            "loss":           loss,
            "moe_loss":       moe_loss,
            "encoder_hidden": encoder_hidden_states,
        }

        return outputs

    @torch.no_grad()
    def generate_from_hidden(
        self,
        encoder_hidden,
        max_length   = 100,
        temperature  = 1.0,
        top_k        = 50,
        top_p        = 0.9,
        bos_token_id = 1,
        eos_token_id = 2,
        pad_token_id = 0
    ):
        """Autoregressive generation from pre-computed encoder hidden states"""

        was_training = self.training
        self.eval()
        device = encoder_hidden.device
        batch  = encoder_hidden.size(0)

        try:
            return self._generate_loop(
                encoder_hidden, batch, device, max_length,
                temperature, top_k, top_p, bos_token_id, eos_token_id, pad_token_id
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate(
        self,
        encoder_input_ids,
        max_length   = 100,
        temperature  = 1.0,
        top_k        = 50,
        top_p        = 0.9,
        bos_token_id = 1,
        eos_token_id = 2,
        pad_token_id = 0
    ):
        """Autoregressive generation"""

        was_training = self.training
        self.eval()
        device = encoder_input_ids.device
        batch  = encoder_input_ids.size(0)

        encoder_hidden = self.encode(encoder_input_ids)

        try:
            return self._generate_loop(
                encoder_hidden, batch, device, max_length,
                temperature, top_k, top_p, bos_token_id, eos_token_id, pad_token_id
            )
        finally:
            self.train(was_training)

    def _generate_loop(
        self, encoder_hidden, batch, device, max_length,
        temperature, top_k, top_p, bos_token_id, eos_token_id, pad_token_id
    ):
        """Shared autoregressive generation loop"""

        generated = torch.full((batch, 1), bos_token_id, dtype=torch.long, device=device)
        finished  = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_length):
            outputs = self.forward(
                encoder_input_ids     = None,
                decoder_input_ids     = generated,
                encoder_hidden_states = encoder_hidden,
            )

            logits = outputs["logits"][:, -1, :]

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-K
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-P
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token[finished] = pad_token_id
            generated = torch.cat([generated, next_token], dim=1)

            # Check EOS
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        return generated

    def get_param_stats(self):

        enc_emb_train = sum(p.numel() for p in self.encoder_embed.parameters() if p.requires_grad)
        enc_train     = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        enc_frozen    = sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad)
        dec_train     = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        emb_train     = sum(p.numel() for p in self.decoder_embed.parameters() if p.requires_grad)
        head_train    = sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)

        return {
            "encoder_embed_trainable": enc_emb_train,
            "encoder_trainable":       enc_train,
            "encoder_frozen":          enc_frozen,
            "decoder_trainable":       dec_train,
            "embed_trainable":         emb_train,
            "head_trainable":          head_train,
            "total_trainable":         enc_emb_train + enc_train + dec_train + emb_train + head_train,
            "total_frozen":            enc_frozen
        }

    def freeze_encoder(self):
        for param in self.encoder_embed.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder_embed.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True

    def save(self, save_path):

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "encoder_embed": self.encoder_embed.state_dict(),
            "encoder":       self.encoder.state_dict(),
            "decoder":       self.decoder.state_dict(),
            "decoder_embed": self.decoder_embed.state_dict(),
            "lm_head":       self.lm_head.state_dict()
        }

        torch.save(ckpt, path)
        print(f"Saved to {path}")

    def load(self, checkpoint_path, strict=False):

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "encoder_embed" in ckpt:
            self.encoder_embed.load_state_dict(ckpt["encoder_embed"], strict=strict)
        self.encoder.load_state_dict(ckpt["encoder"], strict=strict)
        if "compressor" in ckpt:
            print(f"  Warning: skipping perceiver compressor weights (removed)")
        self.decoder.load_state_dict(ckpt["decoder"], strict=strict)
        self.decoder_embed.load_state_dict(ckpt["decoder_embed"], strict=strict)
        self.lm_head.load_state_dict(ckpt["lm_head"], strict=strict)
        print(f"Loaded from {checkpoint_path}")

    @classmethod
    def from_pretrained(cls, checkpoint_dir, device="cpu", dtype=torch.float32):
        """Load model from checkpoint directory"""

        path = Path(checkpoint_dir)

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        model = cls(
            embed_dim            = config["embed_dim"],
            encoder_num_layers   = config["encoder_num_layers"],
            encoder_num_heads    = config["encoder_num_heads"],
            encoder_ff_dim       = config["encoder_ff_dim"],
            decoder_num_layers   = config["decoder_num_layers"],
            decoder_num_heads    = config["decoder_num_heads"],
            decoder_ff_dim       = config["decoder_ff_dim"],
            decoder_dropout      = config["decoder_dropout"],
            decoder_use_alibi    = config["decoder_use_alibi"],
            decoder_use_moe      = config["decoder_use_moe"],
            decoder_num_experts  = config.get("decoder_num_experts", 8),
            decoder_moe_top_k    = config.get("decoder_moe_top_k", 2),
            decoder_num_kv_heads = config.get("decoder_num_kv_heads"),
            vocab_size           = config["vocab_size"],
            tie_weights          = config["tie_weights"],
            encoder_window_size  = config.get("encoder_window_size", 1024),
            decoder_window_size  = config.get("decoder_window_size", 256),
        )

        model.load(path / "pytorch_model.bin")
        model = model.to(device=device, dtype=dtype)

        print(f"Loaded GeneT5 from {path} (dtype={dtype})")
        stats = model.get_param_stats()
        print(f"  Total params: {stats['total_trainable'] + stats['total_frozen']:,}")

        return model
