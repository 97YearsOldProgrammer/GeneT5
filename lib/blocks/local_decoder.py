import torch.nn as nn

from lib.blocks.decoder      import Decoder
from lib.blocks._blockcross  import CrossAttention, CrossAttentionConfig
from lib.blocks._component   import LayerNorm


class LocalDecoder(nn.Module):
    """Upsample patches to byte resolution with cross-attention and causal decoding"""

    def __init__(
        self,
        byte_vocab_size = 14,
        local_dim       = 256,
        global_dim      = 768,
        num_layers      = 4,
        num_heads       = 4,
        ff_dim          = 1024,
        patch_size      = 8,
        dropout         = 0.0,
    ):
        super().__init__()

        self.patch_size = patch_size

        self.patch_unpool = nn.Linear(global_dim, local_dim, bias=False)
        self.byte_embed   = nn.Embedding(byte_vocab_size, local_dim)

        cross_config = CrossAttentionConfig(
            embed_dim    = local_dim,
            num_heads    = num_heads,
            num_kv_heads = max(1, num_heads // 4),
        )
        self.cross_attn = CrossAttention(cross_config)
        self.cross_norm = LayerNorm(local_dim)

        self.decoder = Decoder(
            num_layers = num_layers,
            embed_dim  = local_dim,
            num_heads  = num_heads,
            ff_dim     = ff_dim,
            dropout    = dropout,
            use_alibi  = True,
            use_moe    = False,
        )

        self.byte_head = nn.Linear(local_dim, byte_vocab_size, bias=False)

    def forward(self, patch_hidden, target_bytes):
        """
        patch_hidden: [B, P, global_dim] from global transformer
        target_bytes: [B, T] teacher-forced byte IDs (already shifted)
        """

        context  = self.patch_unpool(patch_hidden)
        context  = context.repeat_interleave(self.patch_size, dim=1)

        byte_emb = self.byte_embed(target_bytes)

        normed       = self.cross_norm(byte_emb)
        cross_out, _ = self.cross_attn(normed, context)
        byte_emb     = byte_emb + cross_out

        byte_emb, _ = self.decoder(byte_emb)

        return self.byte_head(byte_emb)
