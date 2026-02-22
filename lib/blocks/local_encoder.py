import torch
import torch.nn as nn

from lib.blocks.encoder      import Encoder
from lib.blocks._scatter_ops import scatter_pool, patch_ids_from_boundaries, enforce_patch_constraints

_HASH_PRIME = 31


class NGramEmbedding(nn.Module):
    """Rolling polynomial hash n-gram embeddings (BLT paper design)

    For each position i, computes backward n-gram g = {b[i-n+1], ..., b[i]}
    Hashes via rolling polynomial: hash(g) = sum(b[j] * prime^j) % table_size
    Non-DNA positions (special tokens) get a dedicated mixed bucket
    """

    def __init__(self, embed_dim, sizes=tuple(range(3, 21)), hash_table_size=4096):

        super().__init__()

        self.sizes           = sizes
        self.hash_table_size = hash_table_size

        self.tables = nn.ModuleList([
            nn.Embedding(hash_table_size + 1, embed_dim) for _ in sizes
        ])

        self._num_sizes_plus_1 = len(sizes) + 1

    def _hash_ngrams(self, byte_ids, n):
        """Compute backward n-gram hash IDs at each position

        At position i, the n-gram is bytes [i-n+1, ..., i] (backward)
        Positions 0..n-2 can't form a full backward n-gram -> mixed bucket
        Uses iterative modular hashing to avoid integer overflow
        """

        device   = byte_ids.device
        B, L     = byte_ids.shape
        mixed_id = self.hash_table_size
        tbl_size = self.hash_table_size

        dna_map    = torch.full((14,), -1, dtype=torch.long, device=device)
        dna_map[1] = 0
        dna_map[2] = 1
        dna_map[3] = 2
        dna_map[4] = 3

        dna_idx = dna_map[byte_ids]
        result  = torch.full((B, L), mixed_id, dtype=torch.long, device=device)

        if L < n:
            return result

        valid_L = L - n + 1
        safe    = dna_idx.clamp(min=0)

        h   = safe[..., :valid_L]
        bad = (dna_idx[..., :valid_L] == -1)

        for j in range(1, n):
            next_safe = safe[..., j:j + valid_L]
            bad       = bad | (dna_idx[..., j:j + valid_L] == -1)
            h         = (h * _HASH_PRIME + next_safe) % tbl_size

        h[bad]           = mixed_id
        result[..., n-1:] = h
        return result

    def forward(self, byte_ids):
        """Sum hash-based n-gram embeddings, normalize by (num_sizes + 1)"""

        total = torch.zeros(
            *byte_ids.shape, self.tables[0].embedding_dim,
            device=byte_ids.device, dtype=self.tables[0].weight.dtype,
        )

        for n, table in zip(self.sizes, self.tables):
            ids    = self._hash_ngrams(byte_ids, n)
            total += table(ids)

        return total / self._num_sizes_plus_1


class LocalEncoder(nn.Module):
    """Compress byte sequences to patches via windowed attention + dynamic boundaries"""

    def __init__(
        self,
        byte_vocab_size  = 14,
        local_dim        = 256,
        global_dim       = 768,
        num_layers       = 4,
        num_heads        = 4,
        ff_dim           = 1024,
        patch_size       = 8,
        window_size      = (256, 256),
        dropout          = 0.0,
        ngram_sizes      = tuple(range(3, 21)),
        hash_table_size  = 4096,
        min_patch_size   = 4,
        max_patch_size   = 32,
    ):
        super().__init__()

        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size

        self.byte_embed  = nn.Embedding(byte_vocab_size, local_dim)
        self.ngram_embed = NGramEmbedding(local_dim, ngram_sizes, hash_table_size)

        self.encoder = Encoder(
            num_layers  = num_layers,
            embed_dim   = local_dim,
            num_heads   = num_heads,
            ff_dim      = ff_dim,
            dropout     = dropout,
            use_alibi   = True,
            window_size = window_size,
        )

        self.boundary_head = nn.Linear(local_dim, 1)
        self.patch_proj    = nn.Linear(local_dim, global_dim, bias=False)

    def forward(self, byte_ids, patch_ids=None):
        """byte_ids: [B, L] -> (patches, patch_ids, boundary_logits)"""

        byte_emb  = self.byte_embed(byte_ids)
        ngram_emb = self.ngram_embed(byte_ids)

        x = byte_emb / self.ngram_embed._num_sizes_plus_1 + ngram_emb
        x = self.encoder(x)

        boundary_logits = self.boundary_head(x).squeeze(-1)

        if patch_ids is None:
            boundary_probs = torch.sigmoid(boundary_logits)
            boundary_mask  = boundary_probs > 0.5
            boundary_mask[:, 0] = True
            boundary_mask  = enforce_patch_constraints(
                boundary_mask, self.min_patch_size, self.max_patch_size,
            )
            patch_ids = patch_ids_from_boundaries(boundary_mask)

        num_patches = patch_ids.max(dim=1).values + 1
        max_patches = num_patches.max().item()

        patches = scatter_pool(x, patch_ids, max_patches)
        patches = self.patch_proj(patches)

        return patches, patch_ids, boundary_logits
