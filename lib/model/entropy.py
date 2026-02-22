import torch
import torch.nn as nn

from lib.blocks.encoder import Encoder


class DNAEntropyModel(nn.Module):
    """Byte-level DNA language model for scoring per-position complexity

    Small causal LM trained on next-byte prediction. Entropy of the
    predicted distribution indicates local sequence complexity — high
    entropy at splice sites, low in repetitive introns
    """

    def __init__(
        self,
        vocab_size  = 14,
        dim         = 256,
        num_layers  = 4,
        num_heads   = 4,
        ff_dim      = 1024,
        window_size = 512,
        dropout     = 0.0,
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.dim        = dim

        self.embed = nn.Embedding(vocab_size, dim)

        self.encoder = Encoder(
            num_layers  = num_layers,
            embed_dim   = dim,
            num_heads   = num_heads,
            ff_dim      = ff_dim,
            dropout     = dropout,
            use_alibi   = True,
            window_size = (window_size, 0),
        )

        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, byte_ids):
        """byte_ids: [B, L] -> logits: [B, L, vocab_size]"""

        x = self.embed(byte_ids)
        x = self.encoder(x)
        return self.head(x)

    @torch.no_grad()
    def compute_entropy(self, byte_ids):
        """Per-position entropy of next-byte prediction"""

        logits  = self.forward(byte_ids)
        probs   = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy

    @torch.no_grad()
    def compute_boundaries(self, byte_ids, threshold=1.5, min_patch=4, max_patch=None):
        """Predict patch boundaries from entropy scores

        Returns patch_ids [B, L] assigning each byte to a patch
        """

        entropy = self.compute_entropy(byte_ids)
        B, L    = byte_ids.shape
        device  = byte_ids.device

        boundary_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        boundary_mask[:, 0] = True

        high_entropy  = entropy > threshold
        boundary_mask = boundary_mask | high_entropy

        boundary_mask = _enforce_patch_constraints(boundary_mask, min_patch, max_patch)
        patch_ids     = boundary_mask.long().cumsum(dim=1) - 1

        return patch_ids

    def get_param_count(self):

        return sum(p.numel() for p in self.parameters())

    def save(self, path):

        torch.save(self.state_dict(), path)

    def load(self, path, device="cpu"):

        state = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state)


def _enforce_patch_constraints(boundary_mask, min_patch, max_patch):
    """Enforce min/max patch size constraints on boundary predictions"""

    B, L   = boundary_mask.shape
    device = boundary_mask.device
    result = torch.zeros_like(boundary_mask)

    for b in range(B):
        last_boundary = 0
        result[b, 0]  = True

        for i in range(1, L):
            dist = i - last_boundary

            if max_patch is not None and dist >= max_patch:
                result[b, i]  = True
                last_boundary = i
            elif dist >= min_patch and boundary_mask[b, i]:
                result[b, i]  = True
                last_boundary = i

    return result
