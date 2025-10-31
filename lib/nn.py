import gzip
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataclasses import dataclass


###################
####### CNN #######
###################


class IntraNet(nn.Module):
    """
    CNN → BLSTM hybrid for embedded DNA sequences.

    Input
    -----
    x : FloatTensor, shape (B, L, E)
        B = batch size, L = sequence length (k-mer steps), E = embedding dim
    lengths : Optional[LongTensor], shape (B,)
        True (unpadded) lengths per sequence. If provided, we pack for LSTM.

    Flow
    ----
    (B, L, E)
      → unsqueeze channel → (B, 1, L, E)
      → [Conv2d + BN + ReLU + MaxPool2d] × 3  (channels grow; H/W downsample)
      → mean over embedding-axis (width) → (B, C, L')
      → permute to time-major → (B, L', C)
      → BiLSTM → (B, L', 2*H)
      → temporal pooling (mean + max concat) → (B, 4*H)
      → MLP head → logits (B, num_classes)

    Notes
    -----
    - Uses manual 'same' padding for odd kernels to avoid version issues.
    - If `lengths` is passed, it is downscaled through the pooling factors
      and used with pack_padded_sequence for robust masking.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        conv_channels: List[int] = [64, 128, 256],
        kernel_sizes: List[tuple] = [(3, 3), (3, 3), (3, 3)],
        pool_sizes: List[tuple] = [(2, 2), (2, 2), (2, 2)],
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes) == len(pool_sizes), \
            "conv_channels, kernel_sizes, pool_sizes must have same length"

        self.embedding_dim = embedding_dim
        self.pool_sizes = pool_sizes  # keep to rescale lengths along L axis

        # --- Convolutional trunk ---
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = 1  # treat (L, E) grid as single-channel "image"
        for out_ch, ksz, psz in zip(conv_channels, kernel_sizes, pool_sizes):
            pad = self._same_pad(ksz)  # (pad_h, pad_w) for stride=1
            self.convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=ksz, padding=pad, bias=False))
            self.bns.append(nn.BatchNorm2d(out_ch))
            self.pools.append(nn.MaxPool2d(kernel_size=psz))
            in_ch = out_ch

        c_last = conv_channels[-1]

        # --- BLSTM over the (downsampled) sequence axis ---
        self.blstm = nn.LSTM(
            input_size=c_last,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # --- Classifier head ---
        # Temporal pooling uses mean+max over time and concatenates → 4*lstm_hidden
        head_in = 4 * lstm_hidden
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _same_pad(kernel_hw: tuple) -> tuple:
        """Return (pad_h, pad_w) that preserves H/W for stride=1 (odd kernels)."""
        kh, kw = kernel_hw
        # Works best with odd kernels (3,5,7). For even kernels, output shrinks by 1.
        return (kh // 2, kw // 2)

    def _downscale_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Propagate true lengths through the pooling pipeline along the L (height) axis."""
        L = lengths.clone()
        for (ph, pw) in self.pool_sizes:
            # MaxPool2d with default stride=kernel_size halves (ceil) for odd lengths
            L = torch.div(L, ph, rounding_mode='floor')
        # Ensure at least 1
        L = torch.clamp(L, min=1)
        return L

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, L, E), lengths optional (B,)
        returns logits: (B, num_classes)
        """
        B = x.size(0)

        # Add channel → (B, 1, L, E)
        x = x.unsqueeze(1)

        # Conv blocks
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = pool(F.relu(bn(conv(x))))  # (B, C, L', E')

        # Collapse embedding-axis with mean → (B, C, L')
        x = x.mean(dim=3)

        # Prepare for LSTM: (B, L', C)
        x = x.permute(0, 2, 1).contiguous()

        # Optional length-aware packing for BLSTM
        if lengths is not None:
            lens_ds = self._downscale_lengths(lengths.to(x.device))
            # pack
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lens_ds.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.blstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            # out: (B, L', 2*H); pad positions are zeros by default
        else:
            out, _ = self.blstm(x)  # (B, L', 2*H)

        # Temporal pooling to fixed-length feature
        mean_pool = out.mean(dim=1)                # (B, 2*H)
        max_pool  = out.max(dim=1).values          # (B, 2*H)
        feats = torch.cat([mean_pool, max_pool], dim=1)  # (B, 4*H)

        # Head
        logits = self.fc(self.dropout(feats))      # (B, num_classes)
        return logits