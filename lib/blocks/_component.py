import torch.nn as nn

from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.ops.geglu             import LigerGELUMulFunction
from liger_kernel.ops.swiglu            import LigerSiLUMulFunction


######################
####  Layer Norm  ####
######################


LayerNorm = LigerRMSNorm


########################
####  Feed Forward  ####
########################


_FUSED_ACT = {
    'gelu_new': LigerGELUMulFunction,
    'gelu':     LigerGELUMulFunction,
    'silu':     LigerSiLUMulFunction,
}


class FeedForward(nn.Module):
    """Gated MLP with fused activation kernels"""

    def __init__(self, embed_dim, ff_dim, dropout=0.0, activation='gelu_new'):

        super().__init__()

        self.wi_0      = nn.Linear(embed_dim, ff_dim, bias=False)
        self.wi_1      = nn.Linear(embed_dim, ff_dim, bias=False)
        self.wo        = nn.Linear(ff_dim, embed_dim, bias=False)
        self.dropout   = nn.Dropout(dropout)
        self._fused_act = _FUSED_ACT[activation]

    def forward(self, x):

        hidden = self._fused_act.apply(self.wi_0(x), self.wi_1(x))
        hidden = self.dropout(hidden)
        return self.wo(hidden)
