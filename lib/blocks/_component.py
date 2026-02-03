import torch
import torch.nn as nn


######################
####  Layer Norm  ####
######################


class LayerNorm(nn.Module):
    """RMSNorm without mean centering"""

    def __init__(self, hidden_size, eps=1e-6):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps    = eps

    def forward(self, x):

        variance = x.pow(2).mean(-1, keepdim=True)
        x        = x * torch.rsqrt(variance + self.eps)

        return self.weight * x


########################
####  Feed Forward  ####
########################


class FeedForward(nn.Module):
    """Gated MLP based on GeGLU"""

    def __init__(self, embed_dim, ff_dim, dropout=0.0, activation='gelu_new'):

        super().__init__()

        self.wi_0    = nn.Linear(embed_dim, ff_dim, bias=False)
        self.wi_1    = nn.Linear(embed_dim, ff_dim, bias=False)
        self.wo      = nn.Linear(ff_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        if activation == 'gelu_new':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()

    def forward(self, x):

        hidden = self.act(self.wi_0(x)) * self.wi_1(x)
        hidden = self.dropout(hidden)
        output = self.wo(hidden)

        return output
