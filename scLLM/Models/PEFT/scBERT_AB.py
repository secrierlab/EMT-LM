"""
basic idea is have a selection function to partially select patterns 
that need to be calculated in next layer and eliminate the effect of others
"""
import torch
import torch.nn as nn

from scLLM.Modules.utils import default

############################################################################################################
#               lora version: 0.0.1
############################################################################################################

class FeedForward_active(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def get_active_mask(self, dim):
        self.mask = nn.parameter.Parameter(torch.randn(dim, dim))

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x