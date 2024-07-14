
import torch.nn as nn

from scLLM.Modules.utils import default
from scLLM.Modules.layers.base import BaseLayers


class FeedForward(nn.Module,BaseLayers):
    def __init__(self, dim, mult = 4, dropout = 0., 
                 activation = None, 
                 glu = False,
                 **kwargs):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = self.ops.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = self.ops.Dropout(dropout)
        self.w2 = self.ops.Linear(dim * mult, dim)

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
