import math
import numpy as np
import torch
from torch import nn
from torch import Tensor
from einops import rearrange, repeat
# sinusoidal positional embeddings
from scLLM import logger
from scLLM.Modules.layers.base import BaseLayers

#########################################################################################
#            rotary positional embeddings
#########################################################################################
# rotary positional embedding helpers for scBERT

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

#########################################################################################
#            absolute positional embeddings
#########################################################################################
# positional embeddings

class AbsolutePositionalEmbedding(nn.Module,BaseLayers):

    def __init__(self, dim, max_seq_len,**kwargs):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        self.emb = self.ops.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)
    



