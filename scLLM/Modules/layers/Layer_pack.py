"""
inspired by "Towards a Unified View of Parameter-Efficient Transfer Learning
Junxian He*, Chunting Zhou*, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig
ICLR 2022 (spotlight)"

and 

"PARAMETER-EFFICIENT FINE-TUNING DESIGN SPACES"
layer grouping
tunable parameters alocated to each layer
tunable-group created
training strategy

"""

import torch.nn as nn
from functools import partial
from scLLM.Modules.utils import default

from scLLM.Modules.layers.base import BaseLayers
from scLLM.Modules.layers.fastAttention import SelfAttention
from scLLM.Modules.layers.feedForward import FeedForward

from scLLM.Modules.utils import  ReZero, Chunk


class Attention_LayerPack(nn.Module,BaseLayers):
    def __init__(self, dim, 
                 use_scalenorm,
                 use_rezero,
                 **kwargs):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)

        self.dim = dim
        if use_scalenorm:
            self.wrapper_fn = partial(self.ops.PreScaleNorm, dim)
        elif use_rezero:
            self.wrapper_fn = ReZero
        else:
            self.wrapper_fn = partial(self.ops.PreLayerNorm, dim)

    def module_list(self,
                 causal = False,
                 heads = 8,
                 dim_head = 64,
                 local_heads = 0,
                 local_window_size = 256,
                 nb_features = None,
                 generalized_attention = False,
                 kernel_fn = nn.ReLU(),
                 attn_dropout:float = 0.,
                 no_projection = False,
                 qkv_bias = False,

                 ff_glu:bool = False,                     # use GLU (Gated Linear Units) variant for feedforward
                 ff_dropout:float = 0.,                    # feedforward dropout
                 ff_mult:int = 4,                        # dim of intermediate features after attention / dim of input features
                 ff_chunks:int = 1,                      # chunk feedforward layer, from Reformer
                 **kwargs
                 ):
        return    nn.ModuleList([
                self.wrapper_fn(SelfAttention(self.dim, causal = causal, heads = heads, 
                                         dim_head = dim_head, local_heads = local_heads, 
                                         local_window_size = local_window_size, 
                                         nb_features = nb_features, generalized_attention = generalized_attention,
                                         kernel_fn = kernel_fn, dropout = attn_dropout, 
                                         no_projection = no_projection, qkv_bias = qkv_bias,
                                         **kwargs)),
                self.wrapper_fn(Chunk(ff_chunks, FeedForward(self.dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu,
                                                             **kwargs), along_dim = 1))
            ])
    
