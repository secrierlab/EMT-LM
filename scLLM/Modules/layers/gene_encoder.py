"""

"""
import math
import numpy as np
import torch
from torch import nn
from torch import Tensor
from einops import rearrange, repeat
# sinusoidal positional embeddings
from typing import Optional, Tuple
from scLLM import logger
from scLLM.Modules.layers.base import BaseLayers
#########################################################################################
#            gene 2 vec positional embeddings
#########################################################################################
# Gene2Vec used in scBERT model
class Gene2VecPositionalEmbedding(nn.Module,BaseLayers):
    def __init__(self, gene2vec_weight:str=None, max_seq_len:int=16907,**kwargs):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        logger.debug("Gene2VecPositionalEmbedding initialised")
        if gene2vec_weight is not None:
            gene2vec_weight = np.load(gene2vec_weight)
        else:
            max_seq_len = max_seq_len -1 
            # original paper use gene2vec with 16906x200
            # this is only for loading model
            gene2vec_weight = np.random.randn(max_seq_len, 200)
        # shape: (16906+1, 200) added channel in the end
        gene2vec_weight = np.concatenate((gene2vec_weight, 
                                        np.zeros((1, gene2vec_weight.shape[1]))), axis=0)
        gene2vec_weight = torch.from_numpy(gene2vec_weight)
        self.emb = self.ops.Embedding.from_pretrained(gene2vec_weight)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

