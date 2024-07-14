import torch.nn as nn
from typing import Callable
import attr

from scLLM.Models.init import model_para_base

@attr.s(auto_attribs=True)
class scBERT_para(model_para_base):
    model_name:str = "scBERT"                   # model name
    #----> paras for create model architecture
    num_tokens:int=None                         # num of tokens
    max_seq_len:int=None                        # max length of sequence
    dim:int=None                                # dim of tokens
    depth:int=None                              # layers
    heads:int=None                              # num of heads
    dim_head:int = 64                      # dim of heads
    local_attn_heads:int = 0
    local_window_size:int = 256
    causal:bool = False
    ff_mult:int = 4
    nb_features = None
    feature_redraw_interval:int = 1000
    reversible:bool = False
    ff_chunks:int = 1
    ff_glu = False
    emb_dropout:float = 0.
    ff_dropout:float = 0.
    attn_dropout:float = 0.
    generalized_attention:bool = False
    kernel_fn:Callable = nn.ReLU()
    use_scalenorm:bool = False
    use_rezero:bool = False
    cross_attend:bool = False
    no_projection:bool = False
    tie_embed :bool = False                  # False: output is num of tokens, True: output is dim of tokens  //multiply final embeddings with token weights for logits, like gpt decoder//
    
    g2v_position_emb:bool = True            # priority: gene2vec, no embedding
    g2v_weight_loc:str = None               # location of gene2vec weights
    
    auto_check_redraw:bool = True
    qkv_bias:bool = False