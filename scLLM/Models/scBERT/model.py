import torch
from torch import nn
from scLLM.Modules.utils import  exists, cast_tuple
from scLLM.Modules.layers.gene_encoder import Gene2VecPositionalEmbedding
from scLLM.Modules.Performer import Performer
from scLLM.Modules.utils import  Always
from scLLM.Modules.init import APEX_AVAILABLE
from scLLM.Models.scBERT.paras import scBERT_para
class PerformerLM(nn.Module):
    def __init__(
        self,
        paras:scBERT_para,
    ):
        super().__init__()
        self.paras = paras

        local_attn_heads = cast_tuple(self.paras.local_attn_heads)

        self.max_seq_len = self.paras.max_seq_len
        self.token_emb = nn.Embedding(self.paras.num_tokens, self.paras.dim)

        if self.paras.g2v_position_emb:
            self.pos_emb = Gene2VecPositionalEmbedding(self.paras.g2v_weight_loc,self.paras.max_seq_len,)
                                                       #ops_class_name = self.paras.ops_class_name,
                                                       #ops_class_para = self.paras.ops_class_para)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = torch.zeros_like
            self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(self.paras.emb_dropout)

        self.performer = Performer(self.paras.dim, 
                                   self.paras.depth, 
                                   self.paras.heads, 
                                   self.paras.dim_head, 
                                   local_attn_heads, 
                                   self.paras.local_window_size, 
                                   self.paras.causal, 
                                   self.paras.ff_mult, 
                                   self.paras.nb_features, 
                                   self.paras.feature_redraw_interval, 
                                   self.paras.reversible, 
                                   self.paras.ff_chunks, 
                                   self.paras.generalized_attention, 
                                   self.paras.kernel_fn, 
                                   self.paras.use_scalenorm, 
                                   self.paras.use_rezero, 
                                   self.paras.ff_glu, 
                                   self.paras.ff_dropout, 
                                   self.paras.attn_dropout, 
                                   self.paras.cross_attend, 
                                   self.paras.no_projection, 
                                   self.paras.auto_check_redraw, 
                                   self.paras.qkv_bias,
                                   #----> for operator
                                   ops_class_name = self.paras.ops_class_name,
                                   ops_class_para = self.paras.ops_class_para)
        self.norm = nn.LayerNorm(self.paras.dim)
        self.to_out = nn.Linear(self.paras.dim, self.paras.num_tokens) if not self.paras.tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings = False, output_attentions = False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embedding
        x = self.token_emb(x)
        if output_attentions:
            x.requires_grad_()    # used for attn_map output
        x += self.pos_emb(x)
        x = self.dropout(x)

        # performer layers
        layer_pos_emb = self.layer_pos_emb(x)

        if output_attentions:
            x, attn_weights = self.performer(x, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)
            # norm and to logits
            x = self.norm(x)
            if return_encodings:
                return x, attn_weights

            if exists(self.to_out):
                return self.to_out(x), attn_weights

            return (x @ self.token_emb.weight.t()), attn_weights
        else:
            x = self.performer(x, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)

            # norm and to logits
            x = self.norm(x)
            if return_encodings:
                return x

            if exists(self.to_out):
                x = self.to_out(x)
                return x
            #@ means matrix multiplication __matmul__()
            return x @ self.token_emb.weight.t()
     