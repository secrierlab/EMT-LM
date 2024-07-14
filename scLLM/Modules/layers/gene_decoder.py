import torch
import torch.nn as nn
from torch import Tensor
from typing import  Union, Dict

from scLLM import logger
from scLLM.Modules.sequence.reversible import grad_reverse
from scLLM.Modules.layers.base import BaseLayers



class ExprDecoder(nn.Module,BaseLayers):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
        **kwargs,
    ):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = self.ops.Sequential(
            self.ops.Linear(d_in, d_model),
            self.ops.LeakyReLU(),
            self.ops.Linear(d_model, d_model),
            self.ops.LeakyReLU(),
            self.ops.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = self.ops.Sequential(
                self.ops.Linear(d_in, d_model),
                self.ops.LeakyReLU(),
                self.ops.Linear(d_model, d_model),
                self.ops.LeakyReLU(),
                self.ops.Linear(d_model, 1),
            )
        logger.debug(f"ExprDecoder is built")

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.

