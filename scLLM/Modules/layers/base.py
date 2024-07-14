"""
base class for all layers to make it easy to customize layers

all of the peft methods need to be done in this level rather than in architecture level
"""

import torch
import torch.nn as nn
from scLLM.Modules.ops.base import BasicOps

class BaseLayers:
    def __init__(
        self,
        ops_fn = BasicOps,
        ops_class_name:list=["custom_norm","fast_attention"],
        ops_class_para:list=[None,None],
    ):
        super().__init__()
        self.ops = ops_fn(ops_class_name,ops_class_para)
