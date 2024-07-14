"""
some PEFT methods need to change basic implementation of layers 
give it a common name to make it not change anything in later implementation
"""

import torch
import torch.nn as nn

from functools import partial
from scLLM import logger
# declare implemented methods that will replace basic blocks

peft_supported_methods = ["peft_lora","custom_norm","fast_attention"]

class BasicOps:
    def __init__(self,
                ops_class_name:list=["custom_norm","fast_attention"],
                ops_class_para:list=[None,None],
                ) -> None:
        """
        BasicOps is a class that contains all the basic operations in torch.nn and some customized operations
        ops_class_name: list of str, the name of customized operations
        ops_class_para: list of any, the parameters of customized operations
        adapter_name: str, the name of adapter

        """
        self.ops_class_name = ops_class_name
        self.ops_class_para = ops_class_para

        #-----> bind torch part:
        # get all possible functions in torch.nn
        torch_nn_methods = [method for method in dir(nn) if callable(getattr(nn, method))]
        # use setattr to bind all functions in torch.nn to self
        for nn_method_name in torch_nn_methods: 
            #logger.debug(f"bind {nn_method_name} to self")
            setattr(self,nn_method_name,getattr(nn,nn_method_name))
        #-----> bind customised ops:
        if self.ops_class_name is not None:
            for name,para in zip(self.ops_class_name,self.ops_class_para):
                #logger.debug(f"bind {name} to self")
                self.bind_method(name,para)

        
    def bind_method(self,name:str,para):
        #-----> replace part:
        # replace some functions with customized functions
        if name == "peft_lora":
            logger.debug("replace Linear and Conv2d with LoRA version, and add MergedLinear")
            from scLLM.Modules.ops.lora import LoRALinear,LoRAConv2d,LoRAMergedLinear
            self.Linear = partial(LoRALinear,lora_para = para)
            self.Conv2d = partial(LoRAConv2d,lora_para = para)
            #---> add merged linear for lora when merging multiple layers
            self.MergedLinear = partial(LoRAMergedLinear,lora_para = para)
        elif name == "custom_norm":
            from scLLM.Modules.ops.Norm import PreLayerNorm,PreScaleNorm,DomainSpecificBatchNorm1d,DomainSpecificBatchNorm2d
            self.PreLayerNorm = PreLayerNorm
            self.PreScaleNorm = PreScaleNorm
            self.DomainSpecificBatchNorm1d = DomainSpecificBatchNorm1d
            self.DomainSpecificBatchNorm2d = DomainSpecificBatchNorm2d

        elif name == "fast_attention":
            from local_attention import LocalAttention
            from scLLM.Modules.ops.attention_funcs import FastAttention
            from scLLM.Modules.ops.similarity import CosineSimilarity_div_temp
            self.LocalAttention = LocalAttention
            self.FastAttention = FastAttention
            self.CosineSimilarity_div_temp = CosineSimilarity_div_temp
            from scLLM.Modules.ops.similarity import CosineSimilarity_div_temp

            self.CosineSimilarity_div_temp = CosineSimilarity_div_temp
        else:
            raise NotImplementedError
    