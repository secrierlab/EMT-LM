"""
different type of adapters that can be used in the network
"""
import torch
import torch.nn as nn
############################################################################################################
#    placeholder adapter
############################################################################################################
class Placeholder_Adapter(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        pass

    def forward(self, x, **kwargs):
        return x