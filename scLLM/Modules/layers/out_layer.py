"""
for fine-tuning the output layer
"""
import torch
import torch.nn as nn

from scLLM.Modules.layers.base import BaseLayers


class scBERT_OutLayer(nn.Module,BaseLayers):
    def __init__(self,in_dim, dropout = 0., h_dim = 100, out_dim = 10, out_layer:str="all",**kwargs):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        assert out_layer in ["all","conv1","fc1","fc2",]
        self.out_layer = out_layer

        self.conv1 = self.ops.Conv2d(1, 1, (1, 200))
        self.act = self.ops.ReLU()
        self.fc1 = self.ops.Linear(in_features=in_dim, out_features=512, bias=True)
        self.act1 = self.ops.ReLU()
        self.dropout1 = self.ops.Dropout(dropout)
        self.fc2 = self.ops.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = self.ops.ReLU()
        self.dropout2 = self.ops.Dropout(dropout)
        self.fc3 = self.ops.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):

        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        if self.out_layer=="conv1":
            return x

        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        if self.out_layer=="fc1":
            return x

        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        if self.out_layer=="fc2":
            return x

        x = self.fc3(x)
        if self.out_layer=="all":
            return x
        else:
            raise ValueError(f"out_layer {self.out_layer} not supported")