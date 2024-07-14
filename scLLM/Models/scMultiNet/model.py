
import torch.nn as nn
class MultiNet(nn.Module):
    def __init__(self, Transformer,in_dim, dropout = 0., h_dim = 100, out_dim = 10,  out_layer:str="all"):
        super(MultiNet, self).__init__()
        self.model1 = Transformer
        assert out_layer in ["all","conv1","fc1","fc2",]
        self.out_layer = out_layer

        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def req_grad(self):
      self.conv1.requires_grad_(True)
      self.act.requires_grad_(True)
      self.fc1.requires_grad_(True)
      self.act1.requires_grad_(True)
      self.dropout1.requires_grad_(True)
      self.fc2.requires_grad_(True)
      self.act2.requires_grad_(True)
      self.dropout2.requires_grad_(True)
      self.fc3.requires_grad_(True)

    def forward(self, x, return_weight = False):
      #print(x.shape)
      x1 = self.model1(x,return_encodings=True)
      #print(x1.shape)
      x1 = x1[:,None,:,:]
      x1 = self.conv1(x1)
      x1 = self.act(x1)
      x1 = x1.view(x1.shape[0],-1)
      if self.out_layer=="conv1":
        return x1
      #print(x1.shape)
      x2 = x*x1

      x2 = self.fc1(x2)
      x2 = self.act1(x2)
      x2 = self.dropout1(x2)
      if self.out_layer=="fc1":
        return x2


      x2 = self.fc2(x2)
      x2 = self.act2(x2)
      x2 = self.dropout2(x2)

      if self.out_layer=="fc2":
        return x2

      out = self.fc3(x2)
      if self.out_layer=="all":
        if return_weight:
          return out,x1,x*x1
        else:
          return out
      else:
        raise ValueError(f"out_layer {self.out_layer} not supported")
    