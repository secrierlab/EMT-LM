

# 

DIM = 128


import torch.nn as nn
class FuseNet(nn.Module):
    def __init__(self,in_dim=128*5, dropout = 0., h_dim = 128, out_dim = 5, ):
        nn.Module.__init__(self,)

        self.fc1 = nn.Linear(in_features=in_dim, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x
    

# 从文件夹中读取所有类的feature matrix
import pickle
import numpy as np

train_root = f"Path/to/train/" # ##need change##
val_root = f"/Path/to/val/"    # ##need change##

read_list = ["cls0","cls1","cls2","cls3","cls4"]
#-->train
train_feat_list = []
for i in range(len(read_list)):
    with open(train_root+read_list[i]+".pkl","rb") as f:
        [feat,label] = pickle.load(f)
        train_feat_list.append(feat)
train_label = label
print(len(train_feat_list),train_feat_list[0].shape,train_label.shape)
# 合并feat_list中的feature matrix，[sample_nb,feat_dim] -> [sample_nb,feat_dim*5]
train_feat = np.concatenate(train_feat_list,axis=1) 

#-->val
val_feat_list = []
for i in range(len(read_list)):
    with open(val_root+read_list[i]+".pkl","rb") as f:
        [feat,label] = pickle.load(f)
        val_feat_list.append(feat)
val_label = label
print(len(val_feat_list),val_feat_list[0].shape,val_label.shape)
# 合并feat_list中的feature matrix，[sample_nb,feat_dim] -> [sample_nb,feat_dim*5]
val_feat = np.concatenate(val_feat_list,axis=1)

# train model 
import torch

train_feat = torch.from_numpy(train_feat).float()
train_label = torch.from_numpy(train_label).long() 
# label from [sample_nb,1] -> [sample_nb,class_nb]
train_label = torch.zeros(train_label.shape[0],5).scatter_(1,train_label,1)


val_feat = torch.from_numpy(val_feat).float()
val_label = torch.from_numpy(val_label).long()
# label from [sample_nb,1] -> [sample_nb,class_nb]
val_label = torch.zeros(val_label.shape[0],5).scatter_(1,val_label,1)


# get Dataset and DataLoader
from torch.utils.data import TensorDataset,DataLoader
trainset = TensorDataset(train_feat,train_label)
valset = TensorDataset(val_feat,val_label)

trainloader = DataLoader(trainset,batch_size=DIM,shuffle=True)
valloader = DataLoader(valset,batch_size=DIM,shuffle=True)

# get model
model = FuseNet(in_dim=DIM*5, dropout = 0., h_dim = DIM, out_dim = 5, )

# get optimizer
from torch.optim import Adam
optimizer = Adam(model.parameters(),lr=1e-3)

# get loss
from torch.nn import CrossEntropyLoss
loss_func = CrossEntropyLoss()

# strat train
from tqdm import tqdm
# roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

model.to("cuda")
best_roc  = 0
for epoch in range(20):
    model.train()
    for feat,label in tqdm(trainloader):
        feat = feat.to("cuda")
        label = label.to("cuda")
        optimizer.zero_grad()
        pred = model(feat)

        #print(pred.shape,label.shape)
        loss = loss_func(pred,label)
        loss.backward()
        optimizer.step()
    print(f"epoch:{epoch},loss:{loss.item()}")
    # eval
    model.eval()
    pred_list = []
    label_list = []
    for feat,label in tqdm(valloader):
        feat = feat.to("cuda")
        label = label.to("cuda")
        pred = model(feat)
        pred = pred.detach().cpu().numpy()
        pred_list.append(pred)
        label_list.append(label.detach().cpu().numpy())
    pred_list = np.concatenate(pred_list,axis=0)
    label_list = np.concatenate(label_list,axis=0)

    # from pred_list and label_list get roc_auc for each class
    for i in range(5):
        print(f"roc_auc_score_{i}:",roc_auc_score(label_list[:,i],pred_list[:,i]))

    # from pred_list and label_list get average roc_auc_score
    score = roc_auc_score(label_list,pred_list,average="macro")
    print("average roc_auc_score:",score)
    # save model to ckpt
    if score > best_roc:
        best_roc = score
        save_loc = f"/Path/to/ckpt/model_{best_roc}.ckpt" # ##need change##
        torch.save(model.state_dict(),save_loc)

print(f"best model in {save_loc}")

