

# 

DIM = 128
#which PT to use
EXTERNAL_PT = False
COOK_PT = True

import torch.nn as nn
class FuseNet(nn.Module):
    def __init__(self,in_dim=128*5, dropout = 0., h_dim = 128, out_dim = 1, ):
        nn.Module.__init__(self,)

        self.fc1 = nn.Linear(in_features=in_dim, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

        self.act3 = nn.ReLU()

    def forward(self, x):

        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.act3(x)
        return x


# 从文件夹中读取所有类的feature matrix
import pickle
import numpy as np

train_root = f"Path/to/train/" # ##need change##
val_root = f"Path/to/val/"     # ##need change##

read_list = ["cls0","cls1","cls2","cls3","cls4"]
#-->train
train_feat_list = []
for i in range(len(read_list)):
    with open(train_root+read_list[i]+".pkl","rb") as f:
        [feat,label] = pickle.load(f)
        train_feat_list.append(feat)
train_feat = np.concatenate(train_feat_list,axis=1) 

#train_label from pseudo time
if EXTERNAL_PT:
    PT_train_loc= "Path/to/cook_calc_PT.h5ad" # ##need change##
    import anndata
    adata = anndata.read_h5ad(PT_train_loc)
    train_label = adata.obs['DPT'].values
elif COOK_PT:
    train_label = label
else:
    raise ValueError("No PT label")

print(len(train_feat_list),train_feat_list[0].shape,train_label.shape)
# 合并feat_list中的feature matrix，[sample_nb,feat_dim] -> [sample_nb,feat_dim*5]


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

# concat cook 
#train_feat = np.concatenate([train_feat,val_feat],axis=0)
#train_label = np.concatenate([train_label,val_label],axis=0)


# train model 
import torch

train_feat = torch.from_numpy(train_feat).float()
train_label = torch.from_numpy(train_label).float()  # regression


val_feat = torch.from_numpy(val_feat).float()
val_label = torch.from_numpy(val_label).float() # regression



# get Dataset and DataLoader
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.data import DataLoader, random_split
dataset = TensorDataset(train_feat,train_label)
valset = TensorDataset(val_feat,val_label)

trainloader = DataLoader(dataset,batch_size=DIM,shuffle=True)
valloader = DataLoader(valset,batch_size=DIM,shuffle=True)

# get model
model = FuseNet(in_dim=DIM*5, dropout = 0.3, h_dim = DIM, out_dim = 1, ) # out_dim = 1 for regression

# get optimizer
from torch.optim import Adam
optimizer = Adam(model.parameters(),lr=1e-5)

# get loss
from torch.nn import CrossEntropyLoss
#loss_func = CrossEntropyLoss()
loss_func = nn.MSELoss()
# strat train
from tqdm import tqdm
# roc_auc_score

def eval_metrics(pred_list,label_list):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    # 计算MSE
    mse = mean_squared_error(label_list, pred_list)
    print("Mean Squared Error (MSE):", mse)

    # 计算RMSE
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # 计算MAE
    mae = mean_absolute_error(label_list, pred_list)
    print("Mean Absolute Error (MAE):", mae)

    # 计算R^2
    A_np = label_list
    B_np = pred_list #preds 
    pearson_corr = np.corrcoef(A_np.T, B_np.T)[0, 1]
    print("Paul Corr:", pearson_corr)
    score = pearson_corr
    return score



model.to("cuda")
best_score  = 0
for epoch in range(200):
    model.train()
    for feat,label in tqdm(trainloader):
        feat = feat.to("cuda")
        label = label.to("cuda")
        optimizer.zero_grad()
        pred = model(feat)

        #print(pred.shape,label.shape)
        #label = label.unsqueeze(1)
        loss = loss_func(pred,label)
        loss.backward()
        optimizer.step()
    print(f"epoch:{epoch},loss:{loss.item()}")
    score = loss.item()
    
    # eval
    model.eval()
    pred_list = []
    label_list = []
    for feat,label in tqdm(valloader):
        feat = feat.to("cuda")
        label = label.to("cuda")
        pred = model(feat)
        pred = pred.detach().cpu().numpy()
        pred_list.append(pred*10)
        label_list.append(label.detach().cpu().numpy())
    pred_list = np.concatenate(pred_list,axis=0)
    label_list = np.concatenate(label_list,axis=0)

    # from pred_list and label_list get roc_auc for each class
    score = eval_metrics(pred_list,label_list)
    


    # save model to ckpt
    if score > best_score:
        best_score = score
        save_loc = f"best/model_{best_score}.ckpt"
        torch.save(model.state_dict(),save_loc)

print(f"best model in {save_loc}")

