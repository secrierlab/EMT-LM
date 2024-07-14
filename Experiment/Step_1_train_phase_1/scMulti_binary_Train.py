import argparse
from pathlib import Path
import torch
import torch
import torch.nn as nn


import sys
sys.path.append("Path/to/repo") # repo folder into system path ##need change##


from scLLM.Modules.layers.base import BaseLayers
from scLLM.Models.scBERT.model import PerformerLM
from scLLM.Models.scBERT.paras import scBERT_para




import torch.nn as nn
class MultiNet(nn.Module):
    def __init__(self, Transformer,in_dim, dropout = 0., h_dim = 100, out_dim = 10,):
        super(MultiNet, self).__init__()
        self.model1 = Transformer


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


    def forward(self, x):
        #print(x.shape)
        x1 = self.model1(x,return_encodings=True)
        #print(x1.shape)
        x1 = x1[:,None,:,:]
        x1 = self.conv1(x1)
        x1 = self.act(x1)
        x1 = x1.view(x1.shape[0],-1)
        #print(x1.shape)
        x2 = x*x1


        x2 = self.fc1(x2)
        x2 = self.act1(x2)
        x2 = self.dropout1(x2)
        x2 = self.fc2(x2)
        x2 = self.act2(x2)
        x2 = self.dropout2(x2)
        out = self.fc3(x2)
        return out




def train(task_name,code_loc,raw_data_loc,vocab_loc,
            model_ckpt,vocab_params, out_loc,binarize=None):
    # your original code here
    # 把scLLM的位置添加进system path保证可以import scLLM
    import sys
    sys.path.append(code_loc)


    # 数据集读取
    #---> 定义数据集参数
    from scLLM.Dataset.paras import Dataset_para


    # config follows scBERT model pre-processing requests
    dataset_para = Dataset_para(
    vocab_loc=vocab_loc,
    var_idx = None,#"genes.gene_short_name",
    obs_idx="Ground_truth",  ##need change##
    filter_gene_by_counts=False,
    filter_cell_by_counts=200,
    log1p=True,
    log1p_base=2,


    cls_nb=5,
    data_layer_name="X_log1p",
    label_key = "Ground_truth", ##need change##
    #binarize=None, # not binarize use original label
    )
    #-----> 读取数据集 dill
    import dill
    with open(raw_data_loc,"rb") as f:
        trainset,valset,_,label_dict = dill.load(f)
    trainset.random_sample = False
    valset.random_sample = False
    assert label_dict is not None
    dataset_para.cls_nb = len(label_dict)
    # 输出数据集信息
    print(f"raw_data_loc: {raw_data_loc}")
    print("trainset size: ",len(trainset))
    print("valset size: ",len(valset)) if valset is not None else None
    print("label_dict: ",label_dict)


    if binarize is not None and int(binarize)<100 and int(binarize)<trainset.cls_nb:
        def binarize_label(label_tensor,target_label):
            """
            make label_tensor to be binary when label==target_label
            label_tensor: tensor of shape (n,)
            target_label: int
            """
            import torch
            new_label = torch.zeros_like(label_tensor)
            new_label[label_tensor==target_label] = 1
            return new_label
        #-----> binarize label
        trainset.label = binarize_label(trainset.label,int(binarize))
        valset.label = binarize_label(valset.label,int(binarize))


    dataset_para.cls_nb = 2
    dataset_para.label_key = "binarized_label"




    import torch
    import numpy as np
    from pathlib import Path

    from scLLM.Predefine.scBERT_classification import model_para,trainer_para


    #-----> project
    trainer_para.project = "EMT-scBERT-Multi" # project name  ##need change##
    trainer_para.entity= "Your work space" # entity name  ##need change##
    trainer_para.exp_name = task_name+binarize # experiment name
    #-----> dataset
    trainer_para.task_type = "classification" # "classification","regression"
    trainer_para.class_nb = dataset_para.cls_nb # number of classes
    trainer_para.batch_size =1 # batch size
    #-----> model
    trainer_para.pre_trained = model_ckpt
    trainer_para.ckpt_folder = str(Path(model_ckpt).parent)#
    trainer_para.metrics_names = ["auroc","accuracy","f1_score"] # metrics names
    #-----> pytorch lightning paras
    #accuracy_val
    trainer_para.max_epochs = 25 # max epochs
    trainer_para.save_ckpt = True # save checkpoint or not
    trainer_para.ckpt_format = "_{epoch:02d}-{auroc_val:.2f}" # check_point format # 注意这里我们没有用f-string，而是留下了未格式化的模板字符串
    trainer_para.ckpt_para = { #-----------> paras for pytorch_lightning.callbacks.ModelCheckpoint
    "save_top_k":1,
    "monitor":"auroc_val",
    "mode":"max",}
    trainer_para.trainer_output_dir = out_loc
    trainer_para.wandb_api_key = "Your wandb api key"  ##need change##
    trainer_para.additional_pl_paras.update({"precision":"16"})#"amp_backend":"apex","precision":"bf16",
    #amp_backend="apex"


    #-----> scBERT model paras
    model_para.g2v_weight_loc = vocab_params


    drop = 0.1
    #model_para.ff_dropout = drop # dropout rate
    #model_para.attn_dropout = drop # dropout rate
    #model_para.emb_dropout = drop # dropout rate
    model_para.drop = drop
    #-----> peft paras
    PEFT_name = "lora"
    from scLLM.Modules.ops.lora import default_lora_para
    lora_para = default_lora_para
    lora_para.r = 1
    lora_para.lora_alpha = 1
    lora_para.enable_lora = True
    #########################################################################
    # get pl model #
    #########################################################################
    #-----> init original model
    from scLLM.Models.scBERT.pl import pl_scBERT
    pl_model = pl_scBERT(trainer_paras=trainer_para,model_paras=model_para)


    #--------> change the model to PEFT model
    from scLLM.Models.PEFT import get_peft
    peft = get_peft(pl_model,PEFT_name,lora_para)
    del pl_model


    peft.load_model(original_ckpt = trainer_para.pre_trained)
    #-----> specify lora trainable params
    peft.set_trainable()
    # change output layer
    from scLLM.Modules.layers.out_layer import scBERT_OutLayer

    trans_model = peft.pl_model.model
    trans_model.to_out = None


    full_model = MultiNet(trans_model,in_dim=model_para.max_seq_len,
    dropout=model_para.drop,
    h_dim=128,
    out_dim=2,) # for binary classification
    full_model.req_grad()
    peft.pl_model.model = full_model


    #########################################################################
    # get sampler
    #########################################################################


    from torch.utils.data.sampler import WeightedRandomSampler
    # 根据数据集的类别分布，给每个样本赋予一个权重，使得每个类别的样本被抽到的概率相同
    from collections import Counter
    def get_weights(trainset):
        # 假设 trainset.label 是形状为 [5881, 1] 的tensor
        labels = trainset.label.squeeze().numpy() # 转为一维NumPy数组
        label_count = Counter(labels)


        # 计算每个类别的权重
        total_count = len(labels)
        label_weights = {label: 1.0 / count for label, count in label_count.items()}


        # 生成样本权重列表
        sample_weights = [label_weights[label] for label in labels]
        return sample_weights,total_count
    sample_weights,total_count = get_weights(trainset)
    # 创建 WeightedRandomSampler
    trainsampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_count,)# replacement=True)


    # 根据数据集的类别分布，给每个样本赋予一个权重，使得每个类别的样本被抽到的概率相同
    sample_weights,total_count = get_weights(valset)
    valsampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_count,)


    #--------> get dataloader
    from torch.utils.data import DataLoader
    trainloader = DataLoader(trainset, batch_size=trainer_para.batch_size, sampler=trainsampler)
    valloader = DataLoader(valset, batch_size=trainer_para.batch_size, sampler=valsampler)


    peft.pl_model.build_trainer()
    #with autocast():
    #with torch.autocast(device_type="cuda", dtype=torch.float16):
    print(f"start training with name {trainer_para.exp_name}")
    peft.pl_model.trainer.fit(peft.pl_model,trainloader,valloader)


    #--------> save model
    peft.pl_model.trainer.save_checkpoint(trainer_para.ckpt_folder+trainer_para.exp_name+"_last.ckpt")






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train for Task1.')
    parser.add_argument('--task_name', type=str, help='Name of task')


    parser.add_argument('--code_loc', type=str, help='Location of source code')
    parser.add_argument('--raw_data_loc', type=str, help='Location of data')
    parser.add_argument('--vocab_loc', type=str, help='Location of model vocab')
    parser.add_argument('--model_ckpt', type=str, help='Location of model checkpoint')
    parser.add_argument('--vocab_params', type=str, help='Location of vocab mapping embedding')
    parser.add_argument('--out_loc', type=str, help='Output location')


    parser.add_argument('--binarize', type=str, default='100', help='binarize')


    args = parser.parse_args()


    train(args.task_name,args.code_loc,args.raw_data_loc,args.vocab_loc, args.model_ckpt,args.vocab_params, args.out_loc,args.binarize)



