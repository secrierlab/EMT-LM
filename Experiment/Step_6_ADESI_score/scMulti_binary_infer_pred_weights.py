import argparse
from pathlib import Path

def infer(code_loc,raw_data_loc,vocab_loc, model_ckpt,vocab_params, out_loc,label_key):
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
        obs_idx=label_key,
        filter_gene_by_counts=False,
        filter_cell_by_counts=200,
        log1p=True,
        log1p_base=2,

        test_size=None, #use all data to inference
        cls_nb=5,
        data_layer_name="X_log1p",
        label_key = label_key,
        binarize=None, # not binarize use original label
    )

    # -----> 读取数据集
    # 把scLLM的位置添加进system path保证可以
    import sys
    sys.path.append(code_loc)

    # 数据集读取
    import dill
    # 用dill打开loc0的pkl 文件读取dataset
    with open(raw_data_loc,"rb") as f:
        [trainset,valset,_,label_dict] = dill.load(f)
    # 输出数据集信息
    print("trainset size: ",len(trainset))
    print("valset size: ",len(valset)) if valset is not None else print("no valset")
    print(label_dict)


    import torch
    import numpy as np

    from scLLM.Predefine.scBERT_classification import model_para,trainer_para

    #-----> project
    trainer_para.project = "debug" # project name
    trainer_para.entity= "shipan_work" # entity name
    trainer_para.exp_name = trainer_para.exp_name + "EMT—infer" # experiment name
    #-----> dataset
    trainer_para.task_type = "classification" # "classification","regression"
    trainer_para.class_nb = 2 # number of classes
    trainer_para.batch_size =1 # batch size
    #-----> model
    trainer_para.pre_trained = model_ckpt#"/Users/shipan/Documents/workspace_scLLM/pre_trained/EMT_scBERT/FFT_emt_easy_scLLM_scBERT-3_cls3_LR5e-05_77.ckpt"#"//main/PAN/Exp03_scLLM/pre_trained/scBERT/panglao_pretrain.pth"
    trainer_para.ckpt_folder = str(Path(model_ckpt).parent)+"/" #"/Users/shipan/Documents/workspace_scLLM/pre_trained/EMT_scBERT/"

    #-----> pytorch lightning paras
    trainer_para.trainer_output_dir = str(Path(model_ckpt).parent)+"/" 
    trainer_para.wandb_api_key = "1266ad70f8bf7695542bf9a2d0dec8748c52431c"


    #-----> scBERT model paras
    model_para.g2v_weight_loc = vocab_params#"/Users/shipan/Documents/workspace_scLLM/pre_trained/scBERT/gene2vec_16906_200.npy"


    #-----> peft paras
    PEFT_name = "lora"
    from scLLM.Modules.ops.lora import default_lora_para
    lora_para = default_lora_para
    lora_para.r = 1
    lora_para.lora_alpha = 1
    lora_para.enable_lora = True

    from scLLM.Models.scBERT.pl import pl_scBERT
    pl_model = pl_scBERT(trainer_paras=trainer_para,model_paras=model_para)

    #--------> change the model to PEFT model
    from scLLM.Models.PEFT import get_peft
    peft = get_peft(pl_model,PEFT_name,lora_para)

    # change output layer
    from scLLM.Models.scRankNet.model import MultiNet
    peft.pl_model.model.to_out = None
    peft.pl_model.model = MultiNet(peft.pl_model.model,in_dim=model_para.max_seq_len,
                            dropout=0., 
                            h_dim=128, 
                            out_dim=trainer_para.class_nb,)
    peft.pl_model.load_state_dict(torch.load(trainer_para.pre_trained,map_location=torch.device('cpu'))["state_dict"],strict=True)

    idx = 0
    dataset = trainset#trainset
    # concate sparse matrix
    import scipy.sparse as sp
    #data = sp.vstack([trainset.data,valset.data]) 
    #label = np.concatenate([trainset.label,valset.label],axis=0)
    data = dataset.data
    label = dataset.label

    print(data.shape,label.shape)
    infer_size = data.shape[0]
    infer_class = 5#dataset.cls_nb#trainer_para.class_nb
    #label_list = label

    pred_list = np.zeros([infer_size,trainer_para.class_nb])
    label_list = np.zeros([infer_size,1])
    weights_list = np.zeros([infer_size,model_para.max_seq_len])
    weighted_feature_list = np.zeros([infer_size,model_para.max_seq_len])
    import tqdm
    peft.pl_model.model.eval()

    peft.pl_model.model.to("cuda")
    import torch.nn.functional as F

    with torch.no_grad():
        for idx in range(infer_size):
            if idx % 1000 == 0: print(f"{idx}/{infer_size}")
            #data part
            full_seq = data[idx].toarray()[0]
            full_seq[full_seq > (infer_class - 2)] = infer_class - 2
            full_seq = torch.from_numpy(full_seq).long()
            full_seq = torch.cat((full_seq, torch.tensor([0]))).to("cuda")
            full_seq = full_seq.unsqueeze(0)
            
            if idx !=0 and idx%300 == 0:
                print(full_seq.shape[1]-torch.sum(last_full_seq == full_seq).item())
            last_full_seq = full_seq
            #label part
            y = label[idx]
            
            # model part
            #out=peft.pl_model.model(full_seq,return_weight=False)#,output_attentions = False)
            out,weights,weighted_feature=peft.pl_model.model(full_seq,return_weight=True)
            pred = F.softmax(out,dim=1)
            pred = pred.detach().cpu().numpy()

            #print(out)
            pred_list[idx,:] = pred
            label_list[idx,:] = y
            weights_list[idx,:] = weights.detach().cpu().numpy()
            weighted_feature_list[idx,:] = weighted_feature.detach().cpu().numpy()

    # pkl save pred_list
    import pickle
    with open(out_loc,"wb") as f:
        pickle.dump([pred_list,label_list,weights_list,weighted_feature_list],f)
    
    #raise Exception("Done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for Task1.')
    
    parser.add_argument('--code_loc', type=str, help='Location of source code')
    parser.add_argument('--raw_data_loc', type=str, help='Location of data')
    parser.add_argument('--vocab_loc', type=str, help='Location of model vocab')
    parser.add_argument('--model_ckpt', type=str, help='Location of model checkpoint')
    parser.add_argument('--vocab_params', type=str, help='Location of vocab mapping embedding')
    parser.add_argument('--index_label', type=str, help='index name of label')
    parser.add_argument('--out_loc', type=str, help='Output location')

    args = parser.parse_args()
    infer(args.code_loc,args.raw_data_loc,args.vocab_loc, args.model_ckpt,args.vocab_params, args.out_loc,args.index_label)
