
def add_multi_net(peft,MultiNet:callable,model_para,out_dim=2):
    trans_model = peft.pl_model.model
    trans_model.to_out = None

    full_model = MultiNet(trans_model,in_dim=model_para.max_seq_len,
                            dropout=model_para.drop,
                            h_dim=128,
                            out_dim=out_dim,) # for binary classification
    full_model.req_grad()
    peft.pl_model.model = full_model
    return peft