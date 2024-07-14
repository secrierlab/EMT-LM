
class PEFT_base:
    def __init__(self,pl_base,peft_para = None) -> None:
        self.pl_model = pl_base
        self.peft_para = peft_para

    def reset_ops(self):
        # reset the ops in the base model 
        # and replace the trained parameters with the trained parameters in the peft model
        raise NotImplementedError
    
    def load_model(self,original_ckpt:str,peft_ckpt_loc:str,map_device:str="cuda"):
        raise NotImplementedError
    
    def set_trainable(self,):
        # for PEFT, we only need to set some trainable parameters in the PEFT model
        raise NotImplementedError
    
    def save_peft(self,peft_path:str):
        # save peft separately and manually
        raise NotImplementedError




