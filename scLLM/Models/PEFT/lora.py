from scLLM.Models.PEFT.base import PEFT_base
from scLLM.Modules.ops.lora import LoRA_para,default_lora_para,mark_only_lora_as_trainable,lora_state_dict
from scLLM import logger
import torch

class PEFT_LoRA(PEFT_base):
    def __init__(self,pl_base,peft_para:LoRA_para=default_lora_para) -> None:
        super().__init__(pl_base,peft_para)
        logger.info("LoRA is used as PEFT method")
        self.reset_ops()

    def reset_ops(self):
        # reset the ops in the base model 
        # and replace the trained parameters with the trained parameters in the peft model
        logger.info(f"reset ops to LoRA version in the base model")
        self.pl_model.model=None
        #----> add lora paras into model paras
        self.pl_model.model_paras.ops_class_name.append("peft_lora")
        self.pl_model.model_paras.ops_class_para.append(self.peft_para)
        logger.info(f"ops change to {self.pl_model.model_paras.ops_class_name} ")
        logger.info(f"ops change to {self.pl_model.model_paras.ops_class_para} ")
        #assert self.pl_model.model.
        self.pl_model.create_model()
        self.pl_model.configure_optimizers()

    def set_trainable(self,):
        logger.info("set LoRA params trainable")
        mark_only_lora_as_trainable(self.pl_model.model)

    def load_model(self,original_ckpt:str,peft_ckpt_loc:str=None,map_device:str="cuda"):
        # Load the pretrained checkpoint first
        self.pl_model.model.load_state_dict(torch.load(original_ckpt,map_location=map_device), strict=False)
        if peft_ckpt_loc is not None:
            # Then load the LoRA checkpoint
            self.load_lora(peft_ckpt_loc,map_device)
    
    def load_lora(self,lora_ckpt_loc:str,map_device:str="cuda"):
        # Then load the LoRA checkpoint
        self.pl_model.model.load_state_dict(torch.load(lora_ckpt_loc,map_location=map_device), strict=False)

    def save_peft(self,lora_path:str):
        # save lora separately and manually
        torch.save(lora_state_dict(self.model), lora_path)