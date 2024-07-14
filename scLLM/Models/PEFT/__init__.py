
from scLLM.Models.PEFT.lora import PEFT_LoRA


def get_peft(pl_base,peft_method,peft_para):
    if peft_method == "lora":
        return PEFT_LoRA(pl_base,peft_para)
    else:
        raise NotImplementedError