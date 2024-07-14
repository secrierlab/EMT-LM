import random
import torch
from typing import Dict, Optional, Union,Tuple,List
from torch.utils.data import DataLoader, Dataset


def mask_list(original_list,mask):
    """
    Mask a list with a mask
    :param original_list: list to be masked
    :param mask: mask with True/False values [True, False, True, ...]
    :return: masked list
    """
    return [x for x, m in zip(original_list, mask) if m]

def scBert_sample(data,label,cls_nb,indx=None):
    """
    Randomly sample a gene from the vocabulary.
    """
    if indx is None:
        rand_start = random.randint(0, data.shape[0]-1)
    else:
        rand_start = indx
    full_seq = data[rand_start].toarray()[0]
    full_seq[full_seq > cls_nb] = cls_nb
    full_seq = torch.from_numpy(full_seq).long()
    full_seq = torch.cat((full_seq, torch.tensor([0])))
    seq_label = label[rand_start]
    return full_seq, seq_label

def to_dataloader(dataset:Dataset,trainer_para,sampler:Optional[torch.utils.data.Sampler]=None):
    """
    Get dataloader from dataset
    Args:
        dataset: dataset to get dataloader
    Returns:
        dataloader
    """        
    if sampler is not None:
        trainer_para.shuffle = False
        trainer_para.additional_dataloader_para["sampler"] = sampler
    dataloader = DataLoader(dataset, 
                            batch_size=trainer_para.batch_size, 
                            shuffle=trainer_para.shuffle, 
                            num_workers=trainer_para.num_workers,
                            **trainer_para.additional_dataloader_para,
                            )
    return dataloader