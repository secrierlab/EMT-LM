"""
common dataset class for scLLM
"""
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Union, Optional


############################################################
#               Dataset for scBERT
############################################################
from scLLM.Dataset.utils import scBert_sample
class SCDataset(Dataset):
    def __init__(self, data, label,cls_nb=19,random_sample=False):
        super().__init__()
        self.data = data
        self.label = label
        self.cls_nb = cls_nb

        self.random_sample = random_sample  # 是否随机采样

    def __getitem__(self, index):
        if self.random_sample:
            index = None
        full_seq, seq_label = scBert_sample(self.data, self.label,
                                    cls_nb=self.cls_nb, indx=index,
                                    )
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]
