import torch
from typing import Dict, Optional, Union,Tuple,List
import numpy as np, scanpy as sc, anndata as ad,pandas as pd

from scipy.sparse import issparse,lil_matrix,csr_matrix
from scanpy.get import _get_obs_rep, _set_obs_rep
from anndata import AnnData
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoTokenizer, BertTokenizer
from torch.utils.data import DataLoader, Dataset

from scLLM import logger
from scLLM.Dataset.paras import Dataset_para
from scLLM.Dataset.Vocab import GeneVocab


class scBERTPostprocessor:
    def __init__(self,paras:Dataset_para,vocab:GeneVocab) -> None:
        self.paras = paras
        self.my_vocab = vocab

    def run(self,
                 adata,
                ):
        # data part
        data_type = self.paras.data_layer_name
        # label part
        label_key = self.paras.label_key
        cls_nb = self.paras.cls_nb
        binarize = self.paras.binarize # method to binarize label
        bins = self.paras.bins
        bin_min = self.paras.bin_min
        bin_max = self.paras.bin_max
        save_in_obs = self.paras.save_in_obs

        # split part
        n_splits = self.paras.n_splits
        test_size = self.paras.test_size
        random_state = self.paras.random_state

        # get all data
        all_data = self.to_data(adata=adata,data_type=data_type)
        # get all label
        all_label,class_weight = self.to_label(adata=adata,
                                  label_key=label_key,
                                  #----> for binarize label
                                    binarize=binarize,
                                    bins=bins,
                                    bin_nb=cls_nb,bin_min=bin_min,bin_max=bin_max,
                                    save_in_obs=save_in_obs,
                                  )
        if self.paras.test_size is not None:
            # split train test
            D_train,D_val = self.split_train_test(all_data,all_label,
                                                # for how to split
                                                n_splits=n_splits, 
                                                test_size=test_size, 
                                                random_state=random_state,
                                                )
            # for train part
            trainset = self.create_dataset(D_train,
                                        cls_nb=cls_nb,
                                        )
            # for val part
            valset = self.create_dataset(D_val,
                                            cls_nb=cls_nb,
                                            )
            return trainset,valset,class_weight
        else:
            # for train part
            trainset = self.create_dataset([all_data,all_label],
                                        cls_nb=cls_nb,
                                        )
            return trainset,None,class_weight
    ##############################################################################################################
    #  tokenize steps for scBERT
    #############################################################################################################

    def to_data(self,adata,data_type:str):
        """
        Get processed data from AnnData object
        Args:
            data_type: "X","normed","log1p","binned"
        Returns:
            processed data in sparse matrix format
        """
        data_type_list = ["X","X_normed","X_log1p","X_binned"]
        assert data_type in data_type_list, f"data_type must be in {data_type_list}"
        if data_type == "X":
            if self.paras.result_normed_key is not None: 
                logger.warning(f"X is not normalised, check layer {self.paras.result_normed_key}")
            if self.paras.result_log1p_key is not None:
                logger.warning(f"X is not log1p transformed, check layer {self.paras.result_log1p_key}")
            if self.paras.result_binned_key is not None:
                logger.warning(f"X is not binned, check layer {self.paras.result_binned_key}")
            return adata.X
        else:
            name_dict = {"X_normed":self.paras.result_normed_key,
                        "X_log1p":self.paras.result_log1p_key,
                        "X_binned":self.paras.result_binned_key}
            data_type_name = name_dict[data_type]
            all_counts = (
                    adata.layers[data_type_name].A
                    if issparse(adata.layers[data_type_name])
                    else adata.layers[data_type_name])
            sparse_counts = csr_matrix(all_counts)
            return sparse_counts
        
    def to_label(self,
                adata:AnnData,
                label_key: str, 
                #----> for binarize label
                binarize:str=None,
                bins:np.ndarray=None,
                bin_nb: int=None,bin_min:float=None,bin_max:float=None,
                save_in_obs:bool=True,
                
                ) -> None:
        """
        get label from adata.obs[label_key]
        if binarize is True, then we will binarize the label
        if save_in_obs is True, then we will save the label in adata.obs[label_key]
        Args:
            label_key (:class:`str`):
                The key of :class:`AnnData.obs` to use as label
            binarize (:class:`str`)(optional): ["quantile",""]
                If True, we will binarize the label
            bins (:class:`np.ndarray`)(optional):
                The bins to binarize the label
            bin_nb (:class:`int`)(optional):
                The number of bins to binarize the label
            bin_min (:class:`float`)(optional):
                The min value of bins to binarize the label
            bin_max (:class:`float`)(optional):
                The max value of bins to binarize the label
            save_in_obs (:class:`bool`)(optional):
                If True, we will save the label in adata.obs[label_key]
        Returns:
            label (:class:`torch.tensor`):
                The label of the data
            class_weight (:class:`torch.tensor`):
                The class weight of the each category
        """
        logger.info(f"Discritize label {label_key} in obs_names")
        original_label = adata.obs[label_key] 
        if binarize is not None:
            assert binarize in ["equal_width","equal_instance"]
            if bins is None:
                assert bin_nb is not None 
                if bin_min is None: bin_min = original_label.min()
                if bin_max is None: bin_max = original_label.max()
                if binarize == "equal_width":
                    bins = np.linspace(bin_min, bin_max, bin_nb)
                elif binarize == "equal_instance":
                    c_label = np.sort(original_label.to_numpy().flatten())
                    bins = np.array([ c_label[int(((len(c_label)-1)/bin_nb)*i)] for i in range(bin_nb)])
            bin_names = np.arange(bin_nb)
            digitized = np.digitize(original_label, bins)
            binned_label = bin_names[digitized-1]
            if save_in_obs:
                obs_name = f"{label_key}_binned"
                adata.obs[obs_name] = binned_label
            np_label = binned_label
            class_num = np.unique(np_label, return_counts=True)[1].tolist()
            class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
            label = torch.from_numpy(np_label).unsqueeze(1)
        else:
            np_label = original_label.to_numpy()
            label = torch.from_numpy(np_label).unsqueeze(1)
            class_weight = None
        self.class_weight = class_weight

        return label,class_weight

    def split_train_test(self,all_data,all_label,
                          n_splits=1, test_size=0.2, random_state=2023):
        from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
        if self.paras.cls_nb >1:
            logger.info("Classification task, split with StratifiedShuffleSplit")
            #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
            sss = StratifiedShuffleSplit(n_splits=n_splits, 
                                        test_size=test_size, 
                                        random_state=random_state)

            idx_tr,idx_val = next(iter(sss.split(all_data, all_label)))
        else:
            logger.info("Regression task, split with ShuffleSplit")
            # 使用ShuffleSplit
            ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
            idx_tr,idx_val  = next(iter(ss.split(all_data)))
        data_train, label_train = all_data[idx_tr], all_label[idx_tr]
        data_val, label_val = all_data[idx_val], all_label[idx_val]
        return [data_train,label_train],[data_val,label_val]
    
    def create_dataset(self,data_and_label:list,cls_nb:int=5):
        [data,label] = data_and_label
        from scLLM.Dataset.dataset import SCDataset
        dataset = SCDataset(data, label,cls_nb=cls_nb)
        return dataset


