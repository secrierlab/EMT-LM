"""
from csv or anndata to a usable annData

X = (n_sample, n_gene) matrix
obs["str_label"] = (n_sample, ) array
var["gene_name"] = (n_gene, ) array

"""
from typing import Dict, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np, anndata as ad, pandas as pd, scanpy as sc
from scipy.sparse import issparse,lil_matrix,csr_matrix
from scanpy.get import _get_obs_rep, _set_obs_rep
from anndata import AnnData

from scLLM import logger
from scLLM.Dataset.paras import Dataset_para
from scLLM.Dataset.Vocab import GeneVocab
from scLLM.Dataset.Preprocessor import scPreprocessor
from scLLM.Dataset.Postprocessor import scBERTPostprocessor
class scReader:
    """
    read data from raw csv or anndata object
    """

    def __init__(
        self,
        dataset_para: Dataset_para,
    ):
        r"""
        Set up the preprocessor, use the args to config the workflow steps.
        """
        logger.info("Initializing preprocessor ...")
        #logger.debug(f"dataset_para: {dataset_para}")
        self.para = dataset_para
        self.adata = None
        self.vocab = None

    def init_vocab(self,vocab_list:list=None,vocab_dict:str=None):
        """
        init vocab from a csv file
        """
        if vocab_list is not None:
            logger.info(f"use customised vocab from list")
            self.para.gene_vocab = vocab_list
        elif vocab_dict is not None:
            logger.info(f"use customised vocab from dict")
            self.para.gene_vocab = list(vocab_dict.keys())
        else:
            logger.info(f"use default vocab from dataset_para")
        if self.para.vocab_loc is not None:
            logger.info(f"load vocab from {self.para.vocab_loc}")
            self.vocab = GeneVocab.from_file(self.para.vocab_loc)
        elif self.para.gene_vocab is not None:
            logger.info(f"gene vocab length: {len(self.para.gene_vocab)}")
            self.vocab = GeneVocab(gene_list_or_vocab=self.para.gene_vocab)
        else:
            raise ValueError("shoulde give one of the following: \
                             dataset_para.gene_vocab,dataset_para.vocab_loc, \
                             or vocab_list, vocab_dict.")

    ##############################################################################################################
    #           read from data
    #############################################################################################################
    def load_adata(self, loc:str, translate:bool=False):
        """
        Load data from anndata object
        Args:
            loc: anndata object location
            translate: if True, translate anndata from other source into scLLM format
            var_idx: if translate is True, var_idx is the name of gene name colume in anndata.var object
            obs_idx: if translate is True, obs_idx is the name of label colume in anndata.obs object
        """
        logger.info(f"Load data from anndata object.")
        # read raw data from anndata file
        self.adata = sc.read_h5ad(loc)
        if translate:
            var_idx = self.para.var_idx
            obs_idx = self.para.obs_idx
            is_extend = True if self.para.tokenize_name=="scBERT" else False
            #assert var_idx is not None, "var_idx is None"
            assert obs_idx is not None, "obs_idx is None"
            # get vocab list
            vocab = self.vocab.to_itos()
            # get X info
            if var_idx is None:
                original_gene = self.adata.var_names
                original_gene_list = original_gene.to_list()
            else:
                original_gene = self.adata.var[var_idx]
                original_gene_list = original_gene.to_list()
            logger.debug(f"In original adata with gene {len(original_gene_list)}")
            if is_extend:
                counts = lil_matrix((self.adata.shape[0],len(vocab)),dtype=np.float32)
                logger.debug(f"In original adata with gene {len(original_gene_list)}")
                for i in range(len(vocab)):
                    if i % 2000 == 0: logger.debug(f"processing {i}/{len(vocab)}")
                    if vocab[i] in original_gene_list:
                        # save counts
                        #mask = (adata.var[var_idx]==vocab[i])
                        mask = (original_gene==vocab[i])
                        mask_list = mask.to_list() if isinstance(mask,pd.Series) else mask.tolist()
                        if mask_list.count(True) > 1:
                            logger.warn(f"gene {vocab[i]} has more than one columes, mix them up with mean()")
                            counts[:,i] = self.adata.X[:,mask].mean(axis=1)
                        else:
                            counts[:,i] = self.adata.X[:,mask]
                gene_name = vocab
            else:
                # filter out adata not in vocab
                filted_gene_list = [gene for gene in original_gene_list if gene in vocab]
                counts = lil_matrix((self.adata.shape[0],len(filted_gene_list)),dtype=np.float32)
                logger.debug(f"create sparse matrix with shape:{counts.shape}")

                gene_name = []
                for i in range(len(filted_gene_list)):
                    if i % 1000 == 0: logger.debug(f"processing {i}/{len(filted_gene_list)}")
                    # save counts
                    mask = (original_gene==filted_gene_list[i])
                    mask_list = mask.to_list() if isinstance(mask,pd.Series) else mask.tolist()
                    if mask_list.count(True) > 1:
                        logger.warn(f"gene {vocab[i]} has more than one columes, mix them up with mean()")
                        counts[:,i] = self.adata.X[:,mask].mean(axis=1)
                    else:
                        counts[:,i] = self.adata.X[:,mask]
                    # save gene name
                    gene_name.append(filted_gene_list[i])
            counts = counts.tocsr()
            # get obs info normally sample id
            obs = self.adata.obs[obs_idx].to_frame() # can only accept dataframe
            logger.info(f"create anndata in scLLM format..")
            new = ad.AnnData(X=counts)
            new.var_names = gene_name
            new.obs = obs
            new.obs_names = self.adata.obs_names
            logger.debug(f"restore anndata in scLLM format..")
            self.adata = new
            logger.info(f"Done.")
   
    def from_csv(self,
                csv_loc:str,
                gene_name_mask:list,
                sample_id_idx:str,
                sample_mask:list,
                obs_name_mask:list,
                df:pd.DataFrame=None):
        """
        convert csv to anndata
        in:
            csv_loc: csv file location
            gene_name_mask: gene name mask if this colume in csv file should be included, then 
                           we set this gene name with True
            sample_id_idx: a index in a csv file that can use to sample items (normally sampel id)
            sample_mask: sample mask if this row in csv file should be included, then
                           we set this sample with True
            obs_name_mask: obs name mask if this colume in csv file should be included, then
                           we set this obs name with True
            df: (optional) a dataframe that can be used to convert to anndata
        out:
            anndata object
        """
        if df is None:
            logger.debug(f"Transfer a csv file to anndata object: {csv_loc}")
            df = pd.read_csv(csv_loc)
        else:
            logger.debug(f"Read raw data from a dataframe.")
            df = df
        logger.debug(f"csv file shape: {df.shape}")
        assert df.shape[1] == len(gene_name_mask) and df.shape[1] == len(obs_name_mask)
        assert df.shape[0] == len(sample_mask)

        # get vocab list
        vocab = self.vocab.to_itos()
        assert len(vocab)>0
        # get X info
        original_gene_list = df.columns[gene_name_mask].to_list()
        logger.debug(f"In original adata with gene {len(original_gene_list)}")
        # filter out adata not in vocab
        filted_gene_list = [gene for gene in original_gene_list if gene in vocab]
        filted_gene_nb = len(filted_gene_list)
        # get samples needed
        new_df=df.iloc[sample_mask]
        logger.debug(f"new_df shape: {new_df.shape}")
        logger.debug(f"new_df keys: {new_df.keys()}")
        sample_nb = new_df.shape[0]
        # get obs for those samples
        obs = new_df.iloc[:,obs_name_mask]
        # get X
        gene_name = []
        counts = lil_matrix((sample_nb,filted_gene_nb),dtype=np.float32)
        for i in range(len(filted_gene_list)):
            if i % 2000 == 0: logger.debug(f"processing {i}/{len(original_gene_list)}")
            counts[:,i] = new_df[filted_gene_list[i]]
            gene_name.append(filted_gene_list[i])


        counts = counts.tocsr()
        logger.debug(f"convert to anndata..")
        new = ad.AnnData(X=counts)
        new.var_names = gene_name
        obs_names = pd.Index(new_df[sample_id_idx].to_list())

        new.obs = obs
        new.obs_names = obs_names
        #new.uns = {'log1p': {'base': 2}} #data.uns
        self.adata = new

    ##############################################################################################################
    #  save data
    #############################################################################################################
    def save_adata(self, loc:str):
        """
        Save data to anndata object
        """
        logger.info(f"Save data to anndata object.")
        self.adata.write(loc)

    ##############################################################################################################
    #  data process
    #############################################################################################################
    def preprocess(self,batch_key: Optional[str] = None):
        self.preprocessor = scPreprocessor(self.para)
        self.adata = self.preprocessor(self.adata,batch_key=batch_key)

    def postprocess(self):
        # check if need to map string labels to int
        self.map_str_labels_to_int()

        if self.para.tokenize_name == "scBERT":
            self.postprocess_worker = scBERTPostprocessor(self.para,self.vocab)
            
        else:
            raise ValueError(f"tokenize_name {self.para.tokenize_name} not supported.")
        train_dataset,valid_dataset,w = self.postprocess_worker.run(self.adata)
        return train_dataset,valid_dataset,w
    
    def map_str_labels_to_int(self):
        """
        map string labels to int
        """
        if self.para.auto_map_str_labels:
            logger.info(f"Map string labels to int automatically.")
            # 补充 对于原本就是离散字符标签的数据集，需要map到数字并保存map_dict
            key_name = self.para.label_key
            # 把unique的label转化为map_dict
            cat_unique = self.adata.obs[key_name].unique()
            if type(cat_unique) ==np.ndarray:
                name_list = cat_unique.tolist()
            else:
                name_list = self.adata.obs[key_name].unique().to_list()
            if self.para.map_dict is None:
                map_dict = {value: index for index, value in enumerate(name_list)}
            else:
                assert len(self.para.map_dict) == len(name_list) # map_dict should have the same length as name_list
                print(f"map_dict: {self.para.map_dict}, name_list: {name_list}")
                map_dict = self.para.map_dict
            logger.info(f"Mapping from {map_dict}")
            self.adata.obs[key_name]=self.adata.obs[key_name].map(map_dict)
            self.para.map_dict = map_dict