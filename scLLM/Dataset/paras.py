from typing import Any, Optional, Union
import numpy as np
import attr

@attr.s(auto_attribs=True)
class Dataset_para:
    """
    Dataset parameters
    """
    #--------> data loading steps
    var_idx:str = None # gene i.e. "gene_syb"
    obs_idx:str = None # labels i.e. "pseudotimes"
    # for gene2vec encoding or other embedding methods the entire vocabulary is needed
    vocab_loc: str = None
    gene_vocab:list = None #['gene1','gene2',...] 

    #--------> preprocessing steps
    #(:class:`str`, optional) The key of :class:`~anndata.AnnData` to use for preprocessing.
    use_key: Optional[str] = "X" 
    
    #---> gene filter and cell filter
    #filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``): Whther to filter genes by counts, if :class:`int`, filter genes with counts
    filter_gene_by_counts: Union[int, bool] = False #False
    #filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``): Whther to filter cells by counts, if :class:`int`, filter cells with counts
    filter_cell_by_counts: Union[int, bool] = 200#False
    
    #----> normalization
    #(:class:`float` or :class:`bool`, default: ``1e4``): Whether to normalize the total counts of each cell to a specific value.
    normalize_total: Union[float, bool] = 1e4
    #(:class:`str`, default: ``"X_normed"``): The key of :class:`~anndata.AnnData` to store the normalized data. If :class:`None`, will use normed data to replce the :attr:`use_key`.
    result_normed_key: Optional[str] = "X_normed"
    #---> log1p transform
    #(:class:`bool`, default: ``True``): Whether to apply log1p transform to the normalized data.
    log1p: bool = True #False
    #(:class:`str`, default: ``"X_log1p"``): The key of :class:`~anndata.AnnData` to store the log1p transformed data.
    result_log1p_key: str = "X_log1p"
    #(:class:`float`, default: ``2.0``): The base para of log1p transform funciton.
    log1p_base: float = 2.0
    #--->hvg
    #(:class:`int` or :class:`bool`, default: ``False``): Whether to subset highly variable genes.
    subset_hvg: Union[int, bool] = False
    #(:class:`str`, optional): The key of :class:`~anndata.AnnData` to use for calculating highly variable genes. If :class:`None`, will use :attr:`adata.X`.
    hvg_use_key: Optional[str] = None
    #(:class:`str`, default: ``"seurat_v3"``): The flavor of highly variable genes selection. See :func:`scanpy.pp.highly_variable_genes` for more details.
    hvg_flavor: str = "seurat_v3"
    #---->bined data part
    #(:class:`int`, optional): Whether to bin the data into discrete values of number of bins provided.
    binning: Optional[int] = None
    #(:class:`str`, default: ``"X_binned"``): The key of :class:`~anndata.AnnData` to store the binned data.
    result_binned_key: str = "X_binned"

    #--------> tokenization steps
    #---->tokenize name
    tokenize_name: str = "scBERT" #["scBERT", ]
    #---->tokenize
    return_pt: bool = True
    append_cls: bool = True
    include_zero_gene: bool = False
    cls_token: str = "<cls>"
    #----> add pad
    max_len: int = 16000
    pad_token: str = "<pad>"
    pad_value: int = -2
    cls_appended: bool = True
    #----> mask
    mask_ratio: float = 0.15
    mask_value: int = -1

    #--------> data saving steps
    preprocessed_loc: str = None

    #--------> dataset steps
    data_layer_name:str = "X_log1p" # "X","X_normed","X_binned","X_log1p"
    label_key: str=None
    batch_label_key: str=None
    # number of categories for classification
    cls_nb:int = None
    #-> for binarize label
    binarize:str="equal_instance" # "equal_instance" or "equal_width"
    bins:np.ndarray=None
    #bin_nb: int=None
    bin_min:float=None
    bin_max:float=None
    save_in_obs:bool=True # save binarize label in obs parameter in anndata

    auto_map_str_labels:bool=True # whether to map string labels to int
    map_dict:dict=None # map string labels to int

    #-> split train test
    n_splits:int=1
    test_size:float=0.2
    random_state:int=2023

    shuffle:bool=True
    sort_seq_batch:bool=False
    