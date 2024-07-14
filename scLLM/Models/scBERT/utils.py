import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scLLM import logger
def transfer_gene2vec_as_weight(emb_loc:str,np_loc:str=None):
    # transfer gene2vec as weight
    # currently use https://github.com/jingcheng-du/Gene2vec.git
    #in:
    # emb_loc: gene2vec model location
    # np_loc: numpy file location
    try:
        from gensim.models import Word2Vec, KeyedVectors
        gene2vec_model = KeyedVectors.load_word2vec_format(emb_loc)
        logger.info(f"Loaded gene2vec model with {len(gene2vec_model.index_to_key)} genes")
        if np_loc is not None:
            # save as npy
            np.save(np_loc, gene2vec_model.vectors)
            logger.info(f"Saved gene2vec model as {np_loc}")
        return gene2vec_model
    except:
        raise ValueError ("Please install gensim to use gene2vec and use embedding file in  https://github.com/jingcheng-du/Gene2vec.git ")

def pre_process_sc_raw(raw_data_loc:str,name_ref:list,preprocessed_loc:str):
    """
    from single cell raw data to preprocessed data
    in:
    raw_data_loc: raw data location './data/your_raw_data.h5ad'
    name_ref: gene name reference
    preprocessed_loc: preprocessed data location './data/your_preprocessed_data.h5ad'
    
    """
    logger.info(f"read raw data from {raw_data_loc}")
    from scipy import sparse
    data = sc.read_h5ad(raw_data_loc)
    gene_nb = len(name_ref)
    counts = sparse.lil_matrix((data.X.shape[0],gene_nb),dtype=np.float32)
    obj = data.var_names.tolist()
    logger.debug(f"get gene name from reference and transfer to sparse matrix")
    for i in range(len(name_ref)):
        if i % 2000 == 0: logger.debug(f"processing {i}/{len(name_ref)}")
        if name_ref[i] in obj:
            loc = obj.index(name_ref[i])
            counts[:,i] = data.X[:,loc]
    counts = counts.tocsr()
    logger.debug(f"convert to anndata..")
    new = ad.AnnData(X=counts)
    new.var_names = name_ref
    new.obs_names = data.obs_names
    new.obs = data.obs
    new.uns = {'log1p': {'base': 2}} #data.uns
    logger.info(f"pre-process..")
    sc.pp.filter_cells(new, min_genes=200)
    sc.pp.normalize_total(new, target_sum=1e4)
    sc.pp.log1p(new, base=2)
    if preprocessed_loc is not None:
        logger.info(f"save preprocessed data to {preprocessed_loc}")
        new.write(preprocessed_loc)
    return new


import math 
import torch
from torch.optim.lr_scheduler import _LRScheduler
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

