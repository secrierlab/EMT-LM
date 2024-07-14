from typing import Any, Optional, Union
from functools import partial
import torch

from scLLM import logger
from scLLM.Trainer.metrics import MetricsFactory

class Trainer_para:
    # need to set when init
    project: str = None # project name
    entity: str = None # entity name
    exp_name: str = None # experiment name
    #-----> dataset
    task_type: str = None # "classification","regression"
    class_nb: int = None # number of classes
    batch_size: int = 4 # batch size
    shuffle: bool = True
    num_workers:int = 8
    additional_dataloader_para: dict = {}
    #-----> model
    model_name: str = None #"scFormer","scBERT"
    ckpt_folder:str = None   # checkpoint file location
    pre_trained:str = None # pre-trained ckpt location

    #-----> optimizer and loss
    optimizers:list = [torch.optim.Adam,
                       ] # list of optimizer
    optimizer_paras:list =[
        {"lr":1e-4},# for first optimizer
    ] # list of optimizer paras dict

    schedulers:list = [] # list of scheduler
    scheduler_paras:list = [] # list of scheduler paras dict

    loss: Any = None # loss function class but not create instance here
                     # create instance in pl_basic class with Trainer_para.loss()
    loss_para: dict = None # loss function paras

    additional_loss: list = [] # additional loss fn and paras, each item is a dict {"name":loss_name,"fn":loss_fn,"paras":loss_paras}

    clip_grad: int = int(1e6) # clip gradient

    #-----> training
    max_epochs: int = 100 # max epochs

    #-----> metrics
    metrics_names: list = [] # list of metrics name
    metrics_paras: dict = None#{} # dict of metrics paras specify "classification" or "regression
    #-----> pytorch lightning paras
    trainer_output_dir: str = None # output dir for pytorch lightning
    # additional pytorch lightning paras
    additional_pl_paras={
                #---------> paras for pytorch lightning trainner
                "accumulate_grad_batches":60, # less batch size need accumulated grad
                "accelerator":"auto",#accelerator='gpu', devices=1,
            }

    with_logger:str = None # "wandb",
    wandb_api_key:str = None # wandb api key

    save_ckpt:bool = True # save checkpoint or not
    
    #debug: try to add formated name to ckpt
    ckpt_format:str = "{epoch:02d}-{accuracy_val:.2f}" # check_point format 
    ckpt_para = { #-----------> paras for pytorch_lightning.callbacks.ModelCheckpoint
                    "save_top_k":1,
                    "mode":"max",
                   "monitor":"accuracy_val",}
    
    
    #-----> functions
    def init_opts_and_schs(self):
        """
        init optimizers
        """
        assert len(self.optimizers) >0
        logger.info("init optimizers...")
        logger.debug(f"optimizers: {self.optimizers} with paras: {self.optimizer_paras}")
        self.opt_instances = []
        self.sch_instances = []
        # partial init opts
        for opt,opt_para in zip(self.optimizers,self.optimizer_paras):
            partial_opt = partial(opt, **opt_para)
            self.opt_instances.append(partial_opt)
        # partial init schs
        logger.debug(f"schs: {self.schedulers} with paras: {self.scheduler_paras}")
        if self.schedulers is not None or len(self.schedulers) >0:
            assert len(self.schedulers) == len(self.optimizers)
            for sch,sch_para in zip(self.schedulers,self.scheduler_paras):
                partial_sch = partial(sch, **sch_para)
                self.sch_instances.append(partial_sch)
        
    def init_metrics_factory(self):
        """
        init metrics factory
        """
        add_paras={}
        logger.info("init metrics factory...")
        logger.debug(f"metrics_names: {self.metrics_names} with paras: {self.metrics_paras}")
        if self.metrics_paras is not None:
            add_paras["metrics_paras"]=self.metrics_paras
        self.metrics_factory = MetricsFactory(n_classes=self.class_nb,
                                              metrics_names=self.metrics_names,
                                              **add_paras)
        self.metrics_factory.get_metrics(self.task_type)
