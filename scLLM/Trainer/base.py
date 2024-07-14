import os
import torch
import torch.nn as nn
import random
import pytorch_lightning as pl

from scLLM.Trainer.paras import Trainer_para
from scLLM import logger
class pl_basic(pl.LightningModule):
    def __init__(self,
                trainer_paras:Trainer_para,# trainer para
                model_paras, # model para
                ):
        super(pl_basic,self).__init__()
        """
        A basic class for different PL protocols:
        """
        logger.info("init pytorch-lightning basic part...")
        #----> paras
        self.trainer_paras = trainer_paras
        self.model_paras = model_paras
        #----> create model
        self.model = None

        #----> create loss
        self.create_loss()

        #----> create metrics
        self.create_metrics()

    def create_model(self):
        """
        create model instance
        """
        pass
    
    def create_loss(self):
        """
        create loss instance
        """
        logger.info("create loss instance...")
        logger.debug(f"loss: {self.trainer_paras.loss}")
        if self.trainer_paras.loss_para is None:
            self.loss_fn = self.trainer_paras.loss()
        else:
            self.loss_fn = self.trainer_paras.loss(**self.trainer_paras.loss_para)

    def configure_optimizers(self):
        """
        create optimizer and scheduler

        return [optimizer],[scheduler]
        """
        self.trainer_paras.init_opts_and_schs()
        opt_list = [opt(self.model.parameters()) \
                        for opt in self.trainer_paras.opt_instances]
        if self.trainer_paras.sch_instances is not None \
                or len(self.trainer_paras.sch_instances) >0:
            sch_list = [sch(opt) for sch,opt in \
                            zip(self.trainer_paras.sch_instances,opt_list)]
            return opt_list,sch_list
        else:
            return opt_list

    def create_metrics(self,):
        """
        create metrics instance
        """
        self.trainer_paras.init_metrics_factory()
        self.bar_metrics = self.trainer_paras.metrics_factory.metrics["metrics_on_bar"]
        self.valid_metrics = self.trainer_paras.metrics_factory.metrics["metrics_template"].clone(prefix = 'val_')
        self.train_metrics = self.trainer_paras.metrics_factory.metrics["metrics_template"].clone(prefix = 'train_')

    def collect_step_output(self,key,out,dim=None):
        if dim is None:
            return torch.cat([x[key] for x in out])
        else:
            return torch.cat([x[key] for x in out],dim=dim)

    def build_trainer(self):
        logger.info(f"build trainer for model.")
        trainer_additional_dict = self.trainer_paras.additional_pl_paras
        callbacks_list = []

        # 3. clip gradient
        if self.trainer_paras.clip_grad is not None:
            logger.debug(f"clip gradient with value {self.trainer_paras.clip_grad}")
            trainer_additional_dict.update({"gradient_clip_val":self.trainer_paras.clip_grad,
                                            "gradient_clip_algorithm":"value"})

        # 4. create learning rate logger
        from pytorch_lightning.callbacks import LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks_list.append(lr_monitor)
        if self.trainer_paras.with_logger=="wandb":
            logger.debug(f"with logger wandb")
            # 4. Create wandb logger
            from pytorch_lightning.loggers import WandbLogger
            os.environ["WANDB_API_KEY"]=self.trainer_paras.wandb_api_key

            wandb_logger = WandbLogger(project=self.trainer_paras.project,
                                        entity=self.trainer_paras.entity,
                                        name=self.trainer_paras.exp_name)
            trainer_additional_dict.update({"logger":wandb_logger})

        if self.trainer_paras.save_ckpt:
            # 4. check point
            from pytorch_lightning.callbacks import ModelCheckpoint
            # init ckpt related paras
            ckpt_paras = self.trainer_paras.ckpt_para
            ckpt_name = self.trainer_paras.ckpt_format
            ckpt_dir = self.trainer_paras.ckpt_folder+self.trainer_paras.exp_name+"/"
            logger.debug(f"for exp {self.trainer_paras.exp_name} \
                                        Checkpoint with paras {ckpt_paras}")
            checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                                filename=ckpt_name,
                                                 **ckpt_paras,
                                                )
            
            logger.info(f"Best model will be saved at {ckpt_dir} as {ckpt_name}")
            callbacks_list.append(checkpoint_callback)
            
        if len(callbacks_list)>=1: trainer_additional_dict.update(
                                            {"callbacks":callbacks_list})
        # 4. Trainer and fit
        self.trainer = pl.Trainer(default_root_dir=self.trainer_paras.trainer_output_dir,
                                max_epochs=self.trainer_paras.max_epochs,
                                **trainer_additional_dict
                                )
        logger.info(f"trainer is built.")
    #########################################################################
    #   need to customize for different algorithms
    #########################################################################
    #-----> define for each algorithm
    def train_data_preprocess(self,batch):
        """
        data preprocess
        """
        data, label = batch
        label = label.squeeze(0).type(torch.LongTensor)
        return data, label

    def train_post_process(self,logits,label,loss):
        """
        post process
        """
        #---->metrics step
        softmax = nn.Softmax(dim=-1)
        prob = softmax(logits)
        final = prob.argmax(dim=-1)
        out = {"logits":logits,"Y_prob":prob,
               "Y_hat":final,"label":label,
               "loss":loss}
        return out

    def val_data_preprocess(self,batch):
        """
        data preprocess
        """
        data, label = batch
        label = label.squeeze(0).type(torch.LongTensor)
        return data, label

    def val_post_process(self,logits,label,loss):
        """
        post process
        """
        #---->metrics step
        softmax = nn.Softmax(dim=-1)
        prob = softmax(logits)
        final = prob.argmax(dim=-1)
        out = {"logits":logits,"Y_prob":prob,
               "Y_hat":final,"label":label,
               "loss":loss}
        return out
    
    def log_val_metrics(self,outlist,bar_name:str):
        probs = self.collect_step_output(key="Y_prob",out=outlist,dim=0)
        max_probs = self.collect_step_output(key="Y_hat",out=outlist,dim=0)
        target = self.collect_step_output(key="label",out=outlist,dim=0)
        #----> log part
        self.log(bar_name, self.bar_metrics(probs, target.squeeze()), 
                            prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)
        
    def log_train_metrics(self,outlist,bar_name:str):
        probs = self.collect_step_output(key="Y_prob",out=outlist,dim=0)
        max_probs = self.collect_step_output(key="Y_hat",out=outlist,dim=0)
        target = self.collect_step_output(key="label",out=outlist,dim=0)
        #----> log part
        self.log(bar_name, self.bar_metrics(probs, target.squeeze()), 
                            prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.train_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)    
    #-----> define for each algorithm

    #########################################################################
    #   pytorch lightning style
    #########################################################################
    def training_step(self, batch, batch_idx):
        #---->data preprocess
        data,label =self.train_data_preprocess(batch)

        #---->forward step
        logits = self.model(data)

        #---->loss step
        #debug: change to LongTensor to avoid error
        label = label.type(torch.LongTensor).to(logits.device)
        loss = self.loss_fn(logits, label)

        #---->post process
        out = self.train_post_process(logits,label,loss)
        return out

    def training_epoch_end(self, training_step_outputs):
        #print("#################### train epoch end ####################")
        self.log_train_metrics(training_step_outputs,
            bar_name = self.trainer_paras.metrics_factory.metrics_names[0]+"_train")

    def validation_step(self, batch, batch_idx):
        #---->data preprocess
        data,label =self.val_data_preprocess(batch)

        #---->forward step
        logits = self.model(data)

        #---->loss step
        #debug: change to LongTensor to avoid error
        label = label.type(torch.LongTensor).to(logits.device)
        #print(logits.type(), label.type())
        loss = self.loss_fn(logits, label)

        #---->post process
        out = self.val_post_process(logits,label,loss)
        return out
    
    
    def validation_epoch_end(self, validation_step_outputs):
        self.log_val_metrics(validation_step_outputs,
            bar_name = self.trainer_paras.metrics_factory.metrics_names[0]+"_val")

