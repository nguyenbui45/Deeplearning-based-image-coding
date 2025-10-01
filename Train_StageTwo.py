import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import List, Tuple, Dict
import logging
import torchvision.models as tv_models
from collections import OrderedDict
from models import Analysis_transform,Synthesis_transform,Entropy,Hyper_analysis,Hyper_synthesis,Latent_Space_Transform,YOLOv3
from utils import RateDistortionLoss
import models.yolo_model
import utils.metrics
from utils import PerformanceLogger,BestLogger
import utils
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch, os, math

import math
import numpy as np


class TrainStageTwo(pl.LightningModule):
    def __init__(self,hparams,saved_model_paths):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self.save_hyperparameters(hparams)
        self.saved_model_paths = saved_model_paths
        # components
        self.AnalysisEncoder = Analysis_transform(hparams.latent_channel).to(torch.float32)
        self.SynthesisTransform = Synthesis_transform(hparams.latent_channel).to(torch.float32)
        self.HyperAnalysis = Hyper_analysis(hparams.latent_channel).to(torch.float32)
        self.HyperSynthesis = Hyper_synthesis(hparams.latent_channel).to(torch.float32)
        self.Entropy1 = Entropy(hparams.base_channel,hparams.base_channel).to(torch.float32)
        self.Entropy2 = Entropy(hparams.enhancement_channel,hparams.enhancement_channel).to(torch.float32)
        self.LST = Latent_Space_Transform(num_filters=hparams.base_channel,out_channel=hparams.LST_out_channel).to(torch.float32)
        self.YOLOv3 = YOLOv3(start_idx=hparams.YOLO_evaluation_layer,image_size=hparams.train_patch_size).to(torch.float32)
        self.YOLOv3.eval()
        
        # load the weights trained from stage I
        self._load_model_weight()
        
        # loss
        self.RD_loss = RateDistortionLoss()
        
        # logging 
        # self.train_performance_logger = PerformanceLogger(self.saved_model_paths.train_stageII,'train')
        # self.validation_performance_logger = PerformanceLogger(self.saved_model_paths.validation_stageII,'validation')
        # self.best_validation_logger = BestLogger(self.saved_model_paths.best_validation_stageII,'validation')
        
        self.train_loss = torchmetrics.MeanMetric()
        self.train_R = torchmetrics.MeanMetric()
        self.train_D = torchmetrics.MeanMetric()
        self.train_bitrate = torchmetrics.MeanMetric()

        self.val_loss = torchmetrics.MeanMetric()
        self.val_R = torchmetrics.MeanMetric()
        self.val_D = torchmetrics.MeanMetric()
        self.val_bitrate = torchmetrics.MeanMetric()
             
        return 
    
    
    def training_step(self, batch:Tuple[List[torch.Tensor]], batch_idx):
        x = batch
        
        y = self.AnalysisEncoder(x)
        z = self.HyperAnalysis(y)
        
        quant_noise_y = torch.empty_like(y).uniform_(-0.5, 0.5)

        quant_noise_z = torch.empty_like(z).uniform_(-0.5, 0.5)
        
        
        # Quantization
        quantized_y = y + quant_noise_y
        quantized_z = z + quant_noise_z
        
        quantized_y1 = quantized_y[:,:self.hparams.base_channel,:,:] # extract base channel
        quantized_y2 = quantized_y[:,self.hparams.base_channel:,:,:] # extract enhancement channel
        
        
        h = self.HyperSynthesis(quantized_z)
        h1 = h[:,:quantized_y1.size(1)*2,:,:]
        h2 = h[:,quantized_y1.size(1)*2:,:,:]
        
        mean_y1,scale_y1 = self.Entropy1(h1,quantized_y1)
        mean_y2,scale_y2 = self.Entropy2(h2,quantized_y2)
        
        scale_y1 = F.softplus(scale_y1) + 1e-6 # guarentee positive std
        scale_y2 = F.softplus(scale_y2) + 1e-6 # guarentee positive std
        
        # reconstruct image
        x_hat = self.SynthesisTransform(torch.cat([quantized_y1,quantized_y2],dim=1))
        
        # LST 
        F13_tilde = self.LST(quantized_y1)
        F13 = models.yolo_model.detect_image(self.YOLOv3,x_hat,'train')
        
        # calculate loss
        loss,R,D = self.RD_loss(x, x_hat,F13,F13_tilde,self.hparams.gamma, [mean_y1,mean_y2], [scale_y1,scale_y2],[quantized_y1,quantized_y2], quantized_z, self.hparams.lam[3])
        
        bitrate_object_detection = (utils.metrics.latent_rate(quantized_y1,mean_y1,scale_y1) + utils.metrics.hyperlatent_rate(quantized_z,'sigmoid')) / (x.size(0) * x.size(2) * x.size(3))

        self.val_loss.update(loss.detach())
        self.val_R.update(R.detach())
        self.val_D.update(D.detach())
        self.val_bitrate.update(bitrate_object_detection.detach())

        self.print(f"Epoch {self.current_epoch} | Step {self.global_step} | Loss {loss.item():.4f} | Rate {R.item():.4f} + Distortion {D.item():.4f}")
        
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x = batch
        
        y = self.AnalysisEncoder(x)
        z = self.HyperAnalysis(y)
        
        # Quantization
        quantized_y = torch.round(y)
        quantized_z = torch.round(z)
        
        quantized_y1 = quantized_y[:,:self.hparams.base_channel,:,:] # extract base channel
        quantized_y2 = quantized_y[:,self.hparams.base_channel:,:,:] # extract enhancement channel
        
        h = self.HyperSynthesis(quantized_z)
        h1 = h[:,:quantized_y1.size(1)*2,:,:]
        h2 = h[:,quantized_y1.size(1)*2:,:,:]
        
        mean_y1,scale_y1 = self.Entropy1(h1,quantized_y1)
        mean_y2,scale_y2 = self.Entropy2(h2,quantized_y2)
        
        scale_y1 = F.softplus(scale_y1) + 1e-6 # guarentee positive std
        scale_y2 = F.softplus(scale_y2) + 1e-6 # guarentee positive std
        
        # reconstruct image
        x_hat = self.SynthesisTransform(torch.cat([quantized_y1,quantized_y2],dim=1))
        
        # LST 
        F13_tilde = self.LST(quantized_y1)
        F13 = models.yolo_model.detect_image(self.YOLOv3,x_hat,'train')
        
        # calculate loss
        loss,R,D = self.RD_loss(x, x_hat,F13,F13_tilde,self.hparams.gamma, [mean_y1,mean_y2], [scale_y1,scale_y2],[quantized_y1,quantized_y2], quantized_z, self.hparams.lam[3])
        
        bitrate_object_detection = (utils.metrics.latent_rate(quantized_y1,mean_y1,scale_y1) + utils.metrics.hyperlatent_rate(quantized_z,'sigmoid')) / (x.size(0) * x.size(2) * x.size(3))
        
        self.val_loss.update(loss.detach())
        self.val_R.update(R.detach())
        self.val_D.update(D.detach())
        self.val_bitrate.update(bitrate_object_detection.detach())
        
        self.log("val/epoch_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    
    def on_train_epoch_end(self):
        self.log("train/epoch_loss", self.train_loss.compute(), prog_bar=True, sync_dist=True)
        self.log("train/epoch_R", self.train_R.compute(), sync_dist=True)
        self.log("train/epoch_D", self.train_D.compute(), sync_dist=True)
        self.log("train/epoch_bitrate", self.train_bitrate.compute(), sync_dist=True)
        self.train_loss.reset()
        self.train_R.reset()
        self.train_D.reset()
        self.train_bitrate.reset()    

    
    def on_validation_epoch_end(self):
        
        self.log("val/epoch_R", self.val_R.compute(), sync_dist=True)
        self.log("val/epoch_D", self.val_D.compute(), sync_dist=True)
        self.log("val/epoch_bitrate", self.val_bitrate.compute(), sync_dist=True)
        self.val_loss.reset()
        self.val_R.reset()
        self.val_D.reset()
        self.val_bitrate.reset()
    
    
    def configure_optimizers(self) -> Tuple[Dict, Dict]:
        
        update_params_list = [
            {'params': self.AnalysisEncoder.parameters()},
            {'params': self.SynthesisTransform.parameters()}, # for supcon distilled encoder
            {'params': self.HyperAnalysis.parameters()},
            {'params': self.HyperSynthesis.parameters()},
            {'params': self.Entropy1.parameters()},
            {'params': self.Entropy2.parameters()},
            {'params': self.LST.parameters()}
        ]
        
        # polynomial decay LR every 10 epoches
        
        total_decay_steps = self.hparams.num_epoch_stageII // self.hparams.decay_every
        def poly_decay(decay_step: int):
            # clamp progress in [0,1]
            if total_decay_steps == 0:
                return 1.0
            progress = min(decay_step / total_decay_steps, 1.0)
            return (1.0 - progress) ** self.hparams.power
        
        if self.hparams.optimizer_stageII == 'Adam':
            optimizer = torch.optim.Adam(update_params_list,lr=self.hparams.LR_stageII)
        elif self.hparams.optimizer_stageII == 'AdamW':
            optimizer = torch.optim.AdamW(update_params_list,lr=self.hparams.LR_stageII)

        if self.hparams.scheduler_stageII == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.num_epoch_stageII)
        elif self.hparams.scheduler_stageII == 'Polynomial':
             scheduler  = LambdaLR(optimizer, lr_lambda=poly_decay)
        elif self.hparams.scheduler_stageII == 'None':
            return {"optimizer": optimizer}

        return ({
            "optimizer": optimizer, 
            "lr_scheduler":{
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 10,
                "name": "poly_every_10"
            }
        })
        
            
    def _load_model_weight(self):
        AnalysisEncoder_state = torch.load(self.saved_model_paths.checkpoint_dir + '/AnalysisEncoder_stageI.pth')
        SynthesisTransform_state = torch.load(self.saved_model_paths.checkpoint_dir + '/SynthesisTransform_stageI.pth')
        HyperAnalysis_state = torch.load(self.saved_model_paths.checkpoint_dir + '/HyperAnalysis_stageI.pth')
        HyperSynthesis_state = torch.load(self.saved_model_paths.checkpoint_dir + '/HyperSynthesis_stageI.pth')    
        Entropy1_state = torch.load(self.saved_model_paths.checkpoint_dir + '/Entropy1_stageI.pth')
        Entropy2s_state = torch.load(self.saved_model_paths.checkpoint_dir + '/Entropy2_stageI.pth')
        LST_state = torch.load(self.saved_model_paths.checkpoint_dir + '/LST_stageI.pth')           
        
        self.AnalysisEncoder.load_state_dict(AnalysisEncoder_state)
        self.SynthesisTransform.load_state_dict(SynthesisTransform_state)
        self.HyperAnalysis.load_state_dict(HyperAnalysis_state)  
        self.HyperSynthesis.load_state_dict(HyperSynthesis_state)
        self.Entropy1.load_state_dict(Entropy1_state)
        self.Entropy2.load_state_dict(Entropy2s_state)  
        self.LST.load_state_dict(LST_state)       
        
        
        

class SaveComponentsOnBest(Callback):
    def __init__(self, monitor: str, mode: str, out_dir: str, top_k: int = 1):
        assert mode in {"min", "max"}
        self.monitor = monitor
        self.mode = mode
        self.out_dir = out_dir
        self.top_k = top_k
        self.best = math.inf if mode == "min" else -math.inf
        self.saved = []  # [(score, epoch, paths_dict), ...]
        self.saved_once = False

        os.makedirs(out_dir, exist_ok=True)

    def _is_improved(self, val):
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return False
        return (val < self.best) if self.mode == "min" else (val > self.best)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        val = trainer.callback_metrics.get(self.monitor)
        if isinstance(val, torch.Tensor):
            val = val.item()
        if val is None or not math.isfinite(val):
            return
        if not self._is_improved(val):
            return

        self.best = val

        # save all components (atomic replace)
        def to_cpu_sd(m):
            return {k: v.detach().cpu() for k, v in m.state_dict().items()}
        comps = {
            "AnalysisEncoder": pl_module.AnalysisEncoder,
            "SynthesisTransform": pl_module.SynthesisTransform,
            "HyperAnalysis": pl_module.HyperAnalysis,
            "HyperSynthesis": pl_module.HyperSynthesis,
            "Entropy1": pl_module.Entropy1,
            "Entropy2": pl_module.Entropy2,
            "LST": pl_module.LST,
        }
        for name, module in comps.items():
            fpath = os.path.join(self.out_dir, f"{name}_stageII.pth")
            tmp = fpath + ".tmp"
            torch.save(to_cpu_sd(module), tmp)
            os.replace(tmp, fpath)

        self.saved_once = True
    
    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        if self.saved_once:
            return
        def to_cpu_sd(m): 
            return {k: v.detach().cpu() for k, v in m.state_dict().items()}
        comps = {
            "AnalysisEncoder": pl_module.AnalysisEncoder,
            "SynthesisTransform": pl_module.SynthesisTransform,
            "HyperAnalysis": pl_module.HyperAnalysis,
            "HyperSynthesis": pl_module.HyperSynthesis,
            "Entropy1": pl_module.Entropy1,
            "Entropy2": pl_module.Entropy2,
            "LST": pl_module.LST,
        }
        for name, module in comps.items():
            fpath = os.path.join(self.out_dir, f"{name}_stageI.pth")
            tmp = fpath + ".tmp"
            torch.save(to_cpu_sd(module), tmp)
            os.replace(tmp, fpath)
        self.saved_once = True