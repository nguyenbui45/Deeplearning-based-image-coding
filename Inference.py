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
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import math
import numpy as np


class Inference(pl.LightningModule):
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
        
        self.mAP_metrics = MeanAveragePrecision(box_format="xyxy",    # your format
                                                iou_type="bbox",      # boxes, not masks
                                                class_metrics=False   # or True if you want AP per class too
        )
                                                    
    
    
    def validation_step(self, batch, batch_idx):
        x = batch['pixels']
        bboxes = batch['bboxes']
        class_ids = batch['class_ids']
        
        # y = self.AnalysisEncoder(x)
        # z = self.HyperAnalysis(y)
        
        # # Quantization
        # quantized_y = torch.round(y)
        # quantized_z = torch.round(z)
        
        # quantized_y1 = quantized_y[:,:self.hparams.base_channel,:,:] # extract base channel
        # quantized_y2 = quantized_y[:,self.hparams.base_channel:,:,:] # extract enhancement channel
        
        # h = self.HyperSynthesis(quantized_z)
        # h1 = h[:,:quantized_y1.size(1)*2,:,:]
        # h2 = h[:,quantized_y1.size(1)*2:,:,:]
        
        # mean_y1,scale_y1 = self.Entropy1(h1,quantized_y1)
        # mean_y2,scale_y2 = self.Entropy2(h2,quantized_y2)
        
        # scale_y1 = F.softplus(scale_y1) + 1e-6 # guarentee positive std
        # scale_y2 = F.softplus(scale_y2) + 1e-6 # guarentee positive std
        
        # # reconstruct image
        # x_hat = self.SynthesisTransform(torch.cat([quantized_y1,quantized_y2],dim=1))
        
        # # LST 
        # F13_tilde = self.LST(quantized_y1)
        # base_channel_detections = models.yolo_model.detect_image(self.YOLOv3,F13_tilde,'inference')
        
        # full_channel_detections = models.yolo_model.detect_image(self.YOLOv3,x_hat,'inference_baseline')
        
        original_detections = models.yolo_model.detect_image(self.YOLOv3,x,'inference_baseline')
        device = self.device
        detections = utils.metrics.pack_yolo_preds_to_tm(original_detections,device=device)
        gt = utils.metrics.pack_targets_to_tm(bboxes,class_ids,device=device)
        
        
        self.mAP_metrics.to(device)
        self.mAP_metrics.update(detections, gt)
        
        # gt = utils.metrics.make_gt(bboxes,class_ids,difficults,crowds)
        
        # base_channel_mAP = utils.metrics.mAP(base_channel_detections,gt,80)
        # full_channel_mAP = utils.metrics.mAP(full_channel_detections,gt,80)
        # orignal_mAP = utils.metrics.mAP(original_detections,gt,80)
        
        # # use only base bitstream y1
        # bitrate_base_channel_detection = (utils.metrics.latent_rate(quantized_y1,mean_y1,scale_y1) + utils.metrics.hyperlatent_rate(quantized_z,'sigmoid')) / (x.size(0) * x.size(2) * x.size(3))
        
        # # use all bitstream
        # bitrate_full_object_detection  = (utils.metrics.latent_rate(quantized_y1,mean_y1,scale_y1) + utils.metrics.hyperlatent_rate(quantized_z,'sigmoid') + utils.metrics.latent_rate(quantized_y2,mean_y2,scale_y2) ) / (x.size(0) * x.size(2) * x.size(3))
        
             
        # self.log("val/bpp_base", bitrate_base_channel_detection, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/bpp_full", bitrate_full_object_detection, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/mAP_base", base_channel_mAP, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/mAP_full", full_channel_mAP, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/original_mAP", orignal_mAP, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return 
    
    
    def on_validation_epoch_end(self):
        res = self.mAP_metrics.compute()  # dict with mAP, mAP_50, mAP_75, etc.
        self.log("val/mAP",     res["map"],     prog_bar=True, sync_dist=True)
        self.log("val/mAP_50",  res["map_50"],  sync_dist=True)
        self.log("val/mAP_75",  res["map_75"],  sync_dist=True)
        # reset for next epoch
        self.mAP_metrics.reset()
         
    def _load_model_weight(self):
        AnalysisEncoder_state = torch.load(self.saved_model_paths.checkpoint_dir + '/AnalysisEncoder_stageII.pth')
        SynthesisTransform_state = torch.load(self.saved_model_paths.checkpoint_dir + '/SynthesisTransform_stageII.pth')
        HyperAnalysis_state = torch.load(self.saved_model_paths.checkpoint_dir + '/HyperAnalysis_stageII.pth')
        HyperSynthesis_state = torch.load(self.saved_model_paths.checkpoint_dir + '/HyperSynthesis_stageII.pth')    
        Entropy1_state = torch.load(self.saved_model_paths.checkpoint_dir + '/Entropy1_stageII.pth')
        Entropy2_state = torch.load(self.saved_model_paths.checkpoint_dir + '/Entropy2_stageII.pth')
        LST_state = torch.load(self.saved_model_paths.checkpoint_dir + '/LST_stageII.pth')           
        
        self.AnalysisEncoder.load_state_dict(AnalysisEncoder_state)
        self.SynthesisTransform.load_state_dict(SynthesisTransform_state)
        self.HyperAnalysis.load_state_dict(HyperAnalysis_state)  
        self.HyperSynthesis.load_state_dict(HyperSynthesis_state)
        self.Entropy1.load_state_dict(Entropy1_state)
        self.Entropy2.load_state_dict(Entropy2_state)  
        self.LST.load_state_dict(LST_state)
        
        self.AnalysisEncoder.eval()
        self.SynthesisTransform.eval()
        self.HyperAnalysis.eval()
        self.HyperSynthesis.eval()
        self.Entropy1.eval()
        self.Entropy2.eval()
        self.LST.eval()
        
        