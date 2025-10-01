import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import csv
import pandas as pd

import torchvision
import torchvision.transforms as transforms
from monai.transforms import Compose, RandFlipd, RandRotate90d,RandAdjustContrastd,NormalizeIntensityd,RandAdjustContrastd,RandGaussianNoised

import data

import Train_StageOne
import Train_StageTwo
import Inference

import yaml
import random
import configs
import datetime
import time
import os
import argparse
from tqdm import tqdm
from terminaltables import AsciiTable
import gc
import numpy as np
import json

import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
from omegaconf import OmegaConf
import warnings


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Scalable image coding')
    parser.add_argument('--project_dir',type=str,default='/home/nguyensolbadguy/Code_Directory/compression/')
    args = parser.parse_args()

    '''
    ==================== Load the paths ====================
    '''
    # dataset paths
    data_paths = OmegaConf.load('configs/config_path.yaml')

    # dataset saved model paths
    saved_model_paths = OmegaConf.load('configs/config_model_path.yaml')
    
    # load pretrain hparams
    hparams = OmegaConf.load("configs/config2layer.yaml")


    '''

    ==================== Load Data  ====================

    '''

    stageI_trainloader,stageI_validation_loader = data.get_dataloader('train_stage_I',data_paths,hparams)
    stageII_trainloader,stageII_validation_loader = data.get_dataloader('train_stage_II',data_paths,hparams)
    inference_loader= data.get_dataloader('inference_object_detection',data_paths,hparams)


    '''

    ==================== Load Logger  ====================

    '''
    train_stageI_logger = TensorBoardLogger(saved_model_paths.checkpoint_dir + '/stageI/logs/', name="stageI-train")
    train_stageII_logger = TensorBoardLogger(saved_model_paths.checkpoint_dir+ '/stageII/logs/', name="stageII-train")
    inference_logger = TensorBoardLogger(saved_model_paths.checkpoint_dir+ '/inference/logs/', name="inference")
    
    '''

    ==================== Load Model  ====================
    1. define model
    2. declare trainer
    3. fit/validate

    '''
    def pick_strategy(devices):
        return DDPStrategy(find_unused_parameters=True) if (isinstance(devices, int) and devices > 1) or (isinstance(devices, (list, tuple)) and len(devices) > 1) else "auto"
    
    strategy = pick_strategy(hparams.gpu)
    
    start_time = time.time()
    
    
    # checkpoint 
    ckpt1 = ModelCheckpoint(
    dirpath=saved_model_paths.checkpoint_dir,
    filename="stageI-{epoch:04d}-{val_epoch_loss:.6f}",
    monitor="val/epoch_loss", mode="min",
    save_top_k=1, save_last=True
    )
    
    ckpt2 = ModelCheckpoint(
    dirpath=saved_model_paths.checkpoint_dir,
    filename="stageII-{epoch:04d}-{val_epoch_loss:.6f}",
    monitor="val/epoch_loss", mode="min",
    save_top_k=1, save_last=True
    )
    

    save_comps_stageI = Train_StageOne.SaveComponentsOnBest(
        monitor="val/epoch_loss",   # must match your logged name
        mode="min",
        out_dir=saved_model_paths.checkpoint_dir,
        top_k=1                     # keep best 2 sets per component
    )
    
    
    save_comps_stageII = Train_StageTwo.SaveComponentsOnBest(
        monitor="val/epoch_loss",   # must match your logged name
        mode="min",
        out_dir=saved_model_paths.checkpoint_dir,
        top_k=1                     # keep best 2 sets per component
    )
    
    
    train_stageI_model = Train_StageOne.TrainStageOne(hparams,saved_model_paths)
    train_stageI_trainer = pl.Trainer(logger=train_stageI_logger,
                                      accelerator='gpu',
                                      devices=hparams.gpu,max_epochs=hparams.num_epoch_stageI,        enable_progress_bar=hparams.enable_progress_bar,gradient_clip_val=1.0,
                                      callbacks=[ckpt1, save_comps_stageI, LearningRateMonitor("step")],
                                      strategy=strategy)
    
    # train_stageI_trainer.fit(train_stageI_model,train_dataloaders=stageI_trainloader,val_dataloaders=stageI_validation_loader)
    
    
    
    
    train_stageII_model = Train_StageTwo.TrainStageTwo(hparams,saved_model_paths)
    
    train_stageII_trainer = pl.Trainer(logger=train_stageII_logger,
                                       accelerator='gpu',
                                       devices=hparams.gpu,max_epochs=hparams.num_epoch_stageII,        enable_progress_bar=hparams.enable_progress_bar,gradient_clip_val=1.0,
                                       callbacks=[ckpt2, save_comps_stageII, LearningRateMonitor("step")],
                                       strategy=strategy)
    
    # train_stageII_trainer.fit(train_stageII_model,train_dataloaders=stageII_trainloader,val_dataloaders=stageII_validation_loader)

    inference_model = Inference.Inference(hparams,saved_model_paths)
    inference_trainer = pl.Trainer(logger=inference_logger,
                                   accelerator='gpu',devices=hparams.gpu,enable_progress_bar=hparams.enable_progress_bar,strategy=strategy)

    inference_trainer.validate(inference_model,dataloaders=inference_loader)
    

    #  saved the time to complete the training & testing
    execute_time_in_s = time.time() - start_time
    hours_execute = int(execute_time_in_s // 3600)
    minutes_execute = int((execute_time_in_s % 3600) // 60)
    second_execute = int(execute_time_in_s % 60)
        

    print('Training done, time duration: {:02d}:{:02d}:{:02d}'.format(hours_execute,minutes_execute,second_execute))
   