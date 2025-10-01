import pandas as pd
import csv 
import os
from datetime import datetime


class BestLogger:
    '''
    Save the best result of training stage
    '''
    def  __init__(self,csv_file,mode):
        self.csv_file = csv_file 
        self.metrics = []
        
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        
        if mode == 'validation':
            if not os.path.exists(csv_file):
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'RD_loss', 'rate_loss', 'distortion_loss', 'bitrate'])
                
        elif mode == 'inference_object_detection':
            if not os.path.exists(csv_file):
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'RD_loss','rate_loss','distortion_loss','bitrate','bitrate_base','bitrate_side','bitrate_enhancement','mAP'
                    ])
                
        elif mode == 'inference_reconstruction':
            if not os.path.exists(csv_file):
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'RD_loss','rate_loss','distortion_loss','bitrate','bitrate_base','bitrate_side','bitrate_enhancement','PSNR','MS-SSIM'
                    ])            

            
    def log_validation(self, epoch,loss,rate_loss,distortion_loss,bitrate):
        """Log metrics for validation"""
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss,rate_loss,distortion_loss,bitrate])

    def log_reconstruction_inference(self, epoch,
                            loss,
                            rate_loss,
                            distortion_loss,
                            bitrate,
                            bitrate_base,
                            bitrate_side,
                            bitrate_enhancement,
                            PSNR,
                            MS_SSIM
                            ):
        """Log metrics for one epoch"""
        # Append to CSV file
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch,loss,rate_loss,distortion_loss,bitrate,bitrate_base,bitrate_side,bitrate_enhancement,PSNR,MS_SSIM])
            
    def log_detection_inference(self, epoch,
                            loss,
                            rate_loss,
                            distortion_loss,
                            bitrate,
                            bitrate_base,
                            bitrate_side,
                            bitrate_enhancement,
                            mAP
                            ):
        """Log metrics for one epoch"""
        # Append to CSV file
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch,loss,rate_loss,distortion_loss,bitrate,bitrate_base,bitrate_side,bitrate_enhancement,mAP])


class PerformanceLogger:
    def  __init__(self,csv_file,mode):
        self.csv_file = csv_file
        self.metrics = []
        
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        if mode == 'train':
            if not os.path.exists(csv_file):
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'RD_loss', 'rate_loss', 'distortion_loss', 'bitrate'])
        elif mode == "inference_object_detection":
            if not os.path.exists(csv_file):
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'RD_loss','rate_loss','distortion_loss','bitrate','bitrate_base','bitrate_side','bitrate_enhancement','mAP'
                    ])
        elif mode == 'inference_reconstruction':
            if not os.path.exists(csv_file):
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'RD_loss','rate_loss','distortion_loss','bitrate','bitrate_base','bitrate_side','bitrate_enhancement','PSNR','MS-SSIM'
                    ])


    def log_train(self, epoch,loss,rate_loss,distortion_loss,bitrate):
        """Log metrics for train"""
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss,rate_loss,distortion_loss,bitrate,bitrate])
            
    def log_validation(self, epoch,loss,rate_loss,distortion_loss,bitrate):
        """Log metrics for validation"""
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss,rate_loss,distortion_loss,bitrate])

    def log_reconstruction_inference(self, epoch,
                            loss,
                            rate_loss,
                            distortion_loss,
                            bitrate,
                            bitrate_base,
                            bitrate_side,
                            bitrate_enhancement,
                            PSNR,
                            MS_SSIM
                            ):
        """Log metrics for one epoch"""
        # Append to CSV file
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch,loss,rate_loss,distortion_loss,bitrate,bitrate_base,bitrate_side,bitrate_enhancement,PSNR,MS_SSIM])
            
    def log_detection_inference(self, epoch,
                            loss,
                            rate_loss,
                            distortion_loss,
                            bitrate,
                            bitrate_base,
                            bitrate_side,
                            bitrate_enhancement,
                            mAP
                            ):
        """Log metrics for one epoch"""
        # Append to CSV file
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch,loss,rate_loss,distortion_loss,bitrate,bitrate_base,bitrate_side,bitrate_enhancement,mAP])

