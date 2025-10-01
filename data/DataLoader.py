import os
import torch
from torch.utils.data import Dataset,DataLoader
import pylidc as pl
import pydicom
from monai.transforms import Resize
import numpy as np
import json
import random
import copy
from typing import List
import pandas as pd
from PIL import Image
from torchvision import transforms
from dotted_dict import DottedDict
import matplotlib.pyplot as plt
import random
import json
from torch.utils.data import DataLoader
          
def collate_fn(batch):
    pixels     = torch.stack([d["pixels"] for d in batch], dim=0)           # (B,C,H,W)
    bboxes     = [torch.as_tensor(d["bbox"],      dtype=torch.float32) for d in batch]  # (Ni,4)
    class_ids  = [torch.as_tensor(d["class_id"],  dtype=torch.int64)   for d in batch]  # (Ni,)
# (Ni,)
    return {"pixels": pixels, "bboxes": bboxes, "class_ids": class_ids}


class Train_DataLoading(Dataset):
    def __init__(self,
                 image_path_list:list,
                 transform:None,
                 patches_per_image:3,
                 patch_size:int):
        self.image_path_list = image_path_list
        self.transform = transform
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        
        self.patch_index = [] # [image_idx,x,y]
        
        # generate patch coordinates list 
        for image_idx,image_path in enumerate(self.image_path_list):
            with Image.open(image_path) as img:
                W,H = img.size
                for _ in range(self.patches_per_image):
                    # check if the image H, W is smaller than the patch size
                    if W < self.patch_size or H < self.patch_size: 
                        raise ValueError(f"Expect image with size at least {self.patch_size}x{self.patch_size}. Got input {image_path} with size {W}x{H}")
                
                    x = random.randint(0, W - self.patch_size)
                    y = random.randint(0, H - self.patch_size)
                    self.patch_index.append((image_idx, x, y))
 
        
    def size(self):
        return len(self.patch_index)
    
    def __len__(self):
        return len(self.patch_index)
        
    
    def __getitem__(self, index):
        image_idx, x, y = self.patch_index[index] # get patch
        image_path = self.image_path_list[image_idx]
        image = Image.open(image_path).convert('RGB')
        
        patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
        
        if self.transform:
           patch = self.transform(patch)
           
        return patch
    
    
class Inference_DataLoading(Dataset):
    def __init__(self,
                 image_path_list:list,
                 annotation_dir: str,
                 image_size:int,
                 transform:None):
        self.image_path_list = image_path_list
        self.image_size = image_size
        self.transform = transform
        self.annotation_dir = annotation_dir
        
    def size(self):
        return len(self.image_path_list)
    
    def __len__(self):
        return len(self.image_path_list)
        
    
    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((self.image_size ,self.image_size),Image.BILINEAR)

        image_name,_ = os.path.splitext(os.path.basename(image_path))
        ann_path = os.path.join(self.annotation_dir, image_name + ".txt")
        
        
        labels, boxes_n = self._read_yolo_txt(ann_path)
        boxes_px = boxes_n.clone()
        boxes_px[:, [0,2]] *= self.image_size 
        boxes_px[:, [1,3]] *= self.image_size 
        # final clamp to pixel bounds
        boxes_px[:, [0,2]] = boxes_px[:, [0,2]].clamp(0, self.image_size  - 1)
        boxes_px[:, [1,3]] = boxes_px[:, [1,3]].clamp(0, self.image_size  - 1)
        
        
        if self.transform:
           image_resized = self.transform(image_resized)
        
        return {
            "pixels": image_resized,                 # Tensor [3,H,W]
            "bbox": boxes_px,              # Tensor [Ni,4] xyxy in pixels
            "class_id": labels,            # Tensor [Ni] 0-based
        }
    
    
    def _read_yolo_txt(self, path_txt):
        """Return tensors: labels (N,), boxes_xyxy_norm (N,4) normalized."""
        if not os.path.exists(path_txt):
            return torch.empty(0, dtype=torch.long), torch.empty((0,4), dtype=torch.float32)

        rows = []
        with open(path_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:  # skip malformed
                    continue
                c, x, y, w, h = parts
                rows.append([int(float(c)),
                             float(x), float(y), float(w), float(h)])
        if not rows:
            return torch.empty(0, dtype=torch.long), torch.empty((0,4), dtype=torch.float32)

        arr = torch.tensor(rows, dtype=torch.float32)
        labels = arr[:, 0].to(torch.long)           # (N,)
        xc, yc, ww, hh = arr[:,1], arr[:,2], arr[:,3], arr[:,4]

        # convert normalized xcycwh -> normalized xyxy
        x1n = xc - ww/2.0
        y1n = yc - hh/2.0
        x2n = xc + ww/2.0
        y2n = yc + hh/2.0
        boxes_n = torch.stack([x1n, y1n, x2n, y2n], dim=1)  # (N,4), normalized
        # clamp to [0,1]
        boxes_n = boxes_n.clamp(0.0, 1.0)
        # drop degenerate
        keep = (boxes_n[:,2] > boxes_n[:,0]) & (boxes_n[:,3] > boxes_n[:,1])
        return labels[keep], boxes_n[keep]
    


def get_dataloader(stage,path_params,params):
    
    if stage == 'train_stage_I':
        # mix up CLIC and DIV2K
        with open(path_params.CLIC_image_list,'r') as f:
            CLIC_file_path = f.readlines()
         
        CLIC_file_path = [line.strip() for line in CLIC_file_path]
        
        with open(path_params.DIV2K_image_list,'r') as f:
            DIV2K_file_path = f.readlines()
         
        DIV2K_file_path = [line.strip() for line in DIV2K_file_path]
        
        image_file_path = CLIC_file_path + DIV2K_file_path
        
        data_shuffled = image_file_path.copy()
        random.seed(params.seed)
        random.shuffle(data_shuffled)

        n = len(data_shuffled)
        n_80 = int(0.80 * n)

        train_image_file_path = data_shuffled[:n_80]         # 80%
        validation_image_file_path = data_shuffled[n_80:]  # 20%
        train_image_file_path = train_image_file_path[:200]
        validation_image_file_path = validation_image_file_path[:40]
        
        transform  = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        train_dataset = Train_DataLoading(train_image_file_path,transform,params.patches_per_image,params.train_patch_size)
        
        validation_dataset = Train_DataLoading(validation_image_file_path,transform,params.patches_per_image,params.train_patch_size)
        
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=params.batch_size,
                                  shuffle = True
        )
        
        validation_loader = DataLoader(dataset=validation_dataset,
                                  batch_size=params.batch_size,
                                  shuffle = False
        )
        
        return train_loader,validation_loader
    
    elif stage == 'train_stage_II':

        with open(path_params.VIMEO90K_train_image_list,'r') as f:
            VIMEO90K_train_file_path = f.readlines()
                
        VIMEO90K_train_file_path = [line.strip() for line in VIMEO90K_train_file_path]
            
        with open(path_params.VIMEO90K_validation_image_list,'r') as f:
            VIMEO90K_validation_file_path = f.readlines()
            
        VIMEO90K_validation_file_path = [line.strip() for line in VIMEO90K_validation_file_path]
                    
        VIMEO90K_train_file_path = VIMEO90K_train_file_path[:300]
        VIMEO90K_validation_file_path = VIMEO90K_validation_file_path[:60]
        
        transform  = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        
        train_dataset = Train_DataLoading(VIMEO90K_train_file_path,transform,params.patches_per_image,params.train_patch_size)
        
        validation_dataset = Train_DataLoading(VIMEO90K_validation_file_path,transform,params.patches_per_image,params.train_patch_size)
        
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=params.batch_size,
                                shuffle = True
        )
        
        validation_loader = DataLoader(dataset=validation_dataset,
                                batch_size=params.batch_size,
                                shuffle = False
        )
        
        return train_loader,validation_loader
    
    elif stage == 'inference_object_detection':
        with open(path_params.COCO2014_image_list,'r') as f:
            COCO2014_file_path = f.readlines()
        
        COCO2014_file_path = [line.strip() for line in COCO2014_file_path]
               
        COCO2014_file_path = COCO2014_file_path[:1000]
        
        transform  = transforms.Compose([
        transforms.ToTensor()
        ])
        
        inference_dataset = Inference_DataLoading(COCO2014_file_path,path_params.COCO2014_annotation_dir,params.inference_image_size,transform)


        inference_loader = DataLoader(dataset=inference_dataset,
                                batch_size=params.batch_size,
                                collate_fn=collate_fn,
                                shuffle = False
        )
        
        
        return inference_loader
    
    elif stage == 'inference_reconstruction':
        with open(path_params.KODAK_image_list,'r') as f:
            KODAK_file_path = f.readlines()
                
        KODAK_file_path = [line.strip() for line in KODAK_file_path]
        
        transform  = transforms.Compose([
        transforms.ToTensor()
        ])
        
        train_dataset = Inference_DataLoading(KODAK_file_path,None,params.inference_image_size,transform)

        inference_loader = DataLoader(dataset=train_dataset,
                                batch_size=params.batch_size,
                                shuffle = False
        )
        
        
        return inference_loader
    
    
    



if __name__ == "__main__":
    # test custom dataset
    # image_paths = '/home/nguyensolbadguy/Code_Directory/compression/datasets/CLIC2021/CLIC_image_list.txt'

    # with open(image_paths,'r') as f:
    #      image_file_path = f.readlines()
         
    # image_file_path = [line.strip() for line in image_file_path]
    # print(len(image_file_path))
    # dataset = Train_DataLoading(image_file_path,None,patches_per_image=3, patch_size=128)
    
    # print(len(dataset)) 


    # test dataset loader
    # params = DottedDict({'train_patch_size':256,'batch_size':16,'patches_per_image':3,'seed':42})
    # path_params = DottedDict({'CLIC_image_list':'/home/nguyensolbadguy/Code_Directory/compression/datasets/CLIC2021/CLIC_image_list.txt',
    #                           'DIV2K_image_list':'/home/nguyensolbadguy/Code_Directory/compression/datasets/DIV2K/DIV2K_image_list.txt'})
    
    # train_loader,validation_loader = get_dataloader('train_stage_I',path_params,params)
    # image = next(iter(train_loader))

    # image_test = image[5]
    # image_test = image_test.permute(1, 2, 0).cpu().numpy()
    # fig, ax = plt.subplots(figsize=(5, 5))
    # img = ax.imshow(image_test, cmap=plt.cm.gray)
    # ax.axis('off')
    # title = ax.set_title("Slice 0")
    # plt.show()
    params = DottedDict({'inference_image_size':512,'batch_size':16,'seed':42})
    
    path_params = DottedDict({'COCO2014_image_list': '/home/nguyensolbadguy/Code_Directory/compression/datasets/COCO2014/COCO2014_image_list.txt',
                               'KODAK_image_list':'/home/nguyensolbadguy/Code_Directory/compression/datasets/Kodak/KODAK_image_list.txt',
                               'COCO2014_annotation_list': '/home/nguyensolbadguy/Code_Directory/compression/datasets/COCO2014/validation_annotations.json'})
    
    inference_loader = get_dataloader('inference_object_detection',path_params,params)
    batch = next(iter(inference_loader))
    x = batch['pixels']
    bboxes = batch['bboxes']
    crowds = batch['crowds']
    difficults = batch['difficults']
    class_ids = batch['class_ids']
    
    print(bboxes)
    # image_test = batch[5]['pixels']
    # image_test = image_test.permute(1, 2, 0).cpu().numpy()
    # fig, ax = plt.subplots(figsize=(5, 5))
    # img = ax.imshow(image_test, cmap=plt.cm.gray)
    # ax.axis('off')
    # title = ax.set_title("Slice 0")
    # plt.show()
    
    
    # inference_loader = get_dataloader('inference_reconstruction',path_params,params)
    # image = next(iter(inference_loader))
    # image_test = image[5]
    # image_test = image_test.permute(1, 2, 0).cpu().numpy()
    # fig, ax = plt.subplots(figsize=(5, 5))
    # img = ax.imshow(image_test, cmap=plt.cm.gray)
    # ax.axis('off')
    # title = ax.set_title("Slice 0")
    # plt.show()
    
    
    
    
    
    
