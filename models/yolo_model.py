'''
Adapt original implementation of YOLOv3 from https://github.com/eriklindernoren/PyTorch-YOLOv3.git


'''

from torchinfo import summary
from torch import nn
import torch
import sys
import cv2
sys.path.append('/home/nguyensolbadguy/Code_Directory/compression/models/yolov3') 

from .yolov3.pytorchyolo import models,detect
from .yolov3.pytorchyolo.models import YOLOLayer,create_modules
from .yolov3.pytorchyolo.utils.utils import non_max_suppression,rescale_boxes



class  YOLOv3(nn.Module):
    def __init__(self, start_idx,image_size):
        super().__init__()
        
        model = models.load_model('/home/nguyensolbadguy/Code_Directory/compression/models/yolov3/config/yolov3.cfg','/home/nguyensolbadguy/Code_Directory/compression/models/yolov3/weights/yolov3.weights')
        model.eval()
        
        
        self.module_defs = model.module_defs
        self.module_list = model.module_list    
        self.start_idx = start_idx
        self.yolo_layers = [layer[0]
                            for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.image_size = image_size

    def forward(self, x,mode):
        layer_outputs, yolo_outputs = {}, []
        
        # use in training the F_tilde_1 to be as much as F_1
        if mode == 'train':
            for i in range(0, self.start_idx+1):
                module_def = self.module_defs[i]
                module = self.module_list[i]

                if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                    x = module(x)
                    if i == 12:
                        F_target = x
                        break

                elif module_def["type"] == "route":
                    
                    layers = [int(l) for l in module_def["layers"].split(",")]
                    layers = [l if l >= 0 else i + l for l in layers]
                    
                    try:
                        route_tensors = [layer_outputs[l] for l in layers]
                    except KeyError as e:
                        raise RuntimeError(f"Route error at layer {i}: missing dependency {e}")

                    x = torch.cat(route_tensors, dim=1)
                    
                    if "groups" in module_def:
                        group_size = x.shape[1] // int(module_def["groups"])
                        group_id = int(module_def["group_id"])
                        x = x[:, group_size * group_id : group_size * (group_id + 1)]

                elif module_def["type"] == "shortcut":

                    from_idx = int(module_def["from"])
                    shortcut_idx = i + from_idx if from_idx < 0 else from_idx

                    try:
                        x = layer_outputs[i - 1] + layer_outputs[shortcut_idx]
                    except KeyError as e:
                        raise RuntimeError(f"Shortcut error at layer {i}: missing dependency {e}")

                elif module_def["type"] == "yolo":
                    x = module[0](x, self.image_size)
                    yolo_outputs.append(x)

                layer_outputs[i] = x

            return F_target
        
        # for proposed model inference using y_1
        elif mode == "inference":
            for i in range(self.start_idx, len(self.module_list)):
                module_def = self.module_defs[i]
                module = self.module_list[i]

                if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                    if i == 12:
                        x = module[1](x)  # BatchNorm
                        x = module[2](x)  # Activation
                    else:
                        x = module(x)

                elif module_def["type"] == "route":
                    
                    layers = [int(l) for l in module_def["layers"].split(",")]
                    layers = [l if l >= 0 else i + l for l in layers]
                    
                    try:
                        route_tensors = [layer_outputs[l] for l in layers]
                    except KeyError as e:
                        raise RuntimeError(f"Route error at layer {i}: missing dependency {e}")

                    x = torch.cat(route_tensors, dim=1)
                    
                    if "groups" in module_def:
                        group_size = x.shape[1] // int(module_def["groups"])
                        group_id = int(module_def["group_id"])
                        x = x[:, group_size * group_id : group_size * (group_id + 1)]

                elif module_def["type"] == "shortcut":

                    from_idx = int(module_def["from"])
                    shortcut_idx = i + from_idx if from_idx < 0 else from_idx

                    try:
                        x = layer_outputs[i - 1] + layer_outputs[shortcut_idx]
                    except KeyError as e:
                        raise RuntimeError(f"Shortcut error at layer {i}: missing dependency {e}")

                elif module_def["type"] == "yolo":
                    x = module[0](x, self.image_size)
                    yolo_outputs.append(x)

                layer_outputs[i] = x

            return torch.cat(yolo_outputs, 1)
        
        # for baseline inference
        elif mode == "inference_baseline":
            for i in range(0, len(self.module_list)):
                module_def = self.module_defs[i]
                module = self.module_list[i]

                if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                    x = module(x)

                elif module_def["type"] == "route":
                    
                    layers = [int(l) for l in module_def["layers"].split(",")]
                    layers = [l if l >= 0 else i + l for l in layers]
                    
                    try:
                        route_tensors = [layer_outputs[l] for l in layers]
                    except KeyError as e:
                        raise RuntimeError(f"Route error at layer {i}: missing dependency {e}")

                    x = torch.cat(route_tensors, dim=1)
                    
                    if "groups" in module_def:
                        group_size = x.shape[1] // int(module_def["groups"])
                        group_id = int(module_def["group_id"])
                        x = x[:, group_size * group_id : group_size * (group_id + 1)]

                elif module_def["type"] == "shortcut":

                    from_idx = int(module_def["from"])
                    shortcut_idx = i + from_idx if from_idx < 0 else from_idx

                    try:
                        x = layer_outputs[i - 1] + layer_outputs[shortcut_idx]
                    except KeyError as e:
                        raise RuntimeError(f"Shortcut error at layer {i}: missing dependency {e}")

                elif module_def["type"] == "yolo":
                    x = module[0](x, self.image_size)
                    yolo_outputs.append(x)

                layer_outputs[i] = x

            return torch.cat(yolo_outputs, 1)
            

    
    

def detect_image(model,input_tensor,mode, img_size=512, conf_thres=0.5, nms_thres=0.5):
    '''
    Inferences images with model.

    '''
    if  mode == 'train':
        F_target = model(input_tensor,mode)
        return F_target
    else:
        
        # Get detections
        with torch.no_grad():
            detections = model(input_tensor,mode)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Store image and detections

        return detections
        
        out = model(input_tensor,mode)
        detections = non_max_suppression(out, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, [img_size,img_size])
        return detections



if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_hat = torch.randn(1, 3, 256, 256).to(device)
    F_1_tilde = torch.randn(1, 256, 64, 64).to(device)
    baseline_input = torch.randn(1, 3, 512, 512).to(device)
    
    partial_model = YOLOv3(start_idx=12,image_size=256).to(device)
    
    #F_1_train = detect_image(partial_model,X_hat,'train')
    detection_proposal = detect_image(partial_model,F_1_tilde,'inference')
    #detection_baseline = detect_image(partial_model,baseline_input,'inference_baseline')
    
    print(detection_proposal)
    #print(F_1_train.shape)
    #print(detection_proposal.shape)
    #print(detection_baseline.shape)