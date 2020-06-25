import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from Inputs import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils,models
import torch.nn.functional as F
from pytorch_model import *
from torch.optim import lr_scheduler
from torchvision import transforms, utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_dataset import *
import torch.onnx
import torch.nn.init as init
from collections import OrderedDict


batch_size = 32
model_fine = FineAP(3)


fine_mod = FineAP(3)#.to(device)
state_dict = torch.load('modelf_epoch_99.pth')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] 
    new_state_dict[name] = v

fine_mod.load_state_dict(new_state_dict)

fine_mod.eval()
x = torch.randn(batch_size, 3,16, 64, 64, requires_grad=True)

torch.onnx.export(fine_mod,               
                  x,                         
                  "../model_fine.onnx",   
                  export_params=True,       
                  opset_version=10,          
                  do_constant_folding=True,  
                  input_names = ['input'],   
                  output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},    
                                'output' : {0 : 'batch_size'}})