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
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
#from tqdm.auto import tqdm as tq
from torch.optim import lr_scheduler
from torchvision import transforms, utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class VidDataset(Dataset):
    def __init__(self,train_X,train_Y):
        self.trainX = train_X 
        self.trainY = train_Y
        
        
    def __len__(self):
        return len(self.trainX)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        trainX = self.trainX[idx]
        trainY = self.trainY[idx]
        
        
        return [trainX,trainY]
    
    
