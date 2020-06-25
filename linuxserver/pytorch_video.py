import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
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
from tqdm.notebook import tqdm
from tqdm.auto import tqdm as tq
import time
import numpy as np
import os
import cv2
from pytorch_model import *
from Inputs import *

##################################################################3


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
device = torch.device("cuda")
path = 'modelf_epoch_99.pth'
fine_mod = FineAP(3).to(device)


def pred_coords(predict,test_frames,test_vid=None):
    
    
    predout = predict.squeeze()
    j=0
    predframe = []
    coords_list = get_coords(test_vid)
    for arrframes in predout:
        for frame in arrframes:

            nx = np.argmax((np.max(frame,axis=1)))
            ny = np.argmax((np.max(frame,axis=0)))
    
            x = 320*ny + 160
            y = 180*nx + 90
            try:
                a,b = coords_list[j]
                #print(len(test_frames[i]),len(get_coords(path)) )
                frametemp = cv2.rectangle(test_frames[j],(x-160,y-90),(x+160,y+90),(0, 255, 0), 2)
                predframe.append(cv2.rectangle(frametemp,(a-30,b-30),(a+30,b+30),(255, 0, 0), 2))
                j+=1
            except IndexError:
                pass
            
    return predframe



def get_coords(video,path='test'):
    inparr = []
    gridarr=[]
    inpgrid = []
    path_file = path+'/test_labels/'+video[:-4]+'.txt'
    with open(path_file, "r") as f:
       
        
        content = f.readlines()
        for i in range(len(content)):
            
      #grid = np.zeros((4,4))
            x,y = content[i].strip().split(',')[0:2]
            x,y = int(x),int(y)
            inparr.append([x,y])
    return inparr




def coarse_vid(model_path):
    path = 'test'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
    device = torch.device("cuda")
    coarse_mod = CoarseAP(3).to(device)
    coarse_mod.load_state_dict(torch.load(model_path))
    for video in os.listdir(path+'/'+'test_videos'):
        #print(video)
        opframes=[]
        frames = []
        preds=[]
        test_array,test_frames = get_frames(vidpath=path+'/test_videos/'+video,train=False)
        test_array = np.rollaxis(test_array,-1,1)
        test_array = torch.tensor(test_array,dtype=torch.float32,requires_grad=False)
        coarse_mod.eval()
        with torch.no_grad():
            pred = coarse_mod(test_array).squeeze()
            predframe = pred_coords(pred,path,test_frames)
            out = cv2.VideoWriter('test_course_val' + video,cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280,720))
            for i in range(len(predframe)):
                out.write(cv2.cvtColor(predframe[i], cv2.COLOR_BGR2RGB))
            out.release()
            
            
            
def fine_vid(model_path):
    path = 'test'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
    device = torch.device("cuda")
    fine_mod = FineAP(3).to(device)
    fine_mod = nn.DataParallel(fine_mod, device_ids=[0,1,2,3,4,5,6,7,8,9])
    fine_mod.load_state_dict(torch.load(model_path))
    dist=0
    distlis=[]
    for video in os.listdir(path+'/'+'test_videos'):
        #print(video)
        coords = get_coords(video)
        opframes=[]
        frames = []
        preds=[]
        test_array,test_frames = get_frames(vidpath=path+'/test_videos/'+video,train=False)
        test_array = np.rollaxis(test_array,-1,1)
        test_array = torch.tensor(test_array,dtype=torch.float32,requires_grad=False)
        #print(test_array.shape)
        fine_mod.eval()
        with torch.no_grad():
            for arr in test_array:
                p = fine_mod(arr[None]).squeeze()
                preds.append(p.cpu().numpy())
                
        j=0
        predframe = []
        for arrframes in preds:
            for frame in arrframes:

                nx = np.argmax((np.max(frame,axis=1)))
                ny = np.argmax((np.max(frame,axis=0)))
                x = int(round(ny*20))
                y = int(round(nx*11.25)) 
                a,b = coords[j]
                
                dist=np.sqrt(np.square(x-a)+np.square(y-b))
                distlis.append(dist)
                
   
                frametemp = cv2.rectangle(test_frames[j],(x-25,y-25),(x+25,y+25),(0, 255, 0), 2)
                predframe.append(cv2.rectangle(frametemp,(a-25,b-25),(a+25,b+25),(255, 0, 0), 2))
   
                j+=1
    
        out = cv2.VideoWriter(f'{video}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280,720))
        for i in range(len(predframe)):
            out.write(cv2.cvtColor(predframe[i], cv2.COLOR_BGR2RGB))
    
        out.release()
        print(np.mean(distlis))
            
           
    


################################################################################

fine_vid('modelf_epoch_99.pth')

