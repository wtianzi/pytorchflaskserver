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

def ce_loss(y_pred,hm):
    print(y_pred.shape)
    hm = torch.reshape(torch.tensor(hm),(hm.shape[0]*hm.shape[1],-1))
    target = torch.argmax(hm,dim=1)
    y_pred = y_pred.squeeze()
    print('shape:',y_pred.shape)
    pred = torch.reshape(torch.tensor(y_pred),(y_pred.shape[0]*y_pred.shape[1],-1))
    return nn.CrossEntropyLoss(reduction='mean')(pred,target)



def kl_loss(y_pred,hm):
    hm = torch.reshape(torch.tensor(hm),(hm.shape[0]*hm.shape[1],-1))
    y_pred = y_pred.squeeze()
    pred = torch.reshape(torch.tensor(y_pred),(y_pred.shape[0]*y_pred.shape[1],-1))
    pred = F.log_softmax(pred,dim=1)
    return nn.KLDivLoss(reduction = 'batchmean')(pred,hm)



batch=16
trainX,_ = get_frames()
trainY,heatmap = get_grids()

trainX = np.rollaxis(trainX,-1,1)
train_X, test_X, train_Y, test_Y = train_test_split(trainX, trainY, test_size=0.20, random_state=42)
train_X, test_X, heatmap_train, heatmap_test = train_test_split(trainX, heatmap, test_size=0.20, random_state=42)
print(train_X.shape)



def train_one(model, device, train_loader, optimizer, epochs):
    model.train()
    print('Train Epoch: {} \tLR: {:.6f}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    t = tqdm(train_loader)
    for batch_idx, (train_X, train_Y,_) in enumerate(train_loader):
        train_X, train_Y = train_X.to(device), train_Y.to(device)
        
        optimizer.zero_grad()
        
        
        output = model(torch.tensor(train_X).float())
        
        loss = ce_loss(output, train_Y)
        
        t.set_description(f'train_loss (l={loss:.3f})(m={mask_loss:.2f}) (r={regr_loss:.4f}')
        
        
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step(loss)
        
    

for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    
    train_one(model,device,train_loader,optimizer,epochs)
    evaluate(epoch, history)
    if epoch >= 10:
        torch.save(model.state_dict(), f'./model_epoch_{epoch}.pth')
        
        
        
        
def train(model, device, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (train_X, train_Y,_) in enumerate(train_loader):
            train_X, train_Y = train_X.to(device), train_Y.to(device)
            optimizer.zero_grad()
            output = model(torch.tensor(train_X).float())
            loss = ce_loss(output, train_Y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)
            if batch_idx % 16 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(train_X), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            
            
    torch.save(model.state_dict(), "coarse_mode.pt")
    
    
def train_one(model, device, train_loader, optimizer,lr_scheduler, epochs=2):
    model.train()
    print('Train Epoch: {} \tLR: {:.6f}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    #t = tqdm(train_loader,disable = False)
    for batch_idx, (train_X, train_Y,_) in enumerate(train_loader):
        train_X, train_Y = train_X.to(device), train_Y.to(device)
        
        optimizer.zero_grad()
        
        
        output = model(torch.tensor(train_X).float())
        
        loss = ce_loss(output, train_Y)
        
        #t.set_description(f'train_loss (l={loss:.3f})(m={mask_loss:.2f}) (r={regr_loss:.4f}')
        
        
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step(loss)
        
    

