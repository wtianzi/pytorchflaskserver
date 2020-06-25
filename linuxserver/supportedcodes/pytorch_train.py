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
from tqdm.notebook import tqdm
from tqdm.auto import tqdm as tq
import time
import pickle

def ce_loss(y_pred,hm):
    hm = hm.squeeze()
    hm = torch.reshape((hm),(hm.shape[0]*hm.shape[1],-1))
    target = torch.argmax(hm,dim=1)
    y_pred = y_pred.squeeze()
    pred = torch.reshape((y_pred),(y_pred.shape[0]*y_pred.shape[1],-1))
    return nn.CrossEntropyLoss(reduction='mean')(pred,target)



def kl_loss(y_pred,hm):
    hm = torch.reshape((hm),(hm.shape[0]*hm.shape[1],-1))
    y_pred = y_pred.squeeze()
    pred = torch.reshape((y_pred),(y_pred.shape[0]*y_pred.shape[1],-1))
    pred = F.log_softmax(pred,dim=1)
    return nn.KLDivLoss(reduction = 'batchmean')(pred,hm)



def cap_acc(y_pred,y_true):
    y_true = torch.reshape((y_true),(y_true.shape[0]*y_true.shape[1],-1))
    y_true = torch.argmax(y_true,dim=1)
    y_pred = y_pred.squeeze()
    pred = torch.reshape((y_pred),(y_pred.shape[0]*y_pred.shape[1],-1))
    y_pred = torch.argmax(pred,dim=1)
    
    correct = torch.eq(y_pred,y_true).cpu()
    correct = correct.numpy()
    return np.mean(correct)
    
    

def fgp_acc(y_pred,y_true):
    y_true = torch.reshape((y_true),(y_true.shape[0]*y_true.shape[1],y_true.shape[2],y_true.shape[3]))
    y_pred = y_pred.squeeze()
    
    y_pred = torch.reshape((y_pred),(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2],y_pred.shape[3]))
    x0_true = torch.argmax(torch.max(y_true,dim=-2)[0],dim=-1)
    y0_true = torch.argmax(torch.max(y_true,dim=-1)[0],dim=-1)
    x0_pred = torch.argmax(torch.max(y_pred,dim=-2)[0],dim=-1)
    y0_pred = torch.argmax(torch.max(y_pred,dim=-1)[0],dim=-1)
    #print(x0_true.dtype,x0_pred.dtype)
    ss = torch.square(x0_true-x0_pred,)+torch.square(y0_true-y0_pred)
    
    dist = torch.mean(torch.sqrt(ss.float()))
    return dist.item()
    

    
def generate_plots(df):
    history = pd.read_csv(df)
    plt.plot(history['val_loss'],label='val loss')
    plt.plot(history['train_loss'],label='train loss')
    plt.legend()
    plt.savefig('loss_plot'+(df)+'.png')
    plt.close()
    plt.plot(history['val_acc'],label='val deviation')
    plt.plot(history['train_acc'],label='train deviation')
    plt.legend()
    plt.show()
    plt.savefig('acc_plot'+df+'.png')
    plt.close()
    print('done!')
    
    
    
    
batch=24
trainX,_ = get_frames(n_frames=16)
trainY,heatmap = get_grids(n_frames=16)
print('trainX=',trainX.shape)

trainX = np.rollaxis(trainX,-1,1)

train_Xc, test_Xc, train_Yc, test_Yc = train_test_split(trainX, trainY, test_size=0.20, random_state=42)
train_Xf, test_Xf, heatmap_trainf, heatmap_testf = train_test_split(trainX, heatmap, test_size=0.20, random_state=42)

train_datasetc = VidDataset(train_Xc,train_Yc)
val_datasetc = VidDataset(test_Xc,test_Yc)


train_datasetf = VidDataset(train_Xf,heatmap_trainf)
val_datasetf = VidDataset(test_Xf,heatmap_testf)



train_loaderc = DataLoader(dataset = train_datasetc,batch_size=128,shuffle=True,num_workers=4)

#print(train_datasetc)


val_loaderc = DataLoader(dataset=val_datasetc,batch_size=128,shuffle=True,num_workers=4)

train_loaderf = DataLoader(dataset = train_datasetf,batch_size=24,shuffle=True,num_workers=4)
val_loaderf = DataLoader(dataset=val_datasetf,batch_size=24,shuffle=True,num_workers=4)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"

device = torch.device("cuda")

model_fine = FineAP(3).to(device)

model_coarse = CoarseAP(3).to(device)


def train_coarse(model,epochs=100,ce_loss = ce_loss,tl=False):
    
    
    if tl:
        print('Transfer Learning')
        c3dict = torch.load('c3d.pickle')
        modeldict = model.state_dict()
        pretrained_dict={}
        for i in c3dict.keys():
            for j in modeldict.keys():
                if i in j and 'deconv' not in j:
                    pretrained_dict[j] = c3dict[i]
      
      
        modeldict.update(pretrained_dict)
        model.load_state_dict(modeldict) 

        for par in model.named_parameters():
            if par[0] in pretrained_dict.keys():
                par[1].requires_grad=False
    
    
    history = pd.DataFrame([])
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7,8,9])
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9,weight_decay=1e-2)
    
    total_train_loss=[]
    total_val_loss=[]
    total_train_acc=[]
    total_val_acc=[]
    for e in range(epochs):
        
        loss_epoch = []
        acc_epoch = []
        model.train()
    
        for batch_idx, (train_X, train_Y) in enumerate(train_loaderc):
            train_X, train_Y = train_X.to(device,dtype = torch.float32), train_Y.to(device,dtype = torch.float32)
            optimizer.zero_grad()
            output = model(train_X)
            loss = ce_loss(output, train_Y)
            accuracy = cap_acc(output,train_Y)
            
            l2 = 0
            for p in model.named_parameters():
                if 'conv' and 'weight' in p[0]:
                    l2 = l2+ torch.pow(p[1].norm(2),2)
                    
            loss += (1e-6) * l2
            
            loss.backward()
            optimizer.step()
            
            loss_epoch.append(loss.item())
            acc_epoch.append(accuracy.item())
   
     
        total_train_loss.append(np.mean(loss_epoch))
        total_train_acc.append(np.mean(acc_epoch))
        print(f'Epcoh {e}: ,batch loss:{loss.item()},epoch loss:{np.mean(loss_epoch)},acc:{np.mean(acc_epoch)}',end=' ')
        
        model.eval()
        with torch.no_grad():
            val_loss_epoch=[]
            val_acc_epoch=[]
            for val_X, val_Y in val_loaderc:
                val_X,val_Y = val_X.to(device,dtype = torch.float32),val_Y.to(device,dtype = torch.float32)
                op = model(val_X)
                val_loss = ce_loss(op,val_Y)
                val_acc = cap_acc(op,val_Y)
                val_loss_epoch.append(val_loss.item())
                val_acc_epoch.append(val_acc.item())
             
            total_val_acc.append(np.mean(val_acc_epoch))    
            total_val_loss.append(np.mean(val_loss_epoch))
            print(f',epoch val_loss:{np.mean(val_loss_epoch)},val_acc={np.mean(val_acc_epoch)}')
            
    torch.save(model.state_dict(),f'modelc_epoch_{epochs}.pth')
    history['train_loss'] = total_train_loss
    history['val_loss'] = total_val_loss
    history['train_acc'] = total_train_acc
    history['val_acc'] = total_val_acc
    history.to_csv(f'history_coarse_{e+1}.csv')
    generate_plots(f'history_coarse_{e+1}.csv')         
            
            
            
            
            
            
            
            
def train_fine(model,epochs=100,kl_loss = kl_loss,tl=False):
    history = pd.DataFrame([])
    
    if tl:
        print('Transfer Learning')
        c3dict = torch.load('modelc_epoch_100.pth')
        modeldict = model.state_dict()
        pretrained_dict={}
        for i in c3dict.keys():
            for j in modeldict.keys():
                if i in j and 'deconv' not in j:
                    pretrained_dict[j] = c3dict[i]
      
      
        modeldict.update(pretrained_dict)
        model.load_state_dict(modeldict) 

        for par in model.named_parameters():
            if par[0] in pretrained_dict.keys():
                par[1].requires_grad=False
    
    
    
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7,8,9])
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9,weight_decay=1e-2)
    
    total_train_loss=[]
    total_val_loss=[]
    total_train_acc=[]
    total_val_acc=[]
    for e in range(epochs):
        loss_epoch = []
        acc_epoch = []
        model.train()
    
        for batch_idx, (train_X, train_Y) in enumerate(train_loaderf):
            train_X, train_Y = train_X.to(device,dtype = torch.float32), train_Y.to(device,dtype = torch.float32)
            #print((train_X).dtype)
            optimizer.zero_grad()
            output = model(train_X)
            loss = kl_loss(output, train_Y)
            l2 = 0
            for p in model.named_parameters():
                if 'conv' and 'weight' in p[0]:
                    l2 = l2+ torch.pow(p[1].norm(2),2)
            loss = loss+l2*(1e-6)
            loss.backward()
            optimizer.step()
            
            loss_epoch.append(loss.item())
            accuracy = fgp_acc(output,train_Y)
            acc_epoch.append(accuracy)
        
        
   
     
        
        total_train_loss.append(np.mean(loss_epoch))
        total_train_acc.append(np.mean(acc_epoch))
        print(f'Epcoh {e}: ,batch loss:{loss.item()},epoch loss:{np.mean(loss_epoch)},acc:{np.mean(acc_epoch)}',end=' ')   
    
   
        model.eval()
        with torch.no_grad():
            val_loss_epoch=[]
            val_acc_epoch=[]
            for val_X, val_Y in val_loaderf:
                val_X,val_Y = val_X.to(device,dtype = torch.float32),val_Y.to(device,dtype = torch.float32)
                op = model(val_X)
                val_loss = kl_loss(op,val_Y)
                val_acc = fgp_acc(op,val_Y)
                val_loss_epoch.append(val_loss.item())
                val_acc_epoch.append(val_acc)
             
            total_val_acc.append(np.mean(val_acc_epoch))    
            total_val_loss.append(np.mean(val_loss_epoch))
            print(f',epoch val_loss:{np.mean(val_loss_epoch)},val_acc={np.mean(val_acc_epoch)}')
            
    torch.save(model.state_dict(),f'modelf_epoch_{e}.pth')
    history['train_loss'] = total_train_loss
    history['val_loss'] = total_val_loss
    history['train_acc'] = total_train_acc
    history['val_acc'] = total_val_acc
    history.to_csv(f'history_fine_{e+1}.csv')
    generate_plots(f'history_fine_{e+1}.csv')   
    


    
if __name__=='__main__':
    train_fine(model_fine)
    #train_coarse(model_coarse,tl=True)
    
    print()
    