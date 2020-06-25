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


class FeatureExtractor(nn.Module):
    def __init__(self,input_channels):
        super(FeatureExtractor,self).__init__()
    
        self.input_shape = input_channels

        self.conv1 = nn.Conv3d(input_channels,64,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2),stride = (1,2,2))

        self.conv2 = nn.Conv3d(64,128,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2),stride = (1,2,2))


        self.conv3a = nn.Conv3d(128,256,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn3a = nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256,256,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn3b = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(1,2,2),stride = (1,2,2))

        self.conv4a = nn.Conv3d(256,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn4a = nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn4b = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(1,2,2),stride = (1,2,2))

        self.conv5a = nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn5a = nn.BatchNorm3d(512)
        self.conv5b = nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn5b = nn.BatchNorm3d(512)



    def forward(self,x):
        #print(type(x))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.relu(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.relu(x)
        x = self.pool3(x)


        x = self.conv4a(x)
        x = self.bn4a(x)
        x = self.relu(x)
        x = self.conv4b(x)
        x = self.bn4b(x)
        x = self.relu(x)
        x = self.pool4(x)


        x = self.conv5a(x)
        x = self.bn5a(x)
        x = self.relu(x)
        x = self.conv5b(x)
        x = self.bn5b(x)
        exFeat = self.relu(x)
        return exFeat



class C3D(nn.Module):

    def __init__(self,input_channels=3):
        super(C3D, self).__init__()
        self.input_channels=input_channels

        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
       

        return h
    
    
class CoarseAP(nn.Module):
    def __init__(self,input_channels):
        super(CoarseAP,self).__init__()

        self.input_channels = input_channels
        self.FeatureExtractor = FeatureExtractor(self.input_channels)

        self.conv6a = nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn6a = nn.BatchNorm3d(512)
        self.relu = nn.ReLU()

        self.conv6b = nn.Conv3d(512,512,kernel_size = (1,1,1))
        self.bn6b = nn.BatchNorm3d(512)

        self.conv6c = nn.Conv3d(512,1,kernel_size=(1,1,1))
        self.bn6c = nn.BatchNorm3d(1)


    def forward(self,x):

        x = self.FeatureExtractor(x)

        x = self.conv6a(x)
        x = self.bn6a(x)
        x = self.relu(x)

        x = self.conv6b(x)
        x = self.bn6b(x)
        x = self.relu(x)

        x = self.conv6c(x)
        x = self.bn6c(x)
        

        return x




class FineAP(nn.Module):
    def __init__(self,input_channels):
        super(FineAP,self).__init__()

        self.input_channels = input_channels
        self.exFeat = FeatureExtractor(self.input_channels)
        self.cap = CoarseAP(self.input_channels)


        self.deconv1 = nn.ConvTranspose3d(513,512,kernel_size = (1,4,4),stride=(1,4,4))
        self.bn1 = nn.BatchNorm3d(512)
        self.relu = nn.ReLU()

        self.conv7a = nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn7a = nn.BatchNorm3d(512)


        self.conv7b = nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn7b = nn.BatchNorm3d(512)


        self.deconv2 = nn.ConvTranspose3d(512,512,kernel_size = (1,4,4),stride=(1,4,4))
        self.bn2 = nn.BatchNorm3d(512)
    

        self.conv8a = nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn8a = nn.BatchNorm3d(512)


        self.conv8b = nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn8b = nn.BatchNorm3d(512)

        self.conv11a = nn.Conv3d(512,512,kernel_size=(1,1,1))
        self.bn11a = nn.BatchNorm3d(512)
        self.conv11b = nn.Conv3d(512,1,kernel_size=(1,1,1))
        self.bn11b = nn.BatchNorm3d(1)
    

    def forward(self,x):

        x1 = self.exFeat(x)
        x2 = self.cap(x)
        x = torch.cat([x1,x2],axis=1)
      
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv7a(x)
        x = self.bn7a(x)
        x = self.relu(x)

        x = self.conv7b(x)
        x = self.bn7b(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv8b(x)
        x = self.bn8a(x)
        x = self.relu(x)

        x = self.conv8b(x)
        x = self.bn8b(x)
        x = self.relu(x)

        x = self.conv11a(x)
        x = self.bn11a(x)
        x = self.relu(x)

        x = self.conv11b(x)
        fgp = self.bn11b(x)
        
        

        return fgp
    
    
if __name__=='__main__':    
    model1 = CoarseAP(3)
    model2=FineAP(3)

    i = torch.randn(3,3,16,64,64)
    y = model2(i)
    print(y.shape)