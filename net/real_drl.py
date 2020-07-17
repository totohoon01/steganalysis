# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:17:53 2020

@author: IVCLAB
"""
import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F

hpfs = np.zeros([1,1,5,5],dtype=np.int)
hpf = np.array([[-1,2,-2,2,-1],
                [2,-6,8,-6,2],
                [-2,8,-12,8,-2],
                [2,-6,8,-6,2],
                [-1,2,-2,2,-1]])
hpfs[0,0] = hpf

class Model(nn.Module):
    def __init__(self, classes):
        super(Model, self).__init__()
        self.hpfs = torch.tensor(hpfs, dtype=torch.float).cuda()
        #Conv1
        self.conv1 = nn.Conv2d(1,64,7,padding=3)
        self.maxpool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(64)
        
        #Resl Block
        self.resl_64 = self.bottleNeck_block(64)
        self.resl_64_128= self.bottleNeck_block_dim_inc(64)
        self.resl_128 = self.bottleNeck_block(128)
        self.resl_128_256= self.bottleNeck_block_dim_inc(128)
        self.resl_256 = self.bottleNeck_block(256)
        self.resl_256_512= self.bottleNeck_block_dim_inc(256)
        self.resl_512 = self.bottleNeck_block(512)
        
        #Fully Connected
        self.fc = nn.Sequential(
                nn.Linear(512*8*8,1000),
                nn.Linear(1000,classes),
                nn.Softmax(dim=1)
                )
        
    def bottleNeck_block(self,num_f):
        seq = nn.Sequential(
                nn.Conv2d(num_f,num_f,1),
                nn.BatchNorm2d(num_f),
                nn.ReLU(),
                nn.Conv2d(num_f,num_f,3,padding=1),
                nn.BatchNorm2d(num_f),
                nn.ReLU(),
                nn.Conv2d(num_f,num_f,1),
                nn.BatchNorm2d(num_f),
                )
        return seq
    def bottleNeck_block_dim_inc(self,num_f):
        seq = nn.Sequential(
                nn.Conv2d(num_f,num_f,1),
                nn.BatchNorm2d(num_f),
                nn.ReLU(),
                nn.Conv2d(num_f,num_f,3, padding=1),
                nn.BatchNorm2d(num_f),
                nn.ReLU(),
                nn.Conv2d(num_f,num_f*2,1),
                nn.BatchNorm2d(num_f),
                nn.MaxPool2d(2)
                )
        return seq

    def forward(self,x):
        #Pre-Processing
        x = F.conv2d(x, self.hpfs, padding=2)
        #Conv1
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        
        #ResL Block
        x_id = x
        x = self.resl_64(x)
        x += x_id
        x = torch.relu(x)
        '''n1 n2 n3 n4 // 2 3 5 2'''
        #n11
        x_id = x
        x = self.resl_64(x)
        x += x_id
        x = torch.relu(x)
        #n12
        x_id = x
        x += x_id
        x = self.resl_64_128(x)
        x = torch.relu(x)
        #n21
        x_id = x
        x += x_id
        x = self.resl_128(x)
        x = torch.relu(x)
        #n22
        x_id = x
        x += x_id
        x = self.resl_128(x)
        x = torch.relu(x)
        #n23
        x_id = x
        x += x_id
        x = self.resl_128_256(x)
        x = torch.relu(x)
        #n31
        x_id = x
        x += x_id
        x = self.resl_256(x)
        x = torch.relu(x)
        #n32
        x_id = x
        x += x_id
        x = self.resl_256(x)
        x = torch.relu(x)
        #n33
        x_id = x
        x += x_id
        x = self.resl_256(x)
        x = torch.relu(x)
        #n34
        x_id = x
        x += x_id
        x = self.resl_256(x)
        x = torch.relu(x)
        #n35
        x_id = x
        x += x_id
        x = self.resl_256_512(x)
        x = torch.relu(x)
        #n41
        x_id = x
        x += x_id
        x = self.resl_512(x)
        x = torch.relu(x)
        #n42
        x_id = x
        x += x_id
        x = self.resl_512(x)
        x = torch.relu(x)
        #AvgPooling
        x = F.avg_pool2d(x,2)
        x = x.view(-1,512*8*8)
        #FullyConnected
        x = self.fc(x)
        return x