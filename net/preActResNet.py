# -*- coding: utf-8 -*-
"""
Residual 블록의 순서를 
Conv - bn - relu - con - bn - add - relu에서
bn - relu - conv - bn - relu - conv - add로 변경한다.
"""
import torch
from torch import nn
import torch.nn.functional as F
import Nets.srm as srm


class Model(nn.Module):
    def __init__(self, classes):
        super(Model, self).__init__()
        #High-pass filter
        self.hpf = srm.srm_filter(Useful=True)
        inif = 60
        #Forward Block
        self.L1 = self.dimension_inc(10, inif, 3, 1, 1, False)
        self.L2 = self.dimension_inc(inif, inif*2, 3, 1, 1, True)
        self.L3 = self.dimension_inc(inif*2, inif*4, 3, 1, 1, True)
        self.L4 = self.dimension_inc(inif*4, inif*8, 3, 1, 1, True)
        self.L5 = self.dimension_inc(inif*8, inif*16, 3, 1, 1, True)
        
        #Residual Block
        self.ResL1 = self.residual_block(inif,3,1,1)
        self.ResL2 = self.residual_block(inif*2,3,1,1)
        self.ResL3 = self.residual_block(inif*4,3,1,1)
        self.ResL4 = self.residual_block(inif*8,3,1,1)
        self.ResL5 = self.residual_block(inif*16,3,1,1)
        
        #Pooling & FC
        self.AvgPooling = nn.AvgPool2d(5,2,2)
        self.GlobalPooling = nn.AvgPool2d(8)
        self.fc = nn.Sequential(
                nn.Linear(960,classes),
                nn.Softmax(dim=1)
                )
        
    def dimension_inc(self, in_features, out_features, kernel_size, stride, padding, bias):
        Seq = nn.Sequential(
                    nn.BatchNorm2d(in_features),
                    nn.ReLU(),
                    nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    )
        return Seq
    def residual_block(self, features, kernel_size, stride, padding):
        Seq = nn.Sequential(
                nn.BatchNorm2d(features),
                nn.ReLU(),
                nn.Conv2d(features, features, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.BatchNorm2d(features),
                nn.ReLU(),
                nn.Conv2d(features, features, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                )
        return Seq
    
    def forward(self, x):
        #Prep
        x = F.conv2d(x,self.hpf,padding=2) #1 >> 10
        #L1
        x = self.L1(x)
        x = self.AvgPooling(x)
        #ResL1
        x_id = x
        x = self.ResL1(x)
        x += x_id
        #L2
        x = self.L2(x)
        x = self.AvgPooling(x)
        #ResL1
        x_id = x
        x = self.ResL2(x)
        x += x_id
        #L3
        x = self.L3(x)
        x = self.AvgPooling(x)
        #ResL1
        x_id = x
        x = self.ResL3(x)
        x += x_id
        #L4
        x = self.L4(x)
        x = self.AvgPooling(x)
        #ResL1
        x_id = x
        x = self.ResL4(x)
        x += x_id
        #L5
        x = self.L5(x)
        x = self.AvgPooling(x)
        #ResL1
        x_id = x
        x = self.ResL5(x)
        x += x_id
        #Fully-Connected
        x = self.GlobalPooling(x)
        x = x.view(-1, 960)
        x = self.fc(x)
        return x
    