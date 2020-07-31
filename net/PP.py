# -*- coding: utf-8 -*-
"""
Networks for Multi-Class Classification with SRM filters
"""
import torch
from torch import nn
import torch.nn.functional as F
import Nets.srm as srm

class Model(nn.Module):
    def __init__(self, classes=3):
        """
        PP Net 60 to 1920
        """
        super(Model, self).__init__()
        self.hpf = srm.srm_filter(Useful=True)
        self.conv1 = nn.Conv2d(self.hpf.shape[0],60,kernel_size=5,padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(60)
        self.conv2 = nn.Conv2d(60,120,kernel_size=5,padding=2)
        self.bn2 = nn.BatchNorm2d(120)
        self.conv3 = nn.Conv2d(120,240,kernel_size=5,padding=2)
        self.bn3 = nn.BatchNorm2d(240)
        self.conv4 = nn.Conv2d(240,480,kernel_size=5,padding=2)
        self.bn4 = nn.BatchNorm2d(480)
        self.conv5 = nn.Conv2d(480,960,kernel_size=5,padding=2)
        self.bn5 = nn.BatchNorm2d(960)
        self.conv6 = nn.Conv2d(960,1920,kernel_size=5,padding=2)
        self.bn6 = nn.BatchNorm2d(1920)
        self.AvgPooling = nn.AvgPool2d(5,2,2)
        self.GlobalPooling = nn.AvgPool2d(8)
        self.fc = nn.Sequential(
                nn.Linear(1920,classes),
                nn.Softmax(dim=1)
                )
        
    def forward(self, x):
        #prep
        x = F.conv2d(x, self.hpf, padding=2)
        #layer1
        x = self.conv1(x)
        x = torch.abs(x)
        x = self.bn1(x)
        x = torch.tanh(x)
        x = self.AvgPooling(x)
        #layer2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.tanh(x)
        x = self.AvgPooling(x)
        #layer3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.AvgPooling(x)
        #layer4
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.AvgPooling(x)
        #layer5
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.AvgPooling(x)
        #layer6
        x = self.conv6(x)
        x = self.bn6(x)
        x = torch.relu(x)
        x = self.GlobalPooling(x)
        #Fully-Connected
        x = x.view(-1,1920)
        x = self.fc(x)
        
        return x