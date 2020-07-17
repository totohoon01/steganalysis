# -*- coding: utf-8 -*-
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
        self.conv1 = nn.Conv2d(10,60,kernel_size=5,padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(60)
        self.layer1 = nn.Sequential(
        	nn.Conv2d(60,60,kernel_size=5,padding=2),
        	nn.ReLU(),
        	nn.Conv2d(60,60,kernel_size=5,padding=2),
        	)
        self.conv2 = nn.Conv2d(60,120,kernel_size=5,padding=2)
        self.bn2 = nn.BatchNorm2d(120)
        self.layer2 = nn.Sequential(
        	nn.Conv2d(120,120,kernel_size=5,padding=2),
        	nn.ReLU(),
        	nn.Conv2d(120,120,kernel_size=5,padding=2),
        	)
        self.conv3 = nn.Conv2d(120,240,kernel_size=5,padding=2)
        self.bn3 = nn.BatchNorm2d(240)
        self.layer3 = nn.Sequential(
        	nn.Conv2d(240,240,kernel_size=5,padding=2),
        	nn.ReLU(),
        	nn.Conv2d(240,240,kernel_size=5,padding=2),
        	)
        self.conv4 = nn.Conv2d(240,480,kernel_size=5,padding=2)
        self.bn4 = nn.BatchNorm2d(480)
        self.layer4 = nn.Sequential(
        	nn.Conv2d(480,480,kernel_size=5,padding=2),
        	nn.ReLU(),
        	nn.Conv2d(480,480,kernel_size=5,padding=2),
        	)
        self.conv5 = nn.Conv2d(480,960,kernel_size=5,padding=2)
        self.bn5 = nn.BatchNorm2d(960)
        self.layer5 = nn.Sequential(
        	nn.Conv2d(960,960,kernel_size=5,padding=2),
        	nn.ReLU(),
        	nn.Conv2d(960,960,kernel_size=5,padding=2),
        	)
        self.AvgPooling = nn.AvgPool2d(5,2,2)
        self.GlobalPooling = nn.AvgPool2d(8)
        self.fc = nn.Sequential(
                nn.Linear(960,classes),
                nn.Softmax(dim=1)
                )
        
    def forward(self, x):
        #prep
        x = F.conv2d(x, self.hpf, padding=2)
        #Pooling
        x = self.conv1(x)
        x = torch.abs(x)
        x = self.bn1(x)
        x = torch.tanh(x)
        x = self.AvgPooling(x)
        #Shorcut-1
        idf_x = x
        x = self.layer1(x)
        x += idf_x
        x = torch.relu(x)
        #Pooling
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.AvgPooling(x)
        #Shortcut - 2
        idf_x = x
        x = self.layer2(x)
        x += idf_x
        x = torch.relu(x)
        #Pooling
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.AvgPooling(x)
        #Shortcut - 3
        idf_x = x
        x = self.layer3(x)
        x += idf_x
        x = torch.relu(x)
        #Pooling
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.AvgPooling(x)
        #Shortcut - 4
        idf_x = x
        x = self.layer4(x)
        x += idf_x
        x = torch.relu(x)
        #Pooling
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.AvgPooling(x)
        #Shortcut - 5
        idf_x = x
        x = self.layer5(x)
        x += idf_x
        x = torch.relu(x)
        #Globle Pooling
        x = self.GlobalPooling(x)
        #Fully-Connected
        x = x.view(-1,960)
        x = self.fc(x)
        
        return x
 