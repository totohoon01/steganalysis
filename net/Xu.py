# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

def hpf():
    arr = torch.tensor([[[[-1,2,-2,2,-1],
                       [2,-6,8,-6,2],
                       [-2,8,-12,8,-2],
                       [2,-6,8,-6,2],
                       [-1,2,-2,2,-1]]]], dtype=torch.float)/12
    
    return arr

class Model(nn.Module):
    def __init__(self, classes=3):
        """
        PP Net 60 to 1920
        """
        super(Model, self).__init__()
        #prep
        self.hpf = hpf()
        #Group1
        self.conv1 = nn.Conv2d(1,8,5,1,2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        #Group2
        self.conv2 = nn.Conv2d(8,16,5,1,2)
        self.bn2 = nn.BatchNorm2d(16)
        #Group 3~5
        self.conlayer3 = self.Conv_forward(16,32,5,1,2)
        self.conlayer4 = self.Conv_forward(32,64,5,1,2)
        self.conlayer5 = self.Conv_forward(64,128,5,1,2)
        self.AvgPooling = nn.AvgPool2d(5,2,2)
        self.GlobalPooling = nn.AvgPool2d(16)
        #Fully Connected
        self.fc = nn.Sequential(
                nn.Linear(128,classes),
                nn.Softmax(dim=1)
                )
    def Conv_forward(self, in_featere, out_feature, k_size, stride, padding, bias=False):
        Seq = nn.Sequential(
                nn.Conv2d(in_featere, out_feature, k_size,stride,padding,bias=bias),
                nn.BatchNorm2d(out_feature),
                nn.ReLU(),
                )
        
        return Seq
    
    def forward(self, x):
        #prep
        x = F.conv2d(x, self.hpf, padding=2)
        #Group1
        x = self.conv1(x)
        x = torch.abs(x)
        x = self.bn1(x)
        x = torch.tanh(x)
        x = self.AvgPooling(x)
        #Group2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.tanh(x)
        x = self.AvgPooling(x)
        #Group3~5
        x = self.conlayer3(x)
        x = self.AvgPooling(x)
        x = self.conlayer4(x)
        x = self.AvgPooling(x)
        x = self.conlayer5(x)
        x = self.GlobalPooling(x)        
        
        #Flatten & FC layer
        x = x.view(-1,128)
        x = self.fc(x)
        return x