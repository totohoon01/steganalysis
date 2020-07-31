# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import Nets.srm as srm

class conv_forward(nn.Module):
    def __init__(self, in_f, out_f, kernel_size=5, padding=2, bias=False):
        super(conv_forward, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_f),
            nn.ReLU(),
            nn.AvgPool2d(5, 2, 2),
            )

    def forward(self, x):
        
        return self.seq(x)

class resl_forward(nn.Module):
    def __init__(self, in_f, out_f, kernel_size=5, padding=2, bias=True):
        super(resl_forward, self).__init__()
        mid_f = int((in_f + out_f)/2)
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=in_f, out_channels=mid_f, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(mid_f),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_f, out_channels=out_f, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_f),
            nn.ReLU()
            )
        self.conv_shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_f),
            nn.ReLU(),
            )
    def forward(self, x):
        idf = x
        idf = self.conv_shortcut(idf)
        x = self.seq(x)
        x += idf
        return x

class Model(nn.Module):
    def __init__(self, classes=3):
        """
        PP Net 60 to 1920
        """
        super(Model, self).__init__()
        self.hpf = srm.srm_filter(Useful=True)
        self.conv1 = conv_forward(10, 30)
        self.layer1 = resl_forward(30, 60)
        self.conv2 = conv_forward(60, 90)
        self.layer2 = resl_forward(90, 120)
        self.conv3 = conv_forward(120, 180)
        self.layer3 = resl_forward(180, 240)
        self.conv4 = conv_forward(240, 360)
        self.layer4 = resl_forward(360, 480)
        self.conv5 = conv_forward(480, 720)
        self.layer5 = resl_forward(720, 960)
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
        x = self.layer1(x)
        x = self.conv2(x)
        x = self.layer2(x)
        x = self.conv3(x)
        x = self.layer3(x)
        x = self.conv4(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.layer5(x)
        
        #Globle Pooling
        x = self.GlobalPooling(x)
        #Fully-Connected
        x = x.view(-1,960)
        x = self.fc(x)
        
        return x
 