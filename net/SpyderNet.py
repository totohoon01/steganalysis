# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

class conv_module(nn.Module):
    def __init__(self,in_f, out_f, types='normal'):
        super(conv_module, self).__init__()
        if types == 'preact':
            self.seq = nn.Sequential(
                nn.BatchNorm2d(in_f),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size=3, padding=1),
                nn.AvgPool2d(2)
                )
        else:
            self.seq = nn.Sequential(
                nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.AvgPool2d(2)
                )
        
    def forward(self, x):
        return self.seq(x)


def transtion_module(*args):
    '''concat features and AvgPool'''
    cated = torch.cat([*args], dim=1)
    cated = F.avg_pool2d(cated, kernel_size=2)
    return cated

class FCCNN(nn.Module):
    def __init__(self, types='normal',num_classes=2):
        super(FCCNN, self).__init__()
        self.hpf = torch.tensor([[[[-1,2,-2,2,-1],
                    [2,-6,8,-6,2],
                    [-2,8,-12,8,-2],
                    [2,-6,8,-6,2],
                    [-1,2,-2,2,-1]]]], dtype=torch.float)/12
        self.hpf = self.hpf.cuda()
        #path1
        self.s11 = conv_module(in_f=1, out_f=30, types=types)
        self.s12 = conv_module(in_f=30, out_f=60, types=types)
        self.s13 = conv_module(in_f=60, out_f=120, types=types)
        self.s14 = conv_module(in_f=120, out_f=240, types=types)
        self.s15 = conv_module(in_f=240, out_f=480, types=types)
        #path2
        self.s21 = conv_module(in_f=1, out_f=30, types=types)
        self.s22 = conv_module(in_f=30, out_f=60, types=types)
        self.s23 = conv_module(in_f=60, out_f=120, types=types)
        self.s24 = conv_module(in_f=120, out_f=240, types=types)
        self.s25 = conv_module(in_f=240, out_f=480, types=types)
        
        #FC-layer
        self.GAP = nn.AvgPool2d(8)
        self.fc = nn.Sequential(
            nn.Linear(in_features=960, out_features=num_classes),
            nn.Softmax(dim=1)
            )
    def forward(self, x):
        x = F.conv2d(x, self.hpf)
        #stage1
        path1 = self.s11(x) # 128
        path2 = self.s21(x)
        cated = transtion_module(path1,path2) # 64
        #stage2
        path1 = self.s12(path1) 
        path2 = self.s22(path2)
        path1 += cated
        path2 += cated
        cated = transtion_module(path1,path2) #32
        #stage3
        path1 = self.s13(path1) 
        path2 = self.s23(path2)
        path1 += cated
        path2 += cated
        cated = transtion_module(path1,path2) #16
        #stage4
        path1 = self.s14(path1) 
        path2 = self.s24(path2)
        path1 += cated
        path2 += cated
        cated = transtion_module(path1,path2) #8
        #stage5
        path1 = self.s15(path1)
        path2 = self.s25(path2)
        path1 += cated
        path2 += cated
        
        outputs = torch.cat([path1,path2], dim=1)
        outputs = self.GAP(outputs)
        outputs = outputs.view(-1, 960)
        outputs = self.fc(outputs)
        return outputs
    
class FCCNN2(nn.Module):
    def __init__(self, types='normal',num_classes=2):
        super(FCCNN2, self).__init__()
        self.hpf = torch.tensor([[[[-1,2,-2,2,-1],
                    [2,-6,8,-6,2],
                    [-2,8,-12,8,-2],
                    [2,-6,8,-6,2],
                    [-1,2,-2,2,-1]]]], dtype=torch.float)/12
        self.hpf = self.hpf.cuda()
        
        #path1
        self.s11 = conv_module(in_f=1, out_f=20, types=types)
        self.s12 = conv_module(in_f=20, out_f=60, types=types)
        self.s13 = conv_module(in_f=60, out_f=180, types=types)
        self.s14 = conv_module(in_f=180, out_f=320, types=types)
        #path2
        self.s21 = conv_module(in_f=1, out_f=20, types=types)
        self.s22 = conv_module(in_f=20, out_f=60, types=types)
        self.s23 = conv_module(in_f=60, out_f=180, types=types)
        self.s24 = conv_module(in_f=180, out_f=320, types=types)
        #path3
        self.s31 = conv_module(in_f=1, out_f=20, types=types)
        self.s32 = conv_module(in_f=20, out_f=60, types=types)
        self.s33 = conv_module(in_f=60, out_f=180, types=types)
        self.s34 = conv_module(in_f=180, out_f=320, types=types)
        #FC-layer
        self.GAP = nn.AvgPool2d(16)
        self.fc = nn.Sequential(
            nn.Linear(in_features=960, out_features=num_classes),
            nn.Softmax(dim=1)
            )
    def forward(self, x):
        x = F.conv2d(x, self.hpf)
        #stage1
        path1 = self.s11(x) # 128
        path2 = self.s21(x)
        path3 = self.s31(x)
        cated = transtion_module(path1,path2,path3) # 64
        #stage2
        path1 = self.s12(path1) 
        path2 = self.s22(path2)
        path3 = self.s32(path3)
        path1 += cated
        path2 += cated
        path3 += cated
        cated = transtion_module(path1,path2,path3) #32
        #stage3
        path1 = self.s13(path1) 
        path2 = self.s23(path2)
        path3 = self.s33(path3)
        path1 += cated
        path2 += cated
        path3 += cated
        cated = transtion_module(path1,path2,path3) #16
        #stage4
        path1 = self.s14(path1) 
        path2 = self.s24(path2)
        path3 = self.s34(path3)
        #out
        outputs = torch.cat([path1,path2,path3], dim=1)
        outputs = self.GAP(outputs)
        outputs = outputs.view(-1, 960)
        outputs = self.fc(outputs)
        return outputs
   