# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

'''
overall networks
input > conv > dense > conv > pooling > dense > conv > pooling > dense > pooling > Linear > predict
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class bn_relu_conv(nn.Module):
    '''Forwarding Conv Block(bn - relu - conv)'''
    def __init__(self, in_f, out_f, k_size, stride, padding, bias=False):
        super(bn_relu_conv, self).__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_f),
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size=k_size, stride=stride,
                      padding=padding, bias=bias),
            )
    def forward(self, x):
        return self.seq(x)
    

class bottleneck_conv(nn.Module):
    '''bottleneck strc 1x1 >> 3x3'''
    def __init__(self, in_f, growth_rate, drop_rate=0.2):
        super(bottleneck_conv, self).__init__()
        self.seq = nn.Sequential(
            bn_relu_conv(in_f=in_f, out_f=growth_rate*4, k_size=1, stride=1, padding=0),
            bn_relu_conv(in_f=growth_rate*4, out_f=growth_rate, k_size=3, stride=1, padding=1)
            )
        self.drop_rate = drop_rate
        
    def forward(self, x):
        bottleneck_output = self.seq(x)
        if self.drop_rate > 0:
            bottleneck_output = F.dropout(bottleneck_output, p=self.drop_rate, training=self.training)
        
        return torch.cat((x,bottleneck_output), 1)
 
class transition_conv(nn.Module):
    '''reducing n-features conv - pooling'''
    def __init__(self, in_f, theta=0.5, bias=False):
        super(transition_conv, self).__init__()
        self.seq = nn.Sequential(
            bn_relu_conv(in_f=in_f, out_f=int(in_f*theta), k_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
            )
        
    def forward(self, x):
        return self.seq(x)

class DenseBlock(nn.Sequential):
  def __init__(self, in_f, num_bottleneck_layers, growth_rate, drop_rate=0.2):
      super(DenseBlock, self).__init__()
                        
      for i in range(num_bottleneck_layers):
          nin_bottleneck_layer = in_f + growth_rate * i
          self.add_module('bottleneck_layer_%d' % i, bottleneck_conv(in_f=nin_bottleneck_layer, growth_rate=growth_rate, drop_rate=drop_rate))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, num_layers=100, theta=0.5, drop_rate=0.2, num_classes=10):
        super(DenseNet, self).__init__()
        assert (num_layers - 4) % 6 == 0
        self.hpf = torch.tensor([[[[-1,2,-2,2,-1],
                    [2,-6,8,-6,2],
                    [-2,8,-12,8,-2],
                    [2,-6,8,-6,2],
                    [-1,2,-2,2,-1]]]], dtype=torch.float)/12
        self.hpf = self.hpf.to(device)
        # (num_layers-4)//6 16
        num_bottleneck_layers = (num_layers - 4) // 6
        
        # 32 x 32 x 3 --> 32 x 32 x (growth_rate*2)
        self.dense_init = nn.Conv2d(1, growth_rate*2, kernel_size=3, stride=1, padding=1, bias=True)
                
        # 32 x 32 x (growth_rate*2) --> 32 x 32 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)]
        self.dense_block_1 = DenseBlock(in_f=growth_rate*2, num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)

        # 32 x 32 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)] --> 16 x 16 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)]*theta
        nin_transition_layer_1 = (growth_rate*2) + (growth_rate * num_bottleneck_layers) 
        self.transition_layer_1 = transition_conv(in_f=nin_transition_layer_1, theta=theta)
        
        # 16 x 16 x nin_transition_layer_1*theta --> 16 x 16 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)]
        self.dense_block_2 = DenseBlock(in_f=int(nin_transition_layer_1*theta), num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)

        # 16 x 16 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)] --> 8 x 8 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)]*theta
        nin_transition_layer_2 = int(nin_transition_layer_1*theta) + (growth_rate * num_bottleneck_layers) 
        self.transition_layer_2 = transition_conv(in_f=nin_transition_layer_2, theta=theta)
        
        # 8 x 8 x nin_transition_layer_2*theta --> 8 x 8 x [nin_transition_layer_2*theta + (growth_rate * num_bottleneck_layers)]
        self.dense_block_3 = DenseBlock(in_f=int(nin_transition_layer_2*theta), num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)
        
        nin_transition_layer_3 = int(nin_transition_layer_2*theta) + (growth_rate * num_bottleneck_layers) 
        self.transition_layer_3 = transition_conv(in_f=nin_transition_layer_3, theta=theta)
        
        # 8 x 8 x nin_transition_layer_2*theta --> 8 x 8 x [nin_transition_layer_2*theta + (growth_rate * num_bottleneck_layers)]
        self.dense_block_4 = DenseBlock(in_f=int(nin_transition_layer_3*theta), num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)
        
        nin_transition_layer_4 = int(nin_transition_layer_3*theta) + (growth_rate * num_bottleneck_layers) 
        self.transition_layer_4 = transition_conv(in_f=nin_transition_layer_4, theta=theta)
        self.dense_block_5 = DenseBlock(in_f=int(nin_transition_layer_4*theta), num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)
        
        
        nin_fc_layer = int(nin_transition_layer_4*theta) + (growth_rate * num_bottleneck_layers) 
        
        # [nin_transition_layer_2*theta + (growth_rate * num_bottleneck_layers)] --> num_classes
        self.fc_layer = nn.Sequential(
            nn.Linear(nin_fc_layer, num_classes),
            nn.Softmax(dim=1)
            )
        
    def forward(self, x):
        x = F.conv2d(x, self.hpf)
        x = self.dense_init(x)
        x = self.dense_block_1(x)
        x = self.transition_layer_1(x)
        x = self.dense_block_2(x)
        x = self.transition_layer_2(x)
        x = self.dense_block_3(x)
        x = self.transition_layer_3(x)
        x = self.dense_block_4(x)
        x = self.transition_layer_4(x)
        x = self.dense_block_5(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
def Model(num_classes):
    return DenseNet(growth_rate=12, num_layers=52, theta=0.5, drop_rate=0.2, num_classes=num_classes)
