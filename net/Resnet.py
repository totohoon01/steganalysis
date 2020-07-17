# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

class IdentityPadding(nn.Module):
	def __init__(self, in_channels, out_channels, stride):
		super(IdentityPadding, self).__init__()
		
		self.pooling = nn.MaxPool2d(1, stride=stride)
		self.add_channels = out_channels - in_channels
    
	def forward(self, x):
		out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
		out = self.pooling(out)
		return out

class NoneBottleNeckResidualBlock(nn.Module):
	"""
	non-Bottle-Neck Strucure
	"""
	def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
		super(NoneBottleNeckResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
			                   stride=stride, padding=1, bias=False) 
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
			                   stride=1, padding=1, bias=False) 
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.stride = stride
		
		if down_sample:
			self.down_sample = IdentityPadding(in_channels, out_channels, stride)
		else:
			self.down_sample = None

	def forward(self, x):
		shortcut = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.down_sample is not None:
			shortcut = self.down_sample(x)

		out += shortcut
		out = self.relu(out)
		return out

class BottleNeckResidualBlock(nn.Module):
	"""
	Bottle-Neck Strucure
	"""
	def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
		super(BottleNeckResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
			                   stride=stride, padding=0, bias=False) 
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
			                   stride=1, padding=1, bias=False) 
		self.bn2 = nn.BatchNorm2d(out_channels)
		
		self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, 
			                   stride=1, padding=0, bias=False) 
		self.bn3 = nn.BatchNorm2d(out_channels)
		
		self.stride = stride
		
		if down_sample:
			self.down_sample = IdentityPadding(in_channels, out_channels, stride)
		else:
			self.down_sample = None

	def forward(self, x):
		shortcut = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		
		out = self.conv3(out)
		out = self.bn3(out) 

		if self.down_sample is not None:
			shortcut = self.down_sample(x)

		out += shortcut
		out = self.relu(out)
		return out    

class DimesionIncreasing(nn.Module):
	def __init__(self, in_channels, out_channels, stride=2, down_sample=True):
		super(DimesionIncreasing, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
			                   stride=stride, padding=1, bias=False) 
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.stride = stride

	def forward(self, x):

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		return out    
	
	
class Model(nn.Module):
	def __init__(self, block, num_layers, num_classes=2):
		super(Model, self).__init__()
		self.hpf = torch.tensor([[[[-1,2,-2,2,-1],
                    [2,-6,8,-6,2],
                    [-2,8,-12,8,-2],
                    [2,-6,8,-6,2],
                    [-1,2,-2,2,-1]]]], dtype=torch.float)/12
		self.hpf = self.hpf.cuda()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, 
							   stride=1, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=2,padding=0)
		
		#Dim inc Blocks
		self.Dim_inc1 = self.get_layers(DimesionIncreasing,1,64,128,2)
		self.Dim_inc2 = self.get_layers(DimesionIncreasing,1,128,256,2)
		self.Dim_inc3 = self.get_layers(DimesionIncreasing,1,256,512,2)

		# ResL Blocks
		num_ResL = 2 if block == NoneBottleNeckResidualBlock else 4
		self.ResL = self.get_layers(NoneBottleNeckResidualBlock, num_ResL, 64, 64, 1)
		self.n1ResL = self.get_layers(block, num_layers[0], 64, 64, 1)
		self.n2ResL = self.get_layers(block, num_layers[1], 128, 128, 1)
		self.n3ResL = self.get_layers(block, num_layers[2], 256, 256, 1)
		self.n4ResL = self.get_layers(block, num_layers[3], 512, 512, 1)
		
		each = 2 if block == NoneBottleNeckResidualBlock else 3
		print("Total num of Conv-layers: ", 6 + num_ResL*2 + sum(num_layers)*each)
			
		# output layers
		self.avg_pool = nn.AvgPool2d(16, stride=1)
		self.fc1 = nn.Linear(512, 1000)
		self.fc_out = nn.Sequential(
                nn.Linear(1000, num_classes),
                nn.Softmax(dim=1)
		)
		#initialize
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.uniform_(-0.01, 0.01)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
	def get_layers(self, block, num_layers, in_channels, out_channels, stride):
		if num_layers == 0: return

		if stride == 2:
			down_sample = True
		else:
			down_sample = False
		
		layers_list = nn.ModuleList(
			[block(in_channels, out_channels, stride, down_sample)])
			
		for _ in range(num_layers - 1):
			layers_list.append(block(out_channels, out_channels))
		
		return nn.Sequential(*layers_list)
		
	def forward(self, x):
		#HPF sub-Networks
		x = F.conv2d(x, self.hpf, padding=2)
		
		#ResL sub-Networks
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		
		x = self.ResL(x)  #논문에 정확한 구조가 없어서 Conv갯수로 맞춤(4)
		
		if self.n1ResL is not None:
			x = self.n1ResL(x)
		x = self.Dim_inc1(x)
		
		if self.n2ResL is not None:
			x = self.n2ResL(x)
		x = self.Dim_inc2(x)
		
		if self.n3ResL is not None:
			x = self.n3ResL(x)
		x = self.Dim_inc3(x)
		
		if self.n4ResL is not None:
			x = self.n4ResL(x)
		
		#Classification sub-Network
		x = self.avg_pool(x)
		x = x.view(-1, 512)
		x = self.fc1(x)
		x = self.fc_out(x)
		return x