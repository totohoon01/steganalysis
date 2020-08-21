# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:56:44 2020

@author: IVCLAB
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_numpy_data(alg_dict, train=True):
    #Data Constants
    DATASIZE = 30000 if train else 10000
    NUMCLASSES = sum(list(map(int, alg_dict.values())))
    CODE = "TR" if train else "TE"
    PATH = "D:/steganography/numpy image/ori"

    #Data Holder
    w, h = 256, 256
    data = np.zeros([DATASIZE*NUMCLASSES, 1, w, h], np.uint8)
    
    #loading data from .npy files
    count = 0
    for key, value in alg_dict.items():
        if(value):
            #cover
            if(key == "COVER"):
                loadedData = np.load(f"{PATH}/WOW(fixed key)/{CODE}(bpp0).npy")
            else:
            #stego
                loadedData = np.load(f"{PATH}/{key}/{CODE}(bpp40).npy")
            data[DATASIZE*count : DATASIZE*(count+1), 0, :, :] = loadedData[:, :, :, 0]
            count += 1
            print(key, "done")
            del loadedData
    
    #setting labels, label always follow COVER, LSB, PVD, WOW, UNIWARD order
    label = np.zeros([data.shape[0], NUMCLASSES], np.int8)
    for i in range(NUMCLASSES):
        label[int(data.shape[0]/NUMCLASSES)*i : int(data.shape[0]/NUMCLASSES)*(i+1), i] = 1
        
    return data, label

def create_data_loader(data, label, batch_size=60, train=True):
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).long()
    ds = TensorDataset(data, label)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=train)
    return data_loader

def deal_name(Algs, model):
    algs_keys = list(Algs.keys())
    algs_values = list(Algs.values())
    idx = [idx for idx, value in enumerate(algs_values) if value ==1]
    name = model + '_' + "".join([algs_keys[i][0] for i in idx])
    return name