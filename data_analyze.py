# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:59:31 2020

@author: IVCLAB
"""
import collections
import numpy as np
import sys


def data_analyzing(ypreds, lens):
    if ypreds == None:
        sys.exit()
        
    ypreds = np.array(ypreds.cpu())
    rst = np.zeros([lens,lens], dtype=np.int)
    idx = 0
    for i in range(lens):
        counts = collections.Counter(ypreds[10000*i:10000*(i+1)])
        for j in range(lens):
            rst[idx][j] = counts[j]
        idx+=1
           
    return rst
