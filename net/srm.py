# -*- coding: utf-8 -*-
import torch
import numpy as np

def srm_filter(Useful=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    srm = np.ndarray((30,5,5),dtype = np.float16)
  
    srm.fill(0)
    #1st SPAM  [-1, +1]
    rotation_1st_3rd = [[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]
    n = 0
    for rot in rotation_1st_3rd:
        srm[n][2][2] = -1
        srm[n][2+rot[0]][2+rot[1]] = 1
        n+=1
          
    #2nd SPAM [+1, -2, +1]
    rotation_2nd = [[0,1],[1,1],[1,0],[1,-1]]
    for rot in rotation_2nd:
        srm[n][2][2] = -2
        srm[n][2+rot[0]][2+rot[1]] = 1
        srm[n][2-rot[0]][2-rot[1]] = 1
        n+=1
      
    #3rd SPAM [-1, +3,-3,+1]
    for rot in rotation_1st_3rd:
        srm[n][2][2] = 3
        srm[n][2+rot[0]*2][2+rot[1]*2] = 1
        srm[n][2+rot[0]][2+rot[1]] = -3
        srm[n][2-rot[0]][2-rot[1]] = -1
        srm[n]/=3
        n+=1
      
    #EDGE SPAM [-1, +2,-1]
    #          [+2, -4,+2]
    rotation_edge = rotation_1st_3rd+rotation_1st_3rd                 
    for k in range(8):
        rot = rotation_edge[(k+4):(k+4+5)]
        srm[n][2][2] = -4
        srm[n][2+rot[0][0]][2+rot[0][1]] = +2
        srm[n][2+rot[1][0]][2+rot[1][1]] = -1
        srm[n][2+rot[2][0]][2+rot[2][1]] = +2
        srm[n][2+rot[3][0]][2+rot[3][1]] = -1
        srm[n][2+rot[4][0]][2+rot[4][1]] = +2
        srm[n]/=4
        n+=1

    #SQUARE [-1,2,-1]
    #       [+2,-4,+2]
    #       [-1,2,-1]
    square33 = [[0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0]]
    srm[n] = np.array(square33, dtype=np.float16)/4
    n+=1
  
    square55 = [[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
    srm[n] = np.array(square55, dtype=np.float16)/12
  
    if Useful:  
        useful_filters= np.ndarray((10,1,5,5),dtype = np.float16)
        useful_filters.fill(0)
        
        useful_filters[0:,:,:,:]=srm[8,:,:]
        useful_filters[1:,:,:,:]=srm[9,:,:]
        useful_filters[2:,:,:,:]=srm[10,:,:]
        useful_filters[3:,:,:,:]=srm[11,:,:]
        useful_filters[4:,:,:,:]=srm[12,:,:]
        useful_filters[5:,:,:,:]=srm[14,:,:]
        useful_filters[6:,:,:,:]=srm[18,:,:]
        useful_filters[7:,:,:,:]=srm[19,:,:]
        useful_filters[8:,:,:,:]=srm[24,:,:]
        useful_filters[9:,:,:,:]=srm[29,:,:]
  
        useful_filters = torch.tensor(useful_filters, dtype = torch.float, requires_grad = False)
    else:
        useful_filters = np.ndarray((30,1,5,5),dtype = np.float32) 
        useful_filters.fill(0)
        useful_filters[:,0,:,:] = srm[:,:,:]
        useful_filters = torch.tensor(useful_filters, dtype=torch.float, requires_grad= False)
    
    useful_filters = useful_filters.to(device)
    
    return useful_filters