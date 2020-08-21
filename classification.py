# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:01:01 2020

@author: IVCLAB
"""
import torch
from torch import nn, optim
import os
import tqdm
import importlib

#modules
import data_processing as dp
import data_analyze as da

def Train_Networks(Net, name, alg_dict, batch_size=60):
    # Make folder that save results
    save_path = f"./new_result/{name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #setting data
    data, label = dp.load_numpy_data(alg_dict, train=True)
    train_data = dp.create_data_loader(data, label, train=True)
    del data, label
    
    #setting networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = Net.to(device)
    lr = 0.001
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    
    # EPOCH
    EPOCH = 100

    # Training Session
    train_count = 0
    for epo in range(EPOCH):
        #Save Parameters
        if(epo % 10 == 0):
            print('Paramter Saved')
            torch.save(Net.state_dict(), save_path + "epoch{epo}.prm")

        for X, Y in train_data:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            y_pred = Net(X)
            loss = loss_fn(y_pred, torch.max(Y, 1)[1])
            loss.backward()
            optimizer.step()

            # Predict State
            if(train_count % 100 == 0):
                print(f"Epoch: {epo}/100, renew: {train_count}, loss: {loss.item()}")
                print("Prediction : ", y_pred[0])
            train_count += 1

        # lr decay
        scheduler.step()
    torch.save(Net.state_dict(), save_path + f"epoch{EPOCH}.prm")
    print("TRAINING COMPLETE")

def Evaluate_Networks(Net, name, alg_dict, batch_size=100):
    save_path = f"./new_result/{name}/"
    data_size = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net.load_state_dict(torch.load(save_path + "epoch100.prm"), strict=False)
    Net = Net.to(device).eval()
    
    #data
    data, label = dp.load_numpy_data(alg_dict, train=False)
    test_data = dp.create_data_loader(data, label, train=False)
    del data, label
    
    # Test
    ys = []
    ypreds = []
    for X, Y in tqdm.tqdm(test_data):
        X = X.to(device)
        Y = Y.to(device)

        with torch.no_grad():
            # Value, Indices >> Get Indices
            _, y_pred = Net(X).max(1)
            ys.append(Y.max(1)[1])
            ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)

    count = 0
    for key, value in alg_dict.items():
        if(value):
            acc = (ys[data_size*count:data_size*(count+1)] == ypreds[data_size *count:data_size*(count+1)]).float().sum() / data_size
            print(key, acc.item())
            count += 1

    acc = (ys == ypreds).float().sum() / len(ys)
    print('Total AVG : ', acc.item())
    
    return ypreds

Algs = {
    # Binary Classfication
    "COVER": 1,
    "1LSB": 1,
    "PVD": 1,
    "WOW(fixed key)": 1,
    "S-UNIWARD(fixed key)": 1,
    }
model = "Pelee"

Net = importlib.import_module("Nets."+model).Model(num_classes=2)
name = dp.deal_name(Algs, model)

#Train_Networks(Net, name, Algs)
#da.data_analyzing(ypreds, lens=sum(list(Algs.values())))