# -*- coding: utf-8 -*-
import os
import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


def data_processing(alg_dict, mode='ori', batch_size=60, train=True, labelSmoothing=False):
    # Training Constants
    EPS = 0.1 if labelSmoothing else 0
    DATASIZE = 30000 if train else 10000
    NUMCLASSES = sum(list(map(int, alg_dict.values())))
    TRCODE = "TR" if train else "TE"
    PATH = f"D:/steganography/numpy image/{mode}"  # 'ori' or 'hier'

    # Data Holder
    data = np.zeros([DATASIZE*NUMCLASSES, 1, 256, 256], dtype=np.uint8)

    # Setting Data
    count = 0
    for key, value in alg_dict.items():
        if(value):
            if(key == "COVER"):
                loadedData = np.load(
                    f"{PATH}/WOW(fixed key)/{TRCODE}(bpp0).npy")
            else:
                loadedData = np.load(f"{PATH}/{key}/{TRCODE}(bpp40).npy")
            data[DATASIZE*count:DATASIZE *
                 (count+1), 0, :, :] = loadedData[:, :, :, 0]
            count += 1
            print(key, "done")
            del loadedData

    # Setting Label
    label = np.zeros([data.shape[0], NUMCLASSES])
    if labelSmoothing:
        label.fill(EPS/(NUMCLASSES - 1))
    for i in range(NUMCLASSES):
        label[int(data.shape[0]/NUMCLASSES) *
              i:int(data.shape[0]/NUMCLASSES)*(i+1), i] = 1 - EPS

    # Create Data Loader
    data = torch.tensor(data, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long)
    ds = TensorDataset(data, label)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=train)

    del data, label

    return data_loader


def Train_Networks(Net, name, alg_dict, batch_size=60, labelSmoothing=False):
    # Make folder that save results
    if not os.path.exists(f"./results/{name}"):
        os.makedirs(f"./results/{name}")

    # Data_processing
    train_data = data_processing(
        alg_dict=alg_dict, batch_size=batch_size, train=True, labelSmoothing=labelSmoothing)

    # Network Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = Net.to(device)
    lr = 0.001
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50000, 0.9)

    # EPOCH
    EPOCH = 100

    # Training Session
    train_count = 0
    for epo in range(EPOCH):
        for X, Y in train_data:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            y_pred = Net(X)
            loss = loss_fn(y_pred, torch.max(Y, 1)[1])
            loss.backward()
            optimizer.step()

            # Predict State
            if(train_count % 50 == 0):
                print(
                    f"Epoch: {epo}/100({epo/EPOCH*100}%), loss: {loss.item()}")
                print("Prediction : ", y_pred[0])

            # Save Parameters
            if(train_count % 50000 == 0):
                print('Paramter Saved')
                torch.save(Net.state_dict(), "./results/%s/params" %
                           name+str(train_count)+".prm")
            train_count += 1
        # lr decay
        scheduler.step()
    torch.save(Net.state_dict(), "./results/%s/params_last.prm")
    print("TRAINING COMPLETE")


def Evaluate_Networks(Net, name, alg_dict, batch_size=200, labelSmoothing=False):
    data_size = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net.load_state_dict(torch.load(
        "./results/%s/last.prm" % name), strict=False)
    Net = Net.to(device).eval()

    # Data_processing
    test_data = data_processing(
        alg_dict=alg_dict, batch_size=batch_size, train=False, labelSmoothing=labelSmoothing)

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
            acc = (ys[data_size*count:data_size*(count+1)] == ypreds[data_size *
                                                                     count:data_size*(count+1)]).float().sum() / data_size
            print(key, acc.item())
            count += 1

    acc = (ys == ypreds).float().sum() / len(ys)
    print('Total AVG : ', acc.item())


Algs = {
    # Binary Classfication
    "COVER": 1,
    "1LSB": 0,
    "PVD": 0,
    "WOW(fixed key)": 0,
    "S-UNIWARD(fixed key)": 0,

    # Hier Strc
    "ORIGIN": 0,
    "STEGO": 0,  # s1
    "HANDY": 0,
    "EDGE": 0,  # s2
    "LSB": 0,
    "PVD": 0,  # s3
    "WOW": 0,
    "UNIWARD": 0,  # s4
}

#import mdrl18 as model

Net = model.Model(2)
name = "drl_ch"
Train_Networks(Net, name, Algs, batch_size=60)
Evaluate_Networks(Net, name, Algs, batch_size=200)
