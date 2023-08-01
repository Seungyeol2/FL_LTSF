# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: fed_update.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-01-13 (YYYY-MM-DD)
-----------------------------------------------
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import metrics

torch.manual_seed(2020)
np.random.seed(2020)

class Data(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class LocalUpdate(object):
    def __init__(self, args, train_x, train_y, test_x, test_y):
        self.args = args
        self.train_loader = self.process_data(train_x, train_y)
        self.test_loader = self.process_data(test_x, test_y)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.MSELoss().to(self.device)

    def process_data(self, x,y):
        ds = Data(x, y)
        if self.args.fedsgd == 1:
            loader = DataLoader(ds, shuffle=True, batch_size=len(ds))
        else:
            #print("batch size :", 32)
            loader = DataLoader(ds, shuffle=True, batch_size=32)
        return loader

    def update_weights_new(self, model, global_round):
        model.train()
        epoch_loss = []
        lr = self.args.lr
        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for iter in range(self.args.local_epoch):
            batch_loss = []

            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Move data and target to GPU if available
                data = data.float().to(self.device)
                target = target.float().to(self.device)

                model.zero_grad()
                pred = model(data)
                loss = self.criterion(pred, target.unsqueeze(-1))
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss)/len(epoch_loss), epoch_loss

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []
        lr = self.args.lr

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for iter in range(self.args.local_epoch):
            batch_loss = []

            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                model.zero_grad()
                pred = model(xc, xp)

                loss = self.criterion(y, pred)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss)/len(epoch_loss), epoch_loss

def test_inference_new(args, model, x, y):
    model.eval()
    loss, mse = 0.0, 0.0
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.MSELoss().to(device)
    test_loss_list = []

    # data to dl
    ds = Data(x, y)
    data_loader = DataLoader(ds, shuffle=False, batch_size=args.local_bs)
    #data_loader = DataLoader(list(zip(*dataset)), batch_size=args.local_bs, shuffle=False)
    pred_list, truth_list = [], []

    with torch.no_grad():
        for data, target in data_loader:
            # Move data and target to GPU if available
            data = data.float().to(device)
            target = target.float().to(device)

            pred = model(data)
            batch_loss = criterion(pred, target.unsqueeze(-1))
            
            loss += batch_loss.item()
            mse += batch_loss.item()

            #batch_mse = torch.mean((pred - target) ** 2)
            #mse += batch_mse.item()

            pred_list.append(pred.detach().cpu())
            truth_list.append(target.detach().cpu())
            
    final_prediction = np.concatenate(pred_list).ravel()
    final_truth = np.concatenate(truth_list).ravel()
    nrmse= (metrics.mean_squared_error(final_truth, final_prediction) ** 0.5) / (max(final_truth) - min(final_truth))
    avg_loss = loss / len(data_loader)
    avg_mse = mse / len(data_loader)

    return avg_loss, avg_mse, nrmse, final_prediction, final_truth


def test_inference(args, model, dataset):
    model.eval()
    loss, mse = 0.0, 0.0
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.MSELoss().to(device)
    data_loader = DataLoader(list(zip(*dataset)), batch_size=args.local_bs, shuffle=False)
    pred_list, truth_list = [], []

    with torch.no_grad():
        for batch_idx, (xc, xp, y) in enumerate(data_loader):
            xc, xp = xc.float().to(device), xp.float().to(device)
            y = y.float().to(device)
            pred = model(xc, xp)

            batch_loss = criterion(y, pred)
            loss += batch_loss.item()

            batch_mse = torch.mean((pred - y) ** 2)
            mse += batch_mse.item()

            pred_list.append(pred.detach().cpu())
            truth_list.append(y.detach().cpu())

    final_prediction = np.concatenate(pred_list).ravel()
    final_truth = np.concatenate(truth_list).ravel()
    nrmse= (metrics.mean_squared_error(final_truth, final_prediction) ** 0.5) / (max(final_truth) - min(final_truth))
    avg_loss = loss / len(data_loader)
    avg_mse = mse / len(data_loader)

    return avg_loss, avg_mse, nrmse, final_prediction, final_truth