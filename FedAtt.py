# -*- coding: utf-8 -*-
"""
created by Seungyeol2
Second Commit Test
"""
import numpy as np
import h5py
import tqdm
import copy
import torch
import pandas as pd
import sys
import random
from argparse import Namespace
import matplotlib.pyplot as plt
import csv

sys.path.append('../')
from utils.misc import args_parser, average_weights_att
from utils.misc import get_data, process_isolated 
from utils.misc import time_slide_df, restart_index
from utils.models import LSTM
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear

from utils.fed_update import LocalUpdate, test_inference, test_inference_new
from sklearn import metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #data, _, selected_cells, mean, std, _, _ = get_data(args)
    normalized_df, selected_cells_df, selected_cells, mean, std, cell_lng, cell_lat, ori_df, normalized_ori_df = get_data(args)

    # index clean
    df = normalized_df.reset_index()
    df = df.rename(columns={'index': 'date_time'})

    # train / test 
    train_df = df.iloc[:-24*7].copy()
    test_df = df.iloc[-24*7:].copy()
    test_df = restart_index(test_df)

    device = 'cuda' if args.gpu else 'cpu'
    # print(selected_cells)

    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
                                                                   args.seed)
    log_id = args.directory + parameter_list
    train, val, test = process_isolated(args, normalized_df)


    configs = Namespace(
        seq_len = 72,
        pred_len = 1,
        kernel_size = 25,
        individual = False,
        enc_in = 1
    )
    global_model = DLinear.Model(configs).to(device)
    global_model.train()
    global_weights = global_model.state_dict()
    print("Global Model: ", global_model )

    last_train_data = df.iloc[-configs.seq_len:].copy()
    extended_test_df = pd.concat([last_train_data, test_df], axis=0).reset_index(drop=True)

    train_x, train_y, train_date = time_slide_df(train_df, configs.seq_len, configs.pred_len)
    test_x, test_y, test_date = time_slide_df(extended_test_df, configs.seq_len, configs.pred_len)

    #train_ds = Data(train_x, train_y)
    #test_ds = Data(test_x, test_y)

    best_val_loss = None
    val_loss = []
    val_acc = []
    cell_loss = []
    loss_hist = []

    print("____________Training Start____________")
    for epoch in tqdm.tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        global_model.train()

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)
        
        for cell in cell_idx:
            #cell_train, cell_test = train[cell], test[cell]
            cell_train_x, cell_train_y = train_x[cell], train_y[cell]
            cell_test_x, cell_test_y = test_x[cell], test_y[cell]

            local_model = LocalUpdate(args, cell_train_x, cell_train_y, cell_test_x, cell_test_y)

            global_model.load_state_dict(global_weights)
            global_model.train()

            w, loss, epoch_loss = local_model.update_weights_new(model=copy.deepcopy(global_model),
                                                             global_round=epoch)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            cell_loss.append(loss)

        loss_hist.append(sum(cell_loss)/len(cell_loss))
        '''
        plt.figure(figsize=(10, 6))
        plt.plot(loss_hist, '-o', label='Global Model Loss')
        plt.title('Global Model Loss Over Rounds')
        plt.xlabel('Rounds')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('global_model_loss.png')  # 파일 저장
        plt.show()

        # 손실 데이터를 CSV 파일로 저장
        with open('[FedAvg]global_model_loss.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Loss"])  # 컬럼 헤더 작성
            for i, loss in enumerate(loss_hist):
                writer.writerow([i, loss])
        '''
        # Update global model
        #global_weights = average_weights(local_weights)
        global_weights = average_weights_att(local_weights, global_weights, args.epsilon)

        global_model.load_state_dict(global_weights)

    for index, value in enumerate(loss_hist):
        print(f"Epoch: {index}, Loss: {value}")
    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    global_model.load_state_dict(global_weights)

    for cell in selected_cells:
        cell_test_x, cell_test_y = test_x[cell], test_y[cell]
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference_new(args, global_model, cell_test_x, cell_test_y)
        print(f'Cell {cell} MSE {test_mse:.4f}')
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('FedAtt File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}, Seed: {}'.format(args.file, args.type, mse, mae,
                                                                                     nrmse, args.seed))
    df_pred.to_csv('[FedAtt] File:{:}_Type:{:}_MSE:{:.4f}_MAE:{:.4f}_predictions.csv'.format(args.file, args.type, mse, mae), index=False)
    df_truth.to_csv('[FedAtt] File:{:}_Type:{:}_MSE:{:.4f}_MAE:{:.4f}_truth.csv'.format(args.file, args.type, mse, mae), index=False)