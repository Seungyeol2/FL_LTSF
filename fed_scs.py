# -*- coding: utf-8 -*-
"""
created by Seungyeol2

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
from sklearn import metrics
from scipy.spatial.distance import pdist

sys.path.append('../')
from utils.misc import args_parser, average_weights
from utils.misc import get_data, process_isolated 
from utils.misc import time_slide_df, restart_index
from utils.misc import cell_selection, jfi, cv, select_cells_for_round

from utils.models import LSTM
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear

from utils.fed_update import LocalUpdate, test_inference, test_inference_new
from sklearn import metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_stat_mean(d):
    d = d.iloc[:-24 * args.test_days, :]
    df_avg = d.groupby([d.index.isocalendar().week, d.index.hour]).mean().reset_index().iloc[:, 2:]
    df_avg = (df_avg - df_avg.mean()) / df_avg.std()
    return copy.deepcopy(df_avg.T)

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)

    # index clean
    df = data.reset_index()
    df = df.rename(columns={'index': 'date_time'})

    # train / test 
    train_df = df.iloc[:-24*7].copy()
    test_df = df.iloc[-24*7:].copy()
    test_df = restart_index(test_df)

    device = 'cuda' if args.gpu else 'cpu'
    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
                                                                   args.seed)
    log_id = args.directory + parameter_list
    train, val, test = process_isolated(args, data)

    # get the statistical mean of the traffic data
    df_stats = get_stat_mean(data)
    #print("df_stats :", df_stats.columns)
    #print(df_stats)
    data_dist = pdist(df_stats.values)
    data_jfi = jfi(np.array(data_dist))
    data_cv = cv(np.array(data_dist))
    print('jfi: {:.4f}, cv: {:.4f}'.format(data_jfi, data_cv))

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

    # global_model = LSTM(args).to(device)
    # global_model.train()
    # print(global_model)
    # global_weights = global_model.state_dict()

    train_x, train_y, train_date = time_slide_df(train_df, configs.seq_len, configs.pred_len)
    test_x, test_y, test_date = time_slide_df(test_df, configs.seq_len, configs.pred_len)

    #train_ds = Data(train_x, train_y)
    #test_ds = Data(test_x, test_y)

    best_val_loss = None
    val_loss = []
    val_acc = []
    cell_loss = []
    loss_hist = []

    print("____________Training Start____________")

    threshold = int(args.epochs * 0.3)  # 30% of total epochs

    for epoch in tqdm.tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        global_model.train()

        #m = max(int(args.frac * args.bs), 1)
        #cell_idx = random.sample(selected_cells, m)
        '''
        # If epoch is within the first 30% of total epochs, use random sampling.
        if epoch < threshold:
            cell_idx = random.sample(selected_cells, m)
        # For the remaining 70%, use a different sampling method 
        else:
            # cell_idx = your_other_sampling_method()
        '''

        cell_idx = select_cells_for_round(df_stats, selected_cells)
        print("cell_idx: ", cell_idx)
        print(len(cell_idx))
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

        # Update global model
        global_weights = average_weights(local_weights)
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
    print('FedSCS File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae,
                                                                                     nrmse))
