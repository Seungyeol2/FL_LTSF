# -*- coding: utf-8 -*-
"""
created by Seungyeol2
Federated Learning with Client Trend

"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
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
from collections import defaultdict

sys.path.append('../')
from utils.misc import args_parser, average_weights
from utils.misc import get_data, process_isolated 
from utils.misc import time_slide_df, restart_index
from utils.misc import decompose_data, get_trend_labels, average_tensors, get_cluster_id, local_adaptation
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

    # Decompose Traffic Data
    decomposed_trend, decomposed_seasonal = decompose_data(normalized_df)
    trend_labels, n_trend_clusters = get_cluster_id(decomposed_trend, args.n_cluster_t)
    seasonal_labels, n_seasonal_clusters = get_cluster_id(decomposed_seasonal, args.n_cluster_s)
    #print("Cell 8319 trend_lables: ", trend_labels[8319])
    #print("Cell 8319 seasonal_labels: ", seasonal_labels[8319])

    global_model = DLinear.Model(configs).to(device)
    global_model.train()
    global_weights = global_model.state_dict()
    #global_up_trend_model = DLinear.Model(configs).to(device)
    #global_down_trend_model = DLinear.Model(configs).to(device)
    

    # Save their state_dicts
    #global_up_trend_model_dict = global_up_trend_model.state_dict()
    #global_down_trend_model_dict = global_down_trend_model.state_dict()

    print("Global Model: ", global_model )

    train_x, train_y, train_date = time_slide_df(train_df, configs.seq_len, configs.pred_len)
    test_x, test_y, test_date = time_slide_df(test_df, configs.seq_len, configs.pred_len)

    best_val_loss = None
    val_loss = []
    val_acc = []
    cell_loss = {}
    loss_hist = []

    global_weights = global_model.state_dict()
    trend_cluster_weights = defaultdict(lambda: global_weights['Linear_Trend.weight'])
    seasonal_cluster_weights = defaultdict(lambda: global_weights['Linear_Seasonal.weight'])

    # 1. 클러스터별 손실 초기화
    cluster_losses = {cluster_id: 0 for cluster_id in trend_labels.unique()}
    cluster_counts = {cluster_id: 0 for cluster_id in trend_labels.unique()}

    print("____________Training Start____________")
    for epoch in tqdm.tqdm(range(args.epochs)):
        m = max(int(args.frac * args.bs), 1)
        local_weights, local_losses = [], []
        local_trend_weights, local_seasonal_weights = defaultdict(list), defaultdict(list)

        print(f'\n | Global Training Round: {epoch+1} |\n')
        global_model.train()

        # 이전 라운드의 각 클러스터별 평균 손실 출력
        if epoch > 0:
            for cluster_id, total_loss in cluster_losses.items():
                avg_loss = total_loss / (cluster_counts[cluster_id] if cluster_counts[cluster_id] != 0 else 1)  # 분모가 0이 되는 것을 방지
                print(f"Cluster {cluster_id} Average Loss in Round {epoch}: {avg_loss:.4f}")

        # 1. 클러스터별 손실 업데이트
        for cell, loss in cell_loss.items():
            cluster_id = trend_labels[cell]
            cluster_losses[cluster_id] += loss
            cluster_counts[cluster_id] += 1

        all_clusters_have_values = all(v != 0 for v in cluster_losses.values())
        if all_clusters_have_values:
            # 클러스터별 손실을 기반으로 샘플링 확률 계산
            total_loss = sum(cluster_losses.values())
            sampling_probabilities = {cluster_id: loss / total_loss for cluster_id, loss in cluster_losses.items()}

            sampled_cells = []
            cluster_sampling_counts = {cluster_id: 0 for cluster_id in cluster_losses.keys()}  # 클러스터별 샘플링 횟수를 저장

            while len(sampled_cells) < 10:
                # 확률에 따라 클러스터 하나를 선택합니다.
                selected_cluster = np.random.choice(list(cluster_losses.keys()), p=list(sampling_probabilities.values()))
                # 선택된 클러스터에서의 셀들을 가져옵니다.
                cells_in_cluster = [cell for cell in selected_cells if trend_labels[cell] == selected_cluster]
                
                # 선택된 클러스터의 셀 중에서 아직 샘플링되지 않은 셀을 랜덤하게 하나 선택합니다.
                available_cells = [cell for cell in cells_in_cluster if cell not in sampled_cells]
                
                if available_cells:  # 샘플링할 수 있는 셀이 있을 때만 샘플링
                    new_cell = random.choice(available_cells)
                    sampled_cells.append(new_cell)
                    cluster_sampling_counts[selected_cluster] += 1

            # 샘플링 정보 출력
            for cluster_id, count in cluster_sampling_counts.items():
                print(f"Cluster {cluster_id} sampled {count} times.")
        else:
            # 랜덤 샘플링
            sampled_cells = random.sample(selected_cells, m)

        print(f"Round {epoch+1} sampled_cells {sampled_cells} n_sampled_cells {len(sampled_cells)}")
        #cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)
        
        for cell in sampled_cells:
            #cell_train, cell_test = train[cell], test[cell]
            cell_train_x, cell_train_y = train_x[cell], train_y[cell]
            cell_test_x, cell_test_y = test_x[cell], test_y[cell]

            local_model = LocalUpdate(args, cell_train_x, cell_train_y, cell_test_x, cell_test_y)

            # Local Model Update (Global Model -> Local Model)
            current_weights = global_model.state_dict()
            current_weights['Linear_Trend.weight'] = trend_cluster_weights[trend_labels[cell]]
            current_weights['Linear_Seasonal.weight'] = seasonal_cluster_weights[seasonal_labels[cell]]

            global_model.load_state_dict(current_weights)
            global_model.train()

            w, loss, epoch_loss = local_model.update_weights_new(model=copy.deepcopy(global_model),
                                                             global_round=epoch)
            local_trend_weights[trend_labels[cell]].append(copy.deepcopy(w['Linear_Trend.weight']))
            local_seasonal_weights[seasonal_labels[cell]].append(copy.deepcopy(w['Linear_Seasonal.weight']))

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            cell_loss[cell] = loss  # 클라이언트별 손실 업데이트

        loss_hist.append(sum(cell_loss)/len(cell_loss))
        
        # Calculate and print the average loss for this epoch
        avg_loss = sum(local_losses) / len(local_losses)
        print(f"Average loss for epoch {epoch+1}: {avg_loss}")
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
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        for id in local_trend_weights.keys():
            trend_cluster_weights[id] = average_tensors(local_trend_weights[id])

        for id in local_seasonal_weights.keys():
            seasonal_cluster_weights[id] = average_tensors(local_seasonal_weights[id])


    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    global_model.load_state_dict(global_weights)

    for cell in selected_cells:
        #cell_train_x, cell_train_y = train_x[cell], train_y[cell]
        cell_test_x, cell_test_y = test_x[cell], test_y[cell]
        # Choose the appropriate global model based on the trend of the cell
        # 복사본을 만들어서 수정
        new_state_dict = global_model.state_dict().copy()
        new_state_dict['Linear_Trend.weight'] = trend_cluster_weights[trend_labels[cell]]
        new_state_dict['Linear_Seasonal.weight'] = seasonal_cluster_weights[seasonal_labels[cell]]
        # 수정한 state_dict를 모델에 로드
        global_model.load_state_dict(new_state_dict)

        #Local adaptation
        adapted_model = local_adaptation(args, global_model, cell_train_x, cell_train_y)

        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference_new(args, adapted_model, cell_test_x, cell_test_y)
        print(f'Cluster:{trend_labels[cell]} Cell:{cell} MSE:{test_mse:.4f}')
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)
        # Append the current predictions and truths to the dataframes

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}, Seed: {}, n_cluster_t: {}, n_cluster_s: {}'.format(args.file, args.type, mse, mae,
                                                                                     nrmse,args.seed, n_trend_clusters, n_seasonal_clusters))
