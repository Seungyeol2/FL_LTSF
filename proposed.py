# -*- coding: utf-8 -*-
"""
created by Seungyeol2
Federated Learning with Client Trend

"""
import warnings
import re

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
import matplotlib.pyplot as plt

sys.path.append('../')
from utils.misc import args_parser, average_weights
from utils.misc import get_data, process_isolated 
from utils.misc import time_slide_df, restart_index
from utils.misc import decompose_data, get_trend_labels, average_tensors, get_cluster_id, local_adaptation
from utils.models import LSTM
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear

from utils.fed_update import LocalUpdate, test_inference, test_inference_new
from sklearn import metrics
import pickle

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def sizeof_model(model_dict):
    """Returns the size of the model in bytes."""
    return len(pickle.dumps(model_dict))

def transform_string(s):
    # 첫 문자를 대문자로 변경
    if s and s[0].isalpha():
        s = s[0].upper() + s[1:]
    # .h5 삭제
    s = s.replace('.h5', '')
    return s


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
    last_train_data = df.iloc[-configs.seq_len:].copy()
    extended_test_df = pd.concat([last_train_data, test_df], axis=0).reset_index(drop=True)

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
    test_x, test_y, test_date = time_slide_df(extended_test_df, configs.seq_len, configs.pred_len)

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

    # 1. 각 클러스터의 최근 5번의 손실 값을 저장할 dictionary
    recent_cluster_losses = {cluster_id: [float('inf')]*15 for cluster_id in cluster_losses.keys()}

    seasonal_cluster_losses = {cluster_id: 0 for cluster_id in seasonal_labels.unique()}
    seasonal_cluster_counts = {cluster_id: 0 for cluster_id in seasonal_labels.unique()}
    recent_seasonal_cluster_losses = {cluster_id: [float('inf')]*15 for cluster_id in seasonal_cluster_losses.keys()}

    overfitting_clusters = set()  # 오버피팅 감지된 클러스터 저장
    
    # 학습 완료된 클러스터의 가중치 저장하는 딕셔너리 초기화
    frozen_trend_weights = {}
    frozen_seasonal_weights = {}
    frozen_trend_cluster_usage_counts = {}
    frozen_seasonal_cluster_usage_counts = {}

    communication_cost_list = []
    cumulative_communication_cost = [0]

    total_params = sizeof_model(global_model)
    communication_cost_baseline = total_params
    print("total parmas:",total_params)
    frozen_communication_cost = total_params * 0.5  # 예를 들어 프리징된 클러스터가 통신량의 절반만 차지한다고 가정했습니다.
    overfitting_detected_rounds = []


    print("____________Training Start____________")
    for epoch in tqdm.tqdm(range(args.epochs)):
        current_round_comm_cost = 0

        m = max(int(args.frac * args.bs), 1)
        local_weights, local_losses = [], []
        local_trend_weights, local_seasonal_weights = defaultdict(list), defaultdict(list)

        print(f'\n | Global Training Round: {epoch+1} |\n')
        global_model.train()

        for cell, loss in cell_loss.items():
            trend_cluster_id = trend_labels[cell]
            cluster_losses[trend_cluster_id] += loss
            cluster_counts[trend_cluster_id] += 1

            seasonal_cluster_id = seasonal_labels[cell]
            seasonal_cluster_losses[seasonal_cluster_id] += loss
            seasonal_cluster_counts[seasonal_cluster_id] += 1

        
        # 2. 이전 라운드의 각 클러스터별 평균 손실 출력
        for cluster_id, total_loss in cluster_losses.items():
            avg_loss = total_loss / (cluster_counts[cluster_id] if cluster_counts[cluster_id] != 0 else 1)
            print(f"Trend Cluster {cluster_id} Average Loss in Round {epoch+1}: {avg_loss:.4f}")
            # 클러스터별 오버피팅 감지
            if all([prev_loss < avg_loss for prev_loss in recent_cluster_losses[cluster_id]]):  
                if cluster_id not in frozen_trend_weights:
                    print(f"Detected overfitting for Trend Cluster {cluster_id}: Last 5 losses: {recent_cluster_losses[cluster_id]}, New loss: {avg_loss}")
                    overfitting_clusters.add("trend_" + str(cluster_id))
                    overfitting_detected_rounds.append(epoch)
                    # 해당 클러스터의 트랜드 가중치 저장
                    frozen_trend_weights[cluster_id] = global_weights['Linear_Trend.weight']
                    frozen_trend_cluster_usage_counts[cluster_id] = 0

            if epoch > 0:
                if avg_loss != 0.0:
                    recent_cluster_losses[cluster_id] = recent_cluster_losses[cluster_id][1:] + [avg_loss]  # 3. 최근 손실 값 업데이트

        for cluster_id, total_loss in seasonal_cluster_losses.items():
            avg_loss = total_loss / (seasonal_cluster_counts[cluster_id] if seasonal_cluster_counts[cluster_id] != 0 else 1)
            print(f"Seasonal Cluster {cluster_id} Average Loss in Round {epoch+1}: {avg_loss:.4f}")
            if all([prev_loss < avg_loss for prev_loss in recent_seasonal_cluster_losses[cluster_id]]):  
                if cluster_id not in frozen_seasonal_weights:
                    print(f"Detected overfitting for Seasonal Cluster {cluster_id}: Last 5 losses: {recent_seasonal_cluster_losses[cluster_id]}, New loss: {avg_loss}")
                    overfitting_clusters.add("seasonal_" + str(cluster_id))
                    overfitting_detected_rounds.append(epoch)

                    # 해당 클러스터의 시즈널 가중치 저장
                    frozen_seasonal_weights[cluster_id] = global_weights['Linear_Seasonal.weight']
                    frozen_seasonal_cluster_usage_counts[cluster_id] = 0

            if epoch > 0:
                if avg_loss != 0.0:
                    recent_seasonal_cluster_losses[cluster_id] = recent_seasonal_cluster_losses[cluster_id][1:] + [avg_loss]

        # 모든 클러스터에서 오버피팅이 감지되면 학습 중단
        total_clusters = len(cluster_losses.keys()) + len(seasonal_cluster_losses.keys())
        if len(overfitting_clusters) == total_clusters:
            print("Overfitting detected in all clusters. Stopping training.")
            break
        '''

    
        all_clusters_have_values = all(v != 0 for v in cluster_losses.values())

        if all_clusters_have_values:
            # 1. 클러스터별 손실을 합산
            total_loss = sum(cluster_losses.values()) + sum(seasonal_cluster_losses.values())
            
            # 2. 클러스터별 샘플링 확률 계산
            all_clusters_losses = {**cluster_losses, **seasonal_cluster_losses}
            sampling_probabilities = {cluster_id: loss / total_loss for cluster_id, loss in all_clusters_losses.items()}

            sampled_cells = []
            cluster_sampling_counts = {cluster_id: 0 for cluster_id in all_clusters_losses.keys()}  # 클러스터별 샘플링 횟수를 저장

            while len(sampled_cells) < 10:
                available_clusters = [cluster_id for cluster_id in all_clusters_losses.keys() if cluster_id not in overfitting_clusters]
                
                # available_clusters가 비어있는지 확인
                if not available_clusters:
                    print("All clusters are detected as overfitting. No available clusters for sampling.")
                    break
                
                available_probabilities = [sampling_probabilities[cluster_id] for cluster_id in available_clusters]
                                # 확률 정규화
                sum_probs = sum(available_probabilities)
                available_probabilities = [prob/sum_probs for prob in available_probabilities]
                # 확률에 따라 클러스터 하나를 선택합니다.
                selected_cluster = np.random.choice(available_clusters, p=available_probabilities)
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
            # 랜덤 샘플링 시 오버피팅된 클러스터에 속하는 셀은 제외
            #available_cells = [cell for cell in selected_cells if trend_labels[cell] not in overfitting_clusters]
        '''
        # 샘플링
        sampled_cells = random.sample(selected_cells, m)
        print(f"Round {epoch+1} sampled_cells {sampled_cells} n_sampled_cells {len(sampled_cells)}")

        for cell in sampled_cells:
            # 클러스터의 프리징 상태에 따라 통신량을 결정
            if "trend_" + str(trend_labels[cell]) in overfitting_clusters and "seasonal_" + str(seasonal_labels[cell]) in overfitting_clusters:
                cost = 0
            elif "trend_" + str(trend_labels[cell]) in overfitting_clusters or "seasonal_" + str(seasonal_labels[cell]) in overfitting_clusters:
                cost = frozen_communication_cost
            else:
                cost = communication_cost_baseline

            current_round_comm_cost += cost  

            #cell_train, cell_test = train[cell], test[cell]
            cell_train_x, cell_train_y = train_x[cell], train_y[cell]
            cell_test_x, cell_test_y = test_x[cell], test_y[cell]

            local_model = LocalUpdate(args, cell_train_x, cell_train_y, cell_test_x, cell_test_y)

            # Local Model Update (Global Model -> Local Model)
            # 학습 완료된 클러스터의 가중치 불러오기
            current_weights = global_model.state_dict()
            if "trend_" + str(trend_labels[cell]) in overfitting_clusters:
                current_weights['Linear_Trend.weight'] = frozen_trend_weights[trend_labels[cell]]
                frozen_trend_cluster_usage_counts[trend_labels[cell]] += 1
            else:
                current_weights['Linear_Trend.weight'] = trend_cluster_weights[trend_labels[cell]]

            if "seasonal_" + str(seasonal_labels[cell]) in overfitting_clusters:
                current_weights['Linear_Seasonal.weight'] = frozen_seasonal_weights[seasonal_labels[cell]]
                frozen_seasonal_cluster_usage_counts[seasonal_labels[cell]] += 1
            else:
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

        cumulative_value = cumulative_communication_cost[-1] + current_round_comm_cost
        cumulative_communication_cost.append(cumulative_value)
        print(f"Global round {epoch} = Cost {cumulative_value}")

        loss_hist.append(sum(cell_loss)/len(cell_loss))
        # Calculate and print the average loss for this epoch
        if local_losses:
            avg_loss = sum(local_losses) / len(local_losses)
        else:
            print("No losses to calculate average. Skipping...")
            break
        print(f"Average loss for epoch {epoch+1}: {avg_loss}")

        # Update global model
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

    
        for id in local_trend_weights.keys():
            trend_cluster_weights[id] = average_tensors(local_trend_weights[id])

        for id in local_seasonal_weights.keys():
            seasonal_cluster_weights[id] = average_tensors(local_seasonal_weights[id])

    print("\nFrozen Trend Cluster Usages:")
    for cluster_id, count in frozen_trend_cluster_usage_counts.items():
        print(f"Cluster {cluster_id} used {count} times.")
        
    print("\nFrozen Seasonal Cluster Usages:")
    for cluster_id, count in frozen_seasonal_cluster_usage_counts.items():
        print(f"Cluster {cluster_id} used {count} times.")

    # x축은 글로벌 라운드 번호
    x = list(range(len(cumulative_communication_cost)))
    plt.figure(figsize=(7, 5))

    # Baseline은 일직선이므로 (글로벌 라운드 번호 * baseline 통신량)으로 계산됩니다.
    baseline = [10 * i * communication_cost_baseline for i in x]
    plt.plot(x, cumulative_communication_cost, label='Proposed', color='red', linestyle='--')
    plt.plot(x, baseline, label='FedAvg', color='black')
    title = f'[{transform_string(args.file)}] {args.type}'

    offset_x = 25  # 이 값을 조정하여 주석의 x축 방향 위치를 조절할 수 있습니다.
    offset_y = 15  # 이 값을 조정하여 주석의 y축 방향 위치를 조절할 수 있습니다.
    # 40 round 이후의 detected rounds만 필터링합니다.
    filtered_detected_rounds = [r for r in overfitting_detected_rounds if r > 28]

    for idx, detected_round in enumerate(filtered_detected_rounds):
        y_intersect = cumulative_communication_cost[detected_round]  # 수직선과 value가 만나는 지점의 y값
        plt.axvline(x=detected_round, color='green', linestyle='--' if 'Layer Freezing' not in [l.get_label() for l in plt.gca().get_lines()] else "")
        
        if idx % 2 == 0:
            # 짝수 인덱스에 해당하는 detected_round에 대한 주석은 위쪽에 위치
            text_position = (detected_round - offset_x, y_intersect + offset_y)
        else:
            # 홀수 인덱스에 해당하는 detected_round에 대한 주석은 아래쪽에 위치
            text_position = (detected_round - offset_x, y_intersect - offset_y)

        plt.annotate('Layer Freezing', xy=(detected_round, y_intersect), xytext=text_position, 
                    arrowprops=dict(facecolor='blue', arrowstyle='->', connectionstyle="arc3,rad=-0.3"), fontsize=12, ha='center')



    first_freezing_round = overfitting_detected_rounds[0] if overfitting_detected_rounds else 0
    # 영역 채우기로 Gap 강조
    plt.fill_between(x[first_freezing_round:101], 
                 cumulative_communication_cost[first_freezing_round:101], 
                 baseline[first_freezing_round:101], 
                 color='yellow', alpha=0.5, label='Communication Cost Savings')


    plt.xlabel('Communication Round', fontsize=13, color='black')
    plt.ylabel('Cumulative Communication Cost', fontsize=13, color='black')
    plt.title(title, fontsize=17, color='black')
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'File:{args.file}_Type:{args.type}_comm_cost.png')


    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    global_model.load_state_dict(global_weights)

    for cell in selected_cells:
        cell_train_x, cell_train_y = train_x[cell], train_y[cell]
        cell_test_x, cell_test_y = test_x[cell], test_y[cell]
        # Choose the appropriate global model based on the trend of the cell
        # 복사본을 만들어서 수정
        new_state_dict = global_model.state_dict().copy()
        new_state_dict['Linear_Trend.weight'] = trend_cluster_weights[trend_labels[cell]]
        new_state_dict['Linear_Seasonal.weight'] = seasonal_cluster_weights[seasonal_labels[cell]]
        # 수정한 state_dict를 모델에 로드
        global_model.load_state_dict(new_state_dict)

        #Local adaptation
        if args.local_adapt == True:
            infer_model = local_adaptation(args, global_model, cell_train_x, cell_train_y)
        else:
            infer_model = global_model


        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference_new(args, infer_model, cell_test_x, cell_test_y)
        #print(f'Cluster:{trend_labels[cell]} Cell:{cell} MSE:{test_mse:.4f}')
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)
        # Append the current predictions and truths to the dataframes

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}, Seed: {}, n_cluster_t: {}, n_cluster_s: {}, is_local_adapt: {}'.format(args.file, args.type, mse, mae,
                                                                                     nrmse,args.seed, n_trend_clusters, n_seasonal_clusters,args.local_adapt))
    df_pred.to_csv('[proposed] File:{:}_Type:{:}_MSE:{:.4f}_MAE:{:.4f}_predictions.csv'.format(args.file, args.type, mse, mae), index=False)
    df_truth.to_csv('[proposed] File:{:}_Type:{:}_MSE:{:.4f}_MAE:{:.4f}_truth.csv'.format(args.file, args.type, mse, mae), index=False)