import json
import os.path
import pickle
import time
import random

import numpy as np
import torch
import pandas as pd
import geopandas as gpd
from multiprocessing import Pool

from preprocess import plots_tdrive, plot_one_traj
from utils import *
from pkdd_evaluate import novelty_score

gen_line_col = 'centroids'

params_scale = 1000 # manually read from the params.bin file first
params_x_min = 387958.6103164605
params_x_max = 527915.2735903524
params_y_min = 4370037.0096734855
params_y_max = 4499970.942369499
params_offset = 1000
angle_thres = 30
precede_segment_thres = 111 / params_scale

# def preprocess_generated_tdrive(line_str):
#     print(line_str)
#     crs = "EPSG:4326"
#     local_crs = "EPSG:4796"
#     c = np.array(eval(line_str))
#     points_gpd = gpd.GeoSeries(gpd.points_from_xy(c[:, 0], c[:, 1]), crs=crs)
#     points_gpd = points_gpd.to_crs(local_crs)
#     post_traj_list = [[p.x, p.y] for p in points_gpd]
#     return np.array(post_traj_list)

def preprocess_generated_tdrive(line_str):
    return np.array(eval(line_str))

def compute_segment_length_angle(one_traj):
    if one_traj.shape[0] < 3: return None, None
    one_traj[:, 0] = (one_traj[:, 0] - params_x_min + params_offset) / params_scale
    one_traj[:, 1] = (one_traj[:, 1] - params_y_min + params_offset) / params_scale
    xy = torch.tensor(one_traj).unsqueeze(dim=0)
    batch_orig_angles = compute_angle(xy)
    batch_orig_segment_length = compute_segment_length(xy)
    # print(one_traj.shape, batch_orig_segment_length.shape, batch_orig_angles.shape, '\n')
    return batch_orig_segment_length, batch_orig_angles

def spatial_validity_score_tdrive(tmp_list):
    spatial_validity_mask = []
    for batch_segment_length, batch_angles in tmp_list:
        mask = (batch_angles[:, :-1] < angle_thres) & (batch_angles[:, 1:] < angle_thres)
        # print(batch_angles)
        # print(mask)
        spatial_validity_mask.extend(list(mask[0]))
    validity_score = len(np.where(spatial_validity_mask)[0]) / len(spatial_validity_mask)
    return validity_score

if __name__=="__main__":

    n_jobs = 20
    n_subset = 50000
    model_name = 'llm'
    # is_test = True
    is_test = False
    # data_name = 'pkdd'
    data_name = 'tdrive'
    # data_root = '/Users/lzhang760/Desktop/TrajGen_llm_journal'
    data_root = '/media/liming/Liming1/TrajGen'
    data_log_folder = '{}/logs/{}'.format(data_root, data_name)

    if not os.path.isdir(data_log_folder):
        os.makedirs(data_log_folder)
    # generated_data_file = f'{data_root}/generated_traj_50000_beam_search_with_sampling.csv'
    generated_data_file = f'{data_root}/dataset/{model_name}_{data_name}_data/generated_traj_50000_normal_samling.csv'

    post_traj_file = '{}/dataset/{}/{}_post_traj_file.bin'.format(data_root, data_name, data_name)
    train_traj_file = '{}/dataset/{}/{}_training_traj_file.bin'.format(data_root, data_name, data_name)
    test_traj_file = '{}/dataset/{}/{}_testing_traj_file.bin'.format(data_root, data_name, data_name)

    with open(post_traj_file, 'rb') as f:
        post_traj_list = pickle.load(f)
        if is_test:
            post_traj_list = post_traj_list[:10]
        else:
            post_traj_list = random.sample(post_traj_list, n_subset)
        f.close()
        print('*' * 10, 'load a saved trajectory file: {}'.format(post_traj_file))
        print('original trajectory data\n', len(post_traj_list))

    random_sampling = False

    file_prefix = '{}_angle{}_precedSegment{}_{}_'.format(
        model_name, angle_thres, precede_segment_thres, data_name)

    ### get metrics and plots
    x_min = 0
    x_max = (params_x_max - params_x_min + params_offset) / params_scale
    x_max = int(x_max)
    y_min = 0
    y_max = (params_y_max - params_y_min + params_offset) / params_scale
    y_max = int(y_max)
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)

    n_per_bin = 10
    x_bins_ = x_max * n_per_bin
    y_bins_ = y_max * n_per_bin
    x_tick_range = [60 * n_per_bin, 80 * n_per_bin]
    y_tick_range = [440 * n_per_bin, 470 * n_per_bin]

    print('x_range, y_range, x_bins_, y_bins_', x_range, y_range, x_bins_, y_bins_)
    save_file = '{}/{}evaluation_journal.bin'.format(data_log_folder, file_prefix)

    ### read the other dataset
    print('*' * 10, 'read another generated dataset...')
    print(generated_data_file)
    if is_test:
        data = pd.read_csv(generated_data_file, nrows=10)
    else:
        data = pd.read_csv(generated_data_file)
    # data.columns = [x if x != 'centroids' else 'POLYLINE' for x in data.columns]
    print('generated trajectory data\n', len(data))

    print('*' * 10, 'read tokens sequences of real LLM dataset...')
    real_data_file = f'{data_root}/dataset/{model_name}_{data_name}_data/data_centroids.csv'
    if is_test:
        real_data = pd.read_csv(real_data_file, nrows=10)
    else:
        real_data = pd.read_csv(real_data_file)

    novelty_score = novelty_score(real_data, data)
    print('generated Novelty score : {}'.format(novelty_score))

    # print('-'*8, 'preprocess all generated traj')
    # pool = Pool(n_jobs)
    # t0 = time.time()
    # tmp_list_preprocess_gen = pool.map(preprocess_generated_tdrive, [data.loc[ind, gen_line_col] for ind in range(len(data))])
    # pool.close()
    # print('preprocess gen data use time:', time.time() - t0)
    # post_gen_traj_list = [x for x in tmp_list_preprocess_gen if x.shape[0] > 0]
    # post_gen_traj_list = random.sample(post_gen_traj_list, n_subset)
    #
    # ### plot real traj
    # print('-'*8, 'plot and analysis real data')
    # plots_tdrive(post_traj_list, data_log_folder=data_log_folder, file_prefix=f'{data_name}_spatial_validity_real')
    #
    # print('-'*8, 'analyze spatial validity of real data')
    # pool = Pool(n_jobs)
    # t0 = time.time()
    # tmp_list_orig_pool = pool.map(compute_segment_length_angle, post_traj_list)
    # tmp_list_orig = []
    # for x in tmp_list_orig_pool:
    #     if x[0] is not None:
    #         tmp_list_orig.append(x)
    # pool.close()
    # print('use time:', time.time() - t0)
    #
    # orig_validity_score = spatial_validity_score_tdrive(tmp_list_orig)
    # print('original spatial validity score : {}'.format(orig_validity_score))
    #
    # print('-'*8, 'preprocess and plot one generated traj')
    # one_gen_traj = preprocess_generated_tdrive(data.loc[0, gen_line_col])
    # plot_one_traj(one_gen_traj[:, 0], one_gen_traj[:, 1], doNote=True,
    #               sample_traj_file=f'{data_log_folder}/{data_name}_generated_plot_one_traj.png',
    #               x_range=x_range,
    #               y_range=y_range
    #               )
    #
    # print('-' * 8, 'plot and analysis generated traj')
    # plots_tdrive(post_gen_traj_list, data_log_folder=data_log_folder, file_prefix=f'{data_name}_spatial_validity_llm_gen')
    #
    # print('-' * 8, 'analyze spatial validity of generated traj')
    # pool = Pool(n_jobs)
    # t0 = time.time()
    # tmp_list_gen_pool = pool.map(compute_segment_length_angle, post_gen_traj_list)
    # tmp_list_gen = []
    # for x in tmp_list_gen_pool:
    #     if x[0] is not None:
    #         tmp_list_gen.append(x)
    # pool.close()
    # print('use time:', time.time() - t0)
    #
    # gen_validity_score = spatial_validity_score_tdrive(tmp_list_gen)
    # print('generated spatial validity score : {}'.format(gen_validity_score))
    #
    # print('-'*8, 'analyze the MMD score')
    #
    # orig_angles = np.concatenate([x[1] for x in tmp_list_orig], axis=1).reshape(-1, 1)
    # recon_angles = np.concatenate([x[1] for x in tmp_list_gen], axis=1).reshape(-1, 1)
    # distribution_score = MMD(orig_angles[:n_subset], recon_angles[:n_subset])
    # print('angle distribution score: {:.4f}'.format(distribution_score))
    #
    # orig_segment_length = np.concatenate([x[0] for x in tmp_list_orig], axis=1).reshape(-1, 1)
    # recon_segment_length = np.concatenate([x[0] for x in tmp_list_gen], axis=1).reshape(-1, 1)
    # distribution_score = MMD(orig_segment_length[:n_subset], recon_segment_length[:n_subset])
    # print('segment length distribution score: {:.4f}'.format(distribution_score))
    #
    # orig_total_length = np.array([np.sum(x[0]) for x in tmp_list_orig]).reshape(-1, 1)
    # recon_total_length = np.array([np.sum(x[0]) for x in tmp_list_gen]).reshape(-1, 1)
    # distribution_score = MMD(orig_total_length[:n_subset], recon_total_length[:n_subset])
    # print('total length distribution score: {:.4f}'.format(distribution_score))
    #
    # orig_trajs = np.concatenate(post_traj_list, axis=0)
    # recon_trajs = np.concatenate(post_gen_traj_list, axis=0)
    #
    # # distribution_score = MMD(orig_trajs, recon_trajs)
    # # print('2d point distribution score: {:.4f}'.format(distribution_score))
    #
    # orig_2Dhist = compute_2Dhist_numpy(orig_trajs, x_range, y_range, x_bins_, y_bins_)
    # recon_2Dhist = compute_2Dhist_numpy(recon_trajs, x_range, y_range, x_bins_, y_bins_)
    # distribution_score = MMD(orig_2Dhist, orig_2Dhist)
    # print('2d point histgram distribution score: {:.4f}'.format(distribution_score))
