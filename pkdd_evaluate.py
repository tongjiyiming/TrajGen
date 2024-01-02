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

from preprocess import plots_pkdd, plot_one_traj
from utils import *
from preprocess import get_angles, get_segment_length

gen_line_col = 'centroids'

params_scale = 1000 # manually read from the params.bin file first
params_x_min = -108713.38256994488
params_x_max = 155245.71162283162
params_y_min = -286345.3700102815
params_y_max = 272791.2253923012
params_offset = 1000
angle_thres = 120
# precede_segment_thres = 166 / params_scale
precede_segment_thres = 166

def preprocess_generated_pkdd(line_str):
    crs = "EPSG:4326"
    local_crs = "EPSG:3763"
    c = np.array(eval(line_str))
    points_gpd = gpd.GeoSeries(gpd.points_from_xy(c[:, 0], c[:, 1]), crs=crs)
    points_gpd = points_gpd.to_crs(local_crs)
    # post_traj_list = [
    #     [
    #         (p.x- params_x_min + params_offset) / params_scale, (p.y - params_y_min + params_offset) / params_scale
    #     ]
    #     for p in points_gpd
    # ]
    post_traj_list = [
        [
            p.x, p.y
        ]
        for p in points_gpd
    ]
    return np.array(post_traj_list)

def postprocess_generated(one_traj):
    print('***')
    print(one_traj)
    one_traj[:, 0] = (one_traj[:, 0] - params_x_min + params_offset) / params_scale
    one_traj[:, 1] = (one_traj[:, 1] - params_y_min + params_offset) / params_scale
    print(one_traj)
    return one_traj

def compute_segment_length_angle(one_traj):
    if one_traj.shape[0] < 3: return None, None
    # one_traj[:, 0] = (one_traj[:, 0] - params_x_min + params_offset) / params_scale
    # one_traj[:, 1] = (one_traj[:, 1] - params_y_min + params_offset) / params_scale

    one_traj = torch.tensor(one_traj).unsqueeze(dim=0)
    batch_orig_angles = compute_angle(one_traj)
    batch_orig_segment_length = compute_segment_length(one_traj)

    # batch_orig_angles = get_angles(one_traj)
    # batch_orig_segment_length = get_segment_length(one_traj)

    # print(one_traj.shape, batch_orig_segment_length.shape, batch_orig_angles.shape, '\n')
    # print(batch_orig_angles)
    return batch_orig_segment_length, batch_orig_angles

def spatial_validity_score(tmp_list):
    spatial_validity_mask = []
    for batch_segment_length, batch_angles in tmp_list:
        mask = (batch_segment_length[:, :-1] > precede_segment_thres) & (batch_angles < angle_thres)
        # print(batch_angles)
        # print(mask)
        spatial_validity_mask.extend(list(mask[0]))
    validity_score = len(np.where(spatial_validity_mask)[0]) / len(spatial_validity_mask)
    return validity_score


def novelty_score(real_data, gen_data):
    real_token_list = []
    for i, row in real_data.iterrows():
        real_token_list.append(row['tokens'])
    real_token_set = set(real_token_list)

    novelty_count = 0
    for i, row in gen_data.iterrows():
        traj = eval(row['trajectory'])
        traj = str([int(x) for x in traj[1:-1]])
        if traj not in real_token_set:
            novelty_count += 1
    return novelty_count / len(gen_data)

def plots_pkdd_evaluate(post_traj_list, data_log_folder, file_prefix):
    # plot angles distribution
    angles = []
    preceding_segments = []
    succeeding_segments = []
    both_segments = []
    for traj in post_traj_list:
        one_traj_angles = get_angles(traj)
        angles.extend(list(one_traj_angles))
        segments = get_segment_length(traj)
        preceding_segments.extend(list(segments[:-1]))
        succeeding_segments.extend(list(segments[1:]))
        both_segments.extend(list(segments[:-1] + segments[1:]))
        # break
    # compute ratio
    preceding_segments = np.array(preceding_segments)
    succeeding_segments = np.array(succeeding_segments)
    both_segments = np.array(both_segments)
    angles = np.array(angles)
    dist_thres, angle_thres = 166, 150 # 60 km
    ratio = len(np.where((preceding_segments > dist_thres) \
                         & (angles > angle_thres))[0]) / len(preceding_segments)
    print('''ratio that preceding segments are larger than {}
          and angle are larger than {} is {}'''.format(dist_thres, angle_thres, ratio))
    ratio = len(np.where((preceding_segments > dist_thres) & (succeeding_segments > dist_thres) \
                         & (angles > angle_thres))[0]) / len(preceding_segments)
    print('''ratio that preceding and suceeding segments are both larger than {}
          and angle are larger than {} is {}'''.format(dist_thres, angle_thres, ratio))
    dist_thres, angle_thres = 111, 150 #  40 km/h
    ratio = len(np.where((preceding_segments > dist_thres) & (succeeding_segments > dist_thres) \
                         & (angles > angle_thres))[0]) / len(preceding_segments)
    print('''ratio that preceding and suceeding segments are both larger than {}
          and angle are larger than {} is {}'''.format(dist_thres, angle_thres, ratio))
    dist_thres, angle_thres = 111, 120 # 40 km/h
    ratio = len(np.where((preceding_segments > dist_thres) & (succeeding_segments > dist_thres) \
                         & (angles > angle_thres))[0]) / len(preceding_segments)
    print('''ratio that preceding and suceeding segments are both larger than {}
          and angle are larger than {} is {}'''.format(dist_thres, angle_thres, ratio))

    print('save validity heatmap to files in:', data_log_folder, file_prefix)
    # preceding angle v.s. angle
    h, _, _ = np.histogram2d(angles[:-1], angles[1:], bins=[900, 900], range=[[0, 180], [0, 180]])
    plt.figure(figsize=(4, 3))
    ax = plt.imshow(np.log10(h), cmap='Blues')
    plt.xticks([i*150 for i in range(7)], [str(i*30) for i in range(7)])
    plt.yticks([i*150 for i in range(7)], [str(i*30) for i in range(7)])
    plt.xlabel('Preceding angles')
    plt.ylabel('Succeeding angles')
    plt.colorbar(ax)
    plt.tight_layout()
    plt.savefig('{}/_angle_to_angle_constraint_{}.png'.format(data_log_folder, file_prefix), dpi=120)
    plt.close()

    # preceding segment v.s. angle
    print(np.isnan(angles).sum())
    print(preceding_segments.min(), preceding_segments.max())
    print(angles.min(), angles.max())
    h, _, _ = np.histogram2d(preceding_segments, angles, bins=[1500, 360], range=[[0, 1500], [0, 180]])
    plt.figure(figsize=(4, 3))
    ax = plt.imshow(h, cmap='Blues')
    # ax = plt.imshow(np.log10(h), cmap='Blues')
    plt.xticks([i*60 for i in range(7)], [str(i*30) for i in range(7)])
    plt.xlabel('Angles')
    plt.ylabel('Preceding segment lengths')
    plt.colorbar(ax)
    plt.tight_layout()
    plt.savefig('{}/_preseding_segment_to_angle_constraint_{}.png'.format(data_log_folder, file_prefix), dpi=120)
    plt.close()

    # succeeding segment v.s. angle
    h, _, _ = np.histogram2d(succeeding_segments, angles, bins=[500, 900], range=[[0, 500], [0, 180]])
    plt.figure(figsize=(4, 3))
    # ax = plt.imshow(h, cmap='Blues')
    ax = plt.imshow(np.log10(h), cmap='Blues')
    plt.xticks([i*150 for i in range(7)], [str(i*30) for i in range(7)])
    plt.xlabel('Angles')
    plt.ylabel('Succeeding segment lengths')
    plt.colorbar(ax)
    plt.tight_layout()
    plt.savefig('{}/_succeeding_segment_to_angle_constraint_{}.png'.format(data_log_folder, file_prefix), dpi=120)
    plt.close()

    # sum of both segments v.s. angle
    h, _, _ = np.histogram2d(both_segments, angles, bins=[1000, 900], range=[[0, 1000], [0, 180]])
    plt.figure(figsize=(4, 3))
    # ax = plt.imshow(h, cmap='Blues')
    ax = plt.imshow(np.log10(h), cmap='Blues')
    plt.xticks([i * 150 for i in range(7)], [str(i * 30) for i in range(7)])
    plt.xlabel('Angles')
    plt.ylabel('Sum of both segment lengths')
    plt.colorbar(ax)
    plt.tight_layout()
    plt.savefig('{}/_both_segment_to_angle_constraint_{}.png'.format(data_log_folder, file_prefix), dpi=120)
    plt.close()

    # angles distribution
    plt.hist(angles, bins=180, range=[0, 180])
    plt.title('after dist-cut angle distribution')
    plt.tight_layout()
    plt.savefig('{}/_angle_constraint_{}.png'.format(data_log_folder, file_prefix), dpi=120)
    plt.close()

if __name__=="__main__":

    n_jobs = 20
    n_subset = 50000
    model_name = 'llm'
    # is_test = True
    is_test = False
    data_name = 'pkdd'
    # data_name = 'tdrive'
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

    params_file = '{}/dataset/{}/{}_params.bin'.format(data_root, data_name, data_name)
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
        f.close()
        print('load saved preprocess params file: {}'.format(params_file))
        print(params)

    with open(post_traj_file, 'rb') as f:
        post_traj_list = pickle.load(f)
        if is_test:
            post_traj_list = post_traj_list[:10]
        else:
            post_traj_list = random.sample(post_traj_list, n_subset)
        f.close()
        print('*' * 10, 'load a saved trajectory file: {}'.format(post_traj_file))
        print('original trajectory data\n', len(post_traj_list))
        print(post_traj_list[0][:3, :])

    ### plot real traj
    print('-'*8, 'plot and analysis real data')
    plots_pkdd_evaluate(post_traj_list, data_log_folder=data_log_folder, file_prefix='pkdd_spatial_validity_real')

    traj_len = 32
    batch_size = 128
    test_batch = 128
    epochs = 100
    eval_epoches = 10
    learning_rate = 0.0002
    f_dim = 256
    z_dim = 64
    encode_dims = [48, 16]
    decode_dims = [128]
    beta = 100.0
    alpha = 1.0
    jit = None
    plot_bound = 0.01
    step = 256

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
    if is_test:
        data = pd.read_csv(generated_data_file, nrows=10)
    else:
        data = pd.read_csv(generated_data_file, nrows=n_subset)
    # data.columns = [x if x != 'centroids' else 'POLYLINE' for x in data.columns]
    print('generated trajectory data\n', len(data))
    print('-'*8, 'preprocess all generated traj')
    pool = Pool(n_jobs)
    t0 = time.time()
    tmp_list_preprocess_gen = pool.map(preprocess_generated_pkdd, [data.loc[ind, gen_line_col] for ind in range(len(data))])
    pool.close()
    print(tmp_list_preprocess_gen[0][:3, :])
    print('preprocess gen data use time:', time.time() - t0)
    post_gen_traj_list = [x for x in tmp_list_preprocess_gen if x.shape[0] > 0]
    # if is_test:
    #     post_gen_traj_list = post_gen_traj_list[:10]
    # else:
    #     post_gen_traj_list = random.sample(post_gen_traj_list, n_subset)

    print('-'*8, 'preprocess and plot one generated traj')
    one_gen_traj = preprocess_generated_pkdd(data.loc[0, gen_line_col])
    one_gen_traj[:, 0] = (one_gen_traj[:, 0] - params_x_min + params_offset) / params_scale
    one_gen_traj[:, 1] = (one_gen_traj[:, 1] - params_y_min + params_offset) / params_scale
    plot_one_traj(one_gen_traj[:, 0], one_gen_traj[:, 1], doNote=False,
                  sample_traj_file=f'{data_root}/logs/pkdd/pkdd_generated_plot_one_traj.png',
                  x_range=x_range,
                  y_range=y_range
                  )

    print('-' * 8, 'plot and analysis generated traj')
    plots_pkdd_evaluate(
        post_gen_traj_list,
        data_log_folder=data_log_folder, file_prefix='pkdd_spatial_validity_llm_gen')
    # plots_pkdd_evaluate(
    #     [postprocess_generated(x) for x in post_gen_traj_list],
    #     data_log_folder=data_log_folder, file_prefix='pkdd_spatial_validity_llm_gen')

    # print('*' * 10, 'read tokens sequences of real LLM dataset...')
    # real_data_file = f'{data_root}/dataset/{model_name}_{data_name}_data/data_centroids.csv'
    # if is_test:
    #     real_data = pd.read_csv(real_data_file, nrows=10)
    # else:
    #     real_data = pd.read_csv(real_data_file)
    #
    # novelty_score = novelty_score(real_data, data)
    # print('generated Novelty score : {}'.format(novelty_score))

    # print('-'*8, 'analyze spatial validity of real data')
    #
    # # orig_validity_mask = spatial_validity_score(post_traj_list[0])
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
    # orig_validity_score = spatial_validity_score(tmp_list_orig)
    # print('original spatial violation score : {}'.format(orig_validity_score))

    print('-' * 8, 'analyze spatial validity of generated traj')
    pool = Pool(n_jobs)
    t0 = time.time()
    tmp_list_gen_pool = pool.map(compute_segment_length_angle, post_gen_traj_list)
    tmp_list_gen = []
    for x in tmp_list_gen_pool:
        if x[0] is not None:
            tmp_list_gen.append(x)
    pool.close()
    print('use time:', time.time() - t0)

    gen_validity_score = spatial_validity_score(tmp_list_gen)
    print('generated spatial violation score : {}'.format(gen_validity_score))

    # print('-'*8, 'analyze the MMD score')
    # orig_angles = np.concatenate([x[1] for x in tmp_list_orig], axis=1).reshape(-1, 1)
    # recon_angles = np.concatenate([x[1] for x in tmp_list_gen], axis=1).reshape(-1, 1)
    # distribution_score = MMD(orig_angles, recon_angles)
    # print('angle distribution score: {:.4f}'.format(distribution_score))
    #
    # orig_segment_length = np.concatenate([x[0] for x in tmp_list_orig], axis=1).reshape(-1, 1)
    # recon_segment_length = np.concatenate([x[0] for x in tmp_list_gen], axis=1).reshape(-1, 1)
    # distribution_score = MMD(orig_segment_length, recon_segment_length)
    # print('segment length distribution score: {:.4f}'.format(distribution_score))
    #
    # orig_total_length = np.array([np.sum(x[0]) for x in tmp_list_orig]).reshape(-1, 1)
    # recon_total_length = np.array([np.sum(x[0]) for x in tmp_list_gen]).reshape(-1, 1)
    # distribution_score = MMD(orig_total_length, recon_total_length)
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
