import json
import pickle
import time

import numpy as np
import torch
import pandas as pd
import geopandas as gpd
from multiprocessing import Pool

from preprocess import plots_pkdd, plot_one_traj
from utils import *

angle_thres = 150
model_name = 'llm'
data_name = 'pkdd'
data_log_folder = './logs/{}'.format(data_name)
generated_data_file = '/media/liming/Liming1/llm_porto_results/generated_traj_50000_beam_search_with_sampling.csv'

crs = "EPSG:4326"
local_crs = "EPSG:3763"
gen_line_col = 'centroids'
def preprocess_generated_pkdd(ind):
    # print(ind)
    c = np.array(eval(data.loc[ind, gen_line_col]))
    points_gpd = gpd.GeoSeries(gpd.points_from_xy(c[:, 0], c[:, 1]), crs=crs)
    points_gpd = points_gpd.to_crs(local_crs)
    post_traj_list = [[p.x, p.y] for p in points_gpd]
    return np.array(post_traj_list)

post_traj_file = 'dataset/{}/{}_post_traj_file.bin'.format(data_name, data_name)
train_traj_file = 'dataset/{}/{}_training_traj_file.bin'.format(data_name, data_name)
test_traj_file = 'dataset/{}/{}_testing_traj_file.bin'.format(data_name, data_name)
params_file = 'dataset/{}/{}_params.bin'.format(data_name, data_name)

with open(post_traj_file, 'rb') as f:
    post_traj_list = pickle.load(f)
    f.close()
    print('load a saved trajectory file: {}'.format(post_traj_file))
    print('original trajectory data\n', post_traj_list[0])

print('load saved preprocess params file: {}'.format(params_file))
with open(params_file, 'rb') as f:
    params = pickle.load(f)
    f.close()
    print('load a saved trajectory file: {}'.format(params_file))
    print(params)

### plot real traj
# plots_pkdd(post_traj_list[:10], data_log_folder=data_log_folder, file_prefix='pkdd_spatial_validity_real')
plots_pkdd(post_traj_list, data_log_folder=data_log_folder, file_prefix='pkdd_spatial_validity_real')

### load another dataset. test or generated
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

precede_segment_thres = 166 / params['scale']

file_prefix = '{}_angle{}_precedSegment{}_{}_'.format(
    model_name, angle_thres, precede_segment_thres, data_name)

### get metrics and plots
x_min = 0
x_max = (params['x_max'] - params['x_min'] + params['offset']) / params['scale']
x_max = int(x_max)
y_min = 0
y_max = (params['y_max'] - params['y_min'] + params['offset']) / params['scale']
y_max = int(y_max)
x_range = (x_min, x_max)
y_range = (y_min, y_max)

n_per_bin = 10
x_bins_ = x_max * n_per_bin
y_bins_ = y_max * n_per_bin
n_subset = 20000
x_tick_range = [60 * n_per_bin, 80 * n_per_bin]
y_tick_range = [440 * n_per_bin, 470 * n_per_bin]

print('x_range, y_range, x_bins_, y_bins_', x_range, y_range, x_bins_, y_bins_)
save_file = '{}/{}evaluation_journal.bin'.format(data_log_folder, file_prefix)
print('save result to {}'.format(save_file))

# compute
mde_list = []
recon_trajs = []
rand_trajs = []
orig_trajs = []
recon_angles = []
recon_segment_length = []

def spatial_validity_score(one_traj):
    orig_angles = []
    orig_segment_length = []
    spatial_validity_mask = []
    if one_traj.shape[0] < 3: return []
    one_traj[:, 0] = (one_traj[:, 0] - params['x_min'] + params['offset']) / params['scale']
    one_traj[:, 1] = (one_traj[:, 1] - params['y_min'] + params['offset']) / params['scale']
    xy = torch.tensor(one_traj).unsqueeze(dim=0)
    batch_orig_angles = compute_angle(xy)
    orig_angles.append(batch_orig_angles)
    batch_orig_segment_length = compute_segment_length(xy)
    orig_segment_length.append(batch_orig_segment_length)
    mask = (batch_orig_segment_length[:, :-1] > precede_segment_thres) & (batch_orig_angles > angle_thres)
    return list(mask[0])
    #     spatial_validity_mask.extend(list(mask[0]))
    # validity_score = len(np.where(spatial_validity_mask)[0]) / len(spatial_validity_mask)
    # return validity_score

# orig_validity_score = spatial_validity_score(post_traj_list, params, precede_segment_thres, angle_thres)
pool = Pool()
t0 = time.time()
tmp_list = pool.map(spatial_validity_score, post_traj_list)
print('use time:', time.time() - t0)
spatial_validity_mask = []
for l in tmp_list:
    spatial_validity_mask.extend(l)
orig_validity_score = len(np.where(spatial_validity_mask)[0]) / len(spatial_validity_mask)
print('original spatial validity score : {}'.format(orig_validity_score))

# read the other dataset
data = pd.read_csv(
    generated_data_file,
    # nrows=10
)
# data.columns = [x if x != 'centroids' else 'POLYLINE' for x in data.columns]
print(data.columns)

one_gen_traj = preprocess_generated_pkdd(0)
print('one generated sample\n', one_gen_traj)
plot_one_traj(one_gen_traj[:, 0], one_gen_traj[:, 1], doNote=True,
              sample_traj_file='./logs/pkdd/pkdd_generated_plot_one_traj.png',
              x_range=x_range,
              y_range=y_range
              )

pool = Pool()
t0 = time.time()
tmp_list = pool.map(preprocess_generated_pkdd, range(len(data)))
print('use time:', time.time() - t0)
post_gen_traj_list = []
post_gen_traj_list = [x for x in tmp_list if x.shape[0] > 0]

### heatmap and validity check
plots_pkdd(post_gen_traj_list, data_log_folder=data_log_folder, file_prefix='pkdd_spatial_validity_llm_gen')

# gen_validity_score = spatial_validity_score(post_gen_traj_list, params, precede_segment_thres, angle_thres)
# pool = Pool()
t0 = time.time()
tmp_list = pool.map(spatial_validity_score, post_gen_traj_list)
print('use time:', time.time() - t0)
spatial_validity_mask = []
for l in tmp_list:
    spatial_validity_mask.extend(l)
gen_validity_score = len(np.where(spatial_validity_mask)[0]) / len(spatial_validity_mask)
print('generated spatial validity score : {}'.format(gen_validity_score))