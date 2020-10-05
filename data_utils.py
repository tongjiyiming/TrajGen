import torch
import pandas as pd
import numpy as np
import pickle, os
import matplotlib.pyplot as plt

class Traj(torch.utils.data.Dataset):
    def __init__(self, traj_list, params):
        self.traj_list, self.params = traj_list, params
        self.length = len(self.traj_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        xy = self.traj_list[idx]
        return idx, xy

def load_dataset(data_name, isTrain):
    if data_name == 'pol':
        return load_pol(isTrain)
    if data_name == 'gowalla':
        return load_gowalla(isTrain)
    if data_name == 'pkdd':
        return load_pkdd(isTrain)
    if data_name == 'tdrive':
        return load_tdrive(isTrain)

def min_max_scale(a, min_x, min_y, x_scale, y_scale):
    arr = a.copy()
    arr[:, 0] = (arr[:, 0] - min_x) / x_scale
    arr[:, 1] = (arr[:, 1] - min_y) / y_scale
    return arr

def sample_traj(a, traj_len):
    if len(a) == traj_len:
        return a
    else:
        arr = a.copy()
        idx = np.arange(1, len(arr)-1)
        np.random.shuffle(idx)
        sampled_idx = [0] + list(idx[:traj_len-2]) + [len(arr)-1]
        return arr[sampled_idx]

def load_tdrive(isTrain):
    if isTrain:
        path = 'dataset/tdrive/tdrive_training_traj_file.bin'
    else:
        path = 'dataset/tdrive/tdrive_testing_traj_file.bin'
    params_file = 'dataset/tdrive/tdrive_params.bin'
    with open(path, 'rb') as f:
        traj_list = pickle.load(f)
        f.close()
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
        f.close()
    return traj_list, params

def load_pkdd(isTrain):
    if isTrain:
        path = 'dataset/pkdd/pkdd_training_traj_file.bin'
    else:
        path = 'dataset/pkdd/pkdd_testing_traj_file.bin'
    params_file = 'dataset/pkdd/pkdd_params.bin'
    with open(path, 'rb') as f:
        traj_list = pickle.load(f)
        f.close()
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
        f.close()
    return traj_list, params

def load_gowalla(isTrain):
    data_name = 'gowalla'
    if isTrain:
        path = 'dataset/{}/{}_training_traj_file.bin'.format(data_name, data_name)
    else:
        path = 'dataset/{}/{}_testing_traj_file.bin'.format(data_name, data_name)
    params_file = 'dataset/{}/{}_params.bin'.format(data_name, data_name)
    with open(path, 'rb') as f:
        traj_list = pickle.load(f)
        f.close()
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
        f.close()
    return traj_list, params

def load_pol(isTrain):
    if isTrain:
        path = 'dataset/pol/pol_training_traj_file.bin'
    else:
        path = 'dataset/pol/pol_testing_traj_file.bin'
    params_file = 'dataset/pol/pol_params.bin'
    with open(path, 'rb') as f:
        traj_list = pickle.load(f)
        f.close()
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
        f.close()
    return traj_list, params