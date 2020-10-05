from __future__ import print_function, division

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import platform
import matplotlib

if platform.system() == 'Linux': matplotlib.use('Agg')
if platform.system() == 'Darwin': matplotlib.use('TkAgg')

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, ZeroPadding2D
from keras.layers import BatchNormalization, Activation
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import keras.backend as K

from tqdm import *

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# from sklearn.covariance import EmpiricalCovariance
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import average_precision_score
# from sklearn.model_selection import train_test_split

from data_utils import *
from utils import *
from preprocess import *

import torch
from scipy.spatial import distance

# Uncomment below if you want to not have warnings!

import os, warnings
warnings.filterwarnings("ignore")

def get_both_routes(trajectories):
    routes = []
    for traj in trajectories:
        curr_route = []
        prev_elem = []
        for elem in traj:
            if len(curr_route) % traj_len == 0:
                if len(curr_route) > 0:
                    routes.append(curr_route)
                curr_route = [[elem[1], elem[2], 0.0, 0.0]]
            else:
                curr_route.append(np.hstack([np.asarray(elem[1:]), np.asarray(elem[1:]) - np.asarray(prev_elem)]))
            prev_elem = elem[1:]
    return routes

# isTest = True
isTest = False

reload = False
# reload = True

# doEval = True
doEval = False

# data_name = 'pol'
# data_name = 'gowalla'
# data_name = 'pkdd'
data_name = 'tdrive'

if isTest:
    print('testing run')
    data_name = 'pol'
    raw_traj_len = 16  # *** liming *** change to a two columns since my data do not have velocity
    RAW_NUM_VALS = 2
    traj_len = 8
    NUM_VALS = 4
    cosine_thres = 1.0
    batch_size = 128
    epochs = 1
    reload_epochs = 40000
    eval_freq = 1
    save_freq = 1
    learn_rate = 0.0001
elif data_name == 'pol':
    raw_traj_len = 16  # *** liming *** change to a two columns since my data do not have velocity
    RAW_NUM_VALS = 2
    traj_len = 8
    NUM_VALS = 4
    cosine_thres = 1.0
    batch_size = 128
    epochs = 40000
    reload_epochs = 0
    eval_freq = 10000
    save_freq = 10000
    learn_rate = 0.0001
elif data_name == "gowalla":
    raw_traj_len = 8  # *** liming *** change to a two columns since my data do not have velocity
    RAW_NUM_VALS = 2
    traj_len = 4
    NUM_VALS = 4
    cosine_thres = 1.0
    batch_size = 128
    epochs = 40000
    reload_epochs = 40000
    eval_freq = 10000
    save_freq = 10000
    learn_rate = 0.0001
elif data_name == 'pkdd':
    raw_traj_len = 32  # *** liming *** change to a two columns since my data do not have velocity
    RAW_NUM_VALS = 2
    traj_len = 16
    NUM_VALS = 4
    angle_thres = 30
    cosine_thres = np.cos(angle_thres / 180 * np.pi)
    precede_segment_thres = 166
    batch_size = 128
    epochs = 40000
    reload_epochs = 0
    eval_freq = 10000
    save_freq = 10000
    learn_rate = 0.001
elif data_name == 'tdrive':
    raw_traj_len = 32  # *** liming *** change to a two columns since my data do not have velocity
    RAW_NUM_VALS = 2
    traj_len = 16
    NUM_VALS = 4
    angle_thres = 30
    cosine_thres = np.cos(angle_thres / 180 * np.pi)
    precede_segment_thres = 166
    batch_size = 128
    epochs = 40000
    reload_epochs = 40000
    eval_freq = 10000
    save_freq = 10000
    learn_rate = 0.001



file_prefix = 'igmm_gan_{}_'.format(data_name)
thres_angle = np.arccos(cosine_thres) / np.pi * 180

checkpoints_folder = "checkpoints"
if not os.path.isdir(checkpoints_folder): os.mkdir(checkpoints_folder)
generator_checkpoints = '{}/{}generator.h5'.format(checkpoints_folder, file_prefix)
discriminator_checkpoints = '{}/{}discriminator.h5'.format(checkpoints_folder, file_prefix)

log_folder = './logs/'
if not os.path.isdir(log_folder): os.mkdir(log_folder)
data_log_folder = os.path.join(log_folder, data_name)
if not os.path.isdir(data_log_folder): os.mkdir(data_log_folder)

traj_train, params = load_dataset(data_name, isTrain=True)
traj_test, _ = load_dataset(data_name, isTrain=False)
raw_traj_len = params['traj_len']
traj_len = raw_traj_len // 2
print("initialize dataset with total {} training traj {} testing traj".format(len(traj_train), len(traj_test)))

# X_train = []
# for i in range(len(traj_train)):
#     X_train.append(traj_train[i][0].reshape(traj_len, NUM_VALS)) # *** liming *** change to a two columns since my data do not have velocity
# X_train = np.stack(X_train, axis=0)
# X_train = np.reshape(X_train, (len(X_train), traj_len, NUM_VALS, 1))

X_train = traj_train.reshape(-1, traj_len, NUM_VALS)
X_train = np.reshape(X_train, (len(X_train), traj_len, NUM_VALS, 1))
X_test = traj_test.reshape(-1, traj_len, NUM_VALS)
X_test = np.reshape(X_test, (len(X_test), traj_len, NUM_VALS, 1))

# X_train, X_test, _, _ = train_test_split(
#     remaining_routes, np.zeros(len(remaining_routes)), test_size=0.2, random_state=42)
# '''Reshaping the data'''

print('X_train.shape', X_train.shape)

"""# Model"""
latent_dim = 100
input_shape = (int(traj_len), NUM_VALS, 1)

def make_encoder():
    modelE = Sequential()
    modelE.add(Conv2D(32, kernel_size=(3, 2), padding="same", input_shape=input_shape))
    modelE.add(BatchNormalization(momentum=0.8))
    modelE.add(Activation("relu"))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))
    modelE.add(Conv2D(64, kernel_size=(3, 2), padding="same"))
    modelE.add(BatchNormalization(momentum=0.8))
    modelE.add(Activation("relu"))
    modelE.add(MaxPooling2D(pool_size=(2, 1)))
    modelE.add(Conv2D(128, kernel_size=(3, 2), padding="same"))
    modelE.add(BatchNormalization(momentum=0.8))
    modelE.add(Activation("relu"))
    modelE.add(Flatten())
    modelE.add(Dense(latent_dim))

    return modelE


# Encoder 1

enc_model_1 = make_encoder()

img = Input(shape=input_shape)
z = enc_model_1(img)
encoder1 = Model(img, z)

# Generator

modelG = Sequential()
modelG.add(Dense(128 * int(traj_len / 4) * 1, input_dim=latent_dim))
modelG.add(BatchNormalization(momentum=0.8))
modelG.add(LeakyReLU(alpha=0.2))
modelG.add(Reshape((int(traj_len / 4), 1, 128)))
modelG.add(Conv2DTranspose(128, kernel_size=(3, 2), strides=2, padding="same"))
modelG.add(BatchNormalization(momentum=0.8))
modelG.add(LeakyReLU(alpha=0.2))
modelG.add(Conv2DTranspose(64, kernel_size=(3, 2), strides=2, padding="same"))
modelG.add(BatchNormalization(momentum=0.8))
modelG.add(LeakyReLU(alpha=0.2))
modelG.add(Conv2DTranspose(1, kernel_size=(3, 2), strides=1, padding="same"))

z = Input(shape=(latent_dim,))
gen_img = modelG(z)
generator = Model(z, gen_img)

# Encoder 2

enc_model_2 = make_encoder()

img = Input(shape=input_shape)
z = enc_model_2(img)
encoder2 = Model(img, z)

# Discriminator

modelD = Sequential()
modelD.add(Conv2D(32, kernel_size=3, strides=2, input_shape=input_shape, padding="same"))
modelD.add(LeakyReLU(alpha=0.2))
modelD.add(Dropout(0.25))
modelD.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
modelD.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
modelD.add(BatchNormalization(momentum=0.8))
modelD.add(LeakyReLU(alpha=0.2))
modelD.add(Dropout(0.25))
modelD.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
modelD.add(BatchNormalization(momentum=0.8))
modelD.add(LeakyReLU(alpha=0.2))
modelD.add(Flatten())
modelD.add(Dense(1, activation='sigmoid'))

discriminator = modelD

optimizer = Adam(learn_rate, 0.5)

# Build and compile the discriminator
discriminator.compile(loss=['binary_crossentropy'],
                      optimizer=optimizer,
                      metrics=['accuracy'])

discriminator.trainable = False

# First image encoding
img = Input(shape=input_shape)
z = encoder1(img)

# Generate image from encoding
img_ = generator(z)

# Second image encoding
z_ = encoder2(img_)

# The discriminator takes generated images as input and determines if real or fake
real = discriminator(img_)

# Set up and compile the combined model
# Trains generator to fool the discriminator
# and decrease loss between (img, _img) and (z, z_)
bigan_generator = Model(img, [real, img_, z_])
bigan_generator.compile(loss=['binary_crossentropy', 'mean_absolute_error',
                              'mean_squared_error'], optimizer=optimizer)

# Adversarial ground truths
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

g_loss_list = []
d_loss_list = []

if reload:
    print('reload pretrained model:', discriminator_checkpoints, generator_checkpoints)
    start_epochs = reload_epochs
    discriminator = load_model(discriminator_checkpoints)
    bigan_generator = load_model(generator_checkpoints)
else:
    start_epochs = 0

if doEval:
    X = X_train
    # X = X_test
    eval_epoches = len(X) // batch_size + 1

    mde_list = []
    recon_trajs = []
    orig_trajs = []
    recon_angles = []
    orig_angles = []
    recon_segment_length = []
    orig_segment_length = []
    spatial_validity_mask = []
    # n_subset = 20000
    n_subset = len(X_test)
    x_min = 0
    x_max = (params['x_max'] - params['x_min'] + params['offset']) / params['scale']
    x_max = int(x_max)
    y_min = 0
    y_max = (params['y_max'] - params['y_min'] + params['offset']) / params['scale']
    y_max = int(y_max)
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)
    if data_name == 'pkdd':
        n_per_bin = 10
        x_bins_ = x_max * n_per_bin
        y_bins_ = y_max * n_per_bin
        n_subset = 20000
        x_tick_range = [60 * n_per_bin, 80 * n_per_bin]
        y_tick_range = [440 * n_per_bin, 470 * n_per_bin]
    elif data_name == 'tdrive':
        n_per_bin = 1
        x_bins_ = x_max * n_per_bin
        y_bins_ = y_max * n_per_bin
        n_subset = len(X_test)
        # n_subset = 10000
        x_tick_range = [0, x_bins_]
        y_tick_range = [0, y_bins_]
        # x_tick_range = [750 * n_per_bin, 950 * n_per_bin]
        # y_tick_range = [850 * n_per_bin, 1050 * n_per_bin]
    elif data_name == 'gowalla':
        n_per_bin = 0.1
        x_bins_ = int(x_max * n_per_bin)
        y_bins_ = int(y_max * n_per_bin)
        n_subset = len(X_test)
        # n_subset = 10000
        # x_tick_range = [0, x_bins_]
        # y_tick_range = [0, y_bins_]
        x_tick_range = [1900 * n_per_bin, 2300 * n_per_bin]
        y_tick_range = [1800 * n_per_bin, 2100 * n_per_bin]
    else:
        n_per_bin = 1
        x_bins_ = x_max * n_per_bin
        y_bins_ = y_max * n_per_bin
        n_subset = len(X_test)
        # n_subset = 10000
        x_tick_range = [0, x_bins_]
        y_tick_range = [0, y_bins_]
        # x_tick_range = [750 * n_per_bin, 950 * n_per_bin]
        # y_tick_range = [850 * n_per_bin, 1050 * n_per_bin]
    print('x_range, y_range, x_bins_, y_bins_', x_range, y_range, x_bins_, y_bins_)

    for epoch in tqdm(range(eval_epoches)):
        if epoch < eval_epoches - 1:  idx = np.arange(epoch * batch_size, (epoch+1) * batch_size)
        else: idx = np.arange(epoch * batch_size, len(X) - 1)
        imgs = X[idx]
        z = encoder1.predict(imgs)
        imgs_ = generator.predict(z)
        print('imgs[0]', imgs[0])
        print('imgs_[0]', imgs_[0])
        imgs = imgs.squeeze(axis=-1)
        imgs_ = imgs_.squeeze(axis=-1)
        imgs = imgs.reshape((-1, raw_traj_len, RAW_NUM_VALS))
        imgs_ = imgs_.reshape((-1, raw_traj_len, RAW_NUM_VALS))
        # imgs = np.concatenate([imgs[:, :, 0:RAW_NUM_VALS], imgs[:, :, RAW_NUM_VALS:]], axis=1)
        # imgs_ = np.concatenate([imgs_[:, :, 0:RAW_NUM_VALS], imgs_[:, :, RAW_NUM_VALS:]], axis=1)
        orig_trajs.append(imgs)
        recon_trajs.append(imgs_)
        xy = torch.tensor(imgs)
        recon_xy = torch.tensor(imgs_)

        batch_orig_angles = compute_angle(xy)
        batch_recon_angles = compute_angle(recon_xy)
        orig_angles.append(batch_orig_angles)
        recon_angles.append(batch_recon_angles)
        batch_orig_segment_length = compute_segment_length(xy)
        batch_recon_segment_length = compute_segment_length(recon_xy)
        orig_segment_length.append(batch_orig_segment_length)
        recon_segment_length.append(batch_recon_segment_length)
        mde_list.append(MDE(recon_xy, xy))
        if data_name == 'pkdd':
            mask = (batch_recon_segment_length[:, :-1] > precede_segment_thres) & (batch_recon_angles < angle_thres)
        elif data_name == 'tdrive':
            mask = (batch_recon_angles[:-1] < angle_thres) & (batch_recon_angles[1:] < angle_thres)
        else:
            mask = [True]
        spatial_validity_mask.append(mask)
    # print('self.test.x_range, self.test.y_range', self.test.x_range, self.test.y_range)
    spatial_validity_mask = np.concatenate(spatial_validity_mask, axis=0).reshape(1, -1)[0]
    recon_trajs = np.concatenate(recon_trajs, axis=0)
    orig_trajs = np.concatenate(orig_trajs, axis=0)
    recon_angles = np.concatenate(recon_angles, axis=0)
    orig_angles = np.concatenate(orig_angles, axis=0)
    orig_segment_length = np.concatenate(orig_segment_length, axis=0)
    recon_segment_length = np.concatenate(recon_segment_length, axis=0)
    mde_list = np.array(mde_list)

    # get total length of each traj
    recon_angles = recon_angles.reshape(-1, 1)
    orig_angles = orig_angles.reshape(-1, 1)
    orig_total_length = orig_segment_length.sum(axis=-1).reshape(-1, 1)
    recon_total_length = recon_segment_length.sum(axis=-1).reshape(-1, 1)
    orig_segment_length = orig_segment_length.reshape(-1, 1)
    recon_segment_length = recon_segment_length.reshape(-1, 1)
    orig_2Dhist = compute_2Dhist_numpy(orig_trajs, x_range, y_range, x_bins_, y_bins_)
    recon_2Dhist = compute_2Dhist_numpy(recon_trajs, x_range, y_range, x_bins_, y_bins_)

    # compute spatial validity score
    validity_score = len(np.where(spatial_validity_mask)[0]) / len(spatial_validity_mask)
    print('spatial validity score : {}'.format(validity_score))
    # plot spatial constraints heatmap
    print('orig_trajs.min(), orig_trajs.max()', orig_trajs.min(), orig_trajs.max())
    print('recon_trajs.min(), recon_trajs.max()', recon_trajs.min(), recon_trajs.max())
    if data_name == 'pkdd':
        plots_pkdd(recon_trajs * params['scale'], data_log_folder=data_log_folder, file_prefix=file_prefix)
    elif data_name == 'tdrive':
        plots_tdrive(recon_trajs * params['scale'], data_log_folder=data_log_folder, file_prefix=file_prefix)

    print('Mean Distance Error: ', np.mean(mde_list))

    print('save angle distribution to {}/_angle_dist_{}'.format(data_log_folder, file_prefix))
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.hist(orig_angles, bins=180, range=[0, 180])
    ax.set_xlabel('Angles ranged in [0, 180]')
    ax.set_ylabel('Normalized Counts')
    plt.tight_layout()
    file_name = '{}/_angle_dist_{}_real.png'.format(data_log_folder, file_prefix)
    plt.savefig(file_name, dpi=120)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.hist(recon_angles, bins=180, range=[0, 180])
    ax.set_xlabel('Angles ranged in [0, 180]')
    ax.set_ylabel('Normalized Counts')
    plt.tight_layout()
    file_name = '{}/_angle_dist_{}_fake.png'.format(data_log_folder, file_prefix)
    plt.savefig(file_name, dpi=120)
    plt.close()

    # recon_density, _ = np.histogram(recon_angles, bins=180, range=[0, 180], density=True)
    # orig_density, _ = np.histogram(orig_angles, bins=180, range=[0, 180], density=True)
    # distribution_score = distance.jensenshannon(orig_density, recon_density)
    # distribution_score = distance.jensenshannon(orig_angles, recon_angles)
    # distribution_score = MMD(orig_angles[:n_subset].reshape(-1, 1), recon_angles[:n_subset].reshape(-1, 1))
    # print('orig_angle_ratio: {:.4f} recon_angle_ratio: {:.4f} distribution score: {:.4f}'.format(
    #     orig_angle_ratio, recon_angle_ratio, distribution_score))
    distribution_score = MMD(orig_angles[:n_subset], recon_angles[:n_subset])
    print('angle distribution score: {:.4f}'.format(distribution_score))
    # distribution_score = MMD(orig_angles.reshape(-1, 1), recon_angles.reshape(-1, 1))
    # print('orig_angle_ratio: {:.4f} recon_angle_ratio: {:.4f} MMD: {:.4f}'.format(
    #     orig_angle_ratio, recon_angle_ratio, distribution_score))

    # segment lengths
    orig_segment_length = np.concatenate(orig_segment_length, axis=0).reshape(1, -1)[0]
    recon_segment_length = np.concatenate(recon_segment_length, axis=0).reshape(1, -1)[0]

    print('save segment length distribution to {}/segment_length_dist_{}'.format(data_log_folder, file_prefix))
    min_ = min(orig_segment_length.min(), recon_segment_length.min())
    max_ = max(orig_segment_length.max(), recon_segment_length.max())

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.hist(orig_segment_length, bins=100, range=[min_, max_])
    ax.set_xlabel('Length ranged in [{}, {}]'.format(min_, max_))
    ax.set_ylabel('Normalized Counts')
    plt.tight_layout()
    file_name = '{}/_segment_length_dist_{}_real.png'.format(data_log_folder, file_prefix)
    plt.savefig(file_name, dpi=120)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.hist(recon_segment_length, bins=100, range=[min_, max_])
    ax.set_xlabel('Length ranged in [{}, {}]'.format(min_, max_))
    ax.set_ylabel('Normalized Counts')
    plt.tight_layout()
    file_name = '{}/_segment_length_dist_{}_fake.png'.format(data_log_folder, file_prefix)
    plt.savefig(file_name, dpi=120)
    plt.close()

    # recon_density, _ = np.histogram(recon_segment_length, bins=bins_, range=[min_, max_], density=True)
    # orig_density, _ = np.histogram(orig_segment_length, bins=bins_, range=[min_, max_], density=True)
    # distribution_score = distance.jensenshannon(recon_density, orig_density)
    # distribution_score = distance.jensenshannon(orig_segment_length, recon_segment_length)
    distribution_score = MMD(orig_segment_length[:n_subset].reshape(-1, 1), recon_segment_length[:n_subset].reshape(-1, 1))
    print('segment length distribution score: {:.4f}'.format(distribution_score))
    # distribution_score = MMD(orig_segment_length.reshape(1, -1), recon_segment_length.reshape(1, -1))
    # print('segment length MMD: {:.4f}'.format(distribution_score))

    # total length distribution
    orig_total_length = np.concatenate(orig_total_length, axis=0).reshape(1, -1)[0]
    recon_total_length = np.concatenate(recon_total_length, axis=0).reshape(1, -1)[0]

    print('save segment length distribution to {}/total_length_dist_{}'.format(data_log_folder, file_prefix))
    min_ = min(orig_total_length.min(), recon_total_length.min())
    max_ = max(orig_total_length.max(), recon_total_length.max())

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.hist(orig_total_length, bins=100, range=[min_, max_])
    ax.set_xlabel('Length ranged in [{}, {}]'.format(min_, max_))
    ax.set_ylabel('Normalized Counts')
    plt.tight_layout()
    file_name = '{}/_total_length_dist_{}_real.png'.format(data_log_folder, file_prefix)
    plt.savefig(file_name, dpi=120)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.hist(recon_total_length, bins=100, range=[min_, max_])
    ax.set_xlabel('Length ranged in [{}, {}]'.format(min_, max_))
    ax.set_ylabel('Normalized Counts')
    plt.tight_layout()
    file_name = '{}/_total_length_dist_{}_fake.png'.format(data_log_folder, file_prefix)
    plt.savefig(file_name, dpi=120)
    plt.close()

    # distribution_score = distance.jensenshannon(orig_total_length, recon_total_length)
    distribution_score = MMD(orig_total_length[:n_subset].reshape(-1, 1), recon_total_length[:n_subset].reshape(-1, 1))
    print('total length distribution score: {:.4f}'.format(distribution_score))

    # 2d histgram
    print('save 2d histgram point count distribution to {}/_2dHistgram_dist_{}'.format(
        data_log_folder, file_prefix))
    # bins_ = 100
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    im = ax.imshow(np.log10(orig_2Dhist.T), cmap='Blues')
    # im = ax.imshow(orig_2Dhist.T, cmap='Blues')
    ax.set_xlabel('X ranged in [{}, {}]'.format(x_range[0], x_range[1]))
    ax.set_ylabel('Y ranged in [{}, {}]'.format(y_range[0], y_range[1]))
    ax.set_xlim(x_tick_range)
    ax.set_ylim(y_tick_range)
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_tick_labels)
    # ax.set_yticks(y_ticks)
    # ax.set_yticklabels(y_tick_labels)
    cbar = ax.figure.colorbar(im, ax=ax, cmap="Blues")
    cbar.ax.set_ylabel('Counts')
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    plt.tight_layout()
    file_name = '{}/_2dHistgram_dist_{}_real.png'.format(data_log_folder, file_prefix)
    plt.savefig(file_name, dpi=120)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    im = ax.imshow(np.log10(recon_2Dhist.T), cmap='Blues')
    # im = ax.imshow(recon_2Dhist.T, cmap='Blues')
    ax.set_xlabel('X ranged in [{}, {}]'.format(x_range[0], x_range[1]))
    ax.set_ylabel('Y ranged in [{}, {}]'.format(y_range[0], y_range[1]))
    ax.set_xlim(x_tick_range)
    ax.set_ylim(y_tick_range)
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_tick_labels)
    # ax.set_yticks(y_ticks)
    # ax.set_yticklabels(y_tick_labels)
    cbar = ax.figure.colorbar(im, ax=ax, cmap="Blues")
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    file_name = '{}/_2dHistgram_dist_{}_fake.png'.format(data_log_folder, file_prefix)
    plt.savefig(file_name, dpi=120)
    plt.close()

    print(np.where(orig_2Dhist == orig_2Dhist.max()))
    print(np.where(recon_2Dhist == recon_2Dhist.max()))
    print('orig_2Dhist, recon_2Dhist', orig_2Dhist.min(), orig_2Dhist.max(),
          recon_2Dhist.min(), recon_2Dhist.max())
    # distribution_score = distance.jensenshannon(orig_2Dhist, recon_2Dhist)
    # distribution_score = MMD(orig_2Dhist.reshape(-1, 1), recon_2Dhist.reshape(-1, 1))
    distribution_score = MMD(orig_trajs.reshape(-1, 2)[:n_subset], recon_trajs.reshape(-1, 2)[:n_subset])
    print('2d point histgram distribution score: {:.4f}'.format(distribution_score))

else:
    print("start igmm-gan training ...")
    for epoch in range(start_epochs, epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images and encode/decode/encode
        idx = np.random.randint(0, len(X_train), batch_size)
        imgs = X_train[idx]
        z = encoder1.predict(imgs)
        imgs_ = generator.predict(z)

        # Train the discriminator (imgs are real, imgs_ are fake)
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(imgs_, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (z -> img is valid and img -> z is is invalid)
        g_loss = bigan_generator.train_on_batch(imgs, [real, imgs, z])

        g_loss_list.append(g_loss)
        d_loss_list.append(d_loss)

        # print("%d - %d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (id_remove, epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))

        # If at save interval => save generated image samples
        if (epoch+1) % save_freq == 0:
            bigan_generator.save(generator_checkpoints)
            discriminator.save(discriminator_checkpoints)

        if (epoch+1) % eval_freq == 0:
            # X = X_train
            X = X_test
            eval_epoches = len(X) // batch_size + 1

            mde_list = []
            recon_trajs = []
            orig_trajs = []
            recon_angles = []
            orig_angles = []
            recon_segment_length = []
            orig_segment_length = []
            spatial_validity_mask = []
            # n_subset = 20000
            n_subset = len(X_test)
            x_min = 0
            x_max = (params['x_max'] - params['x_min'] + params['offset']) / params['scale']
            x_max = int(x_max)
            y_min = 0
            y_max = (params['y_max'] - params['y_min'] + params['offset']) / params['scale']
            y_max = int(y_max)
            x_range = (x_min, x_max)
            y_range = (y_min, y_max)
            if data_name == 'pkdd':
                n_per_bin = 10
                x_bins_ = x_max * n_per_bin
                y_bins_ = y_max * n_per_bin
                n_subset = 20000
                x_tick_range = [60 * n_per_bin, 80 * n_per_bin]
                y_tick_range = [440 * n_per_bin, 470 * n_per_bin]
            elif data_name == 'tdrive':
                n_per_bin = 1
                x_bins_ = x_max * n_per_bin
                y_bins_ = y_max * n_per_bin
                n_subset = len(X_test)
                # n_subset = 10000
                x_tick_range = [0, x_bins_]
                y_tick_range = [0, y_bins_]
                # x_tick_range = [750 * n_per_bin, 950 * n_per_bin]
                # y_tick_range = [850 * n_per_bin, 1050 * n_per_bin]
            elif data_name == 'gowalla':
                n_per_bin = 0.1
                x_bins_ = int(x_max * n_per_bin)
                y_bins_ = int(y_max * n_per_bin)
                n_subset = len(X_test)
                # n_subset = 10000
                # x_tick_range = [0, x_bins_]
                # y_tick_range = [0, y_bins_]
                x_tick_range = [1900 * n_per_bin, 2300 * n_per_bin]
                y_tick_range = [1800 * n_per_bin, 2100 * n_per_bin]
            else:
                n_per_bin = 1
                x_bins_ = x_max * n_per_bin
                y_bins_ = y_max * n_per_bin
                n_subset = len(X_test)
                # n_subset = 10000
                x_tick_range = [0, x_bins_]
                y_tick_range = [0, y_bins_]
                # x_tick_range = [750 * n_per_bin, 950 * n_per_bin]
                # y_tick_range = [850 * n_per_bin, 1050 * n_per_bin]
            print('x_range, y_range, x_bins_, y_bins_', x_range, y_range, x_bins_, y_bins_)

            for epoch in tqdm(range(eval_epoches)):
                if epoch < eval_epoches - 1:
                    idx = np.arange(epoch * batch_size, (epoch + 1) * batch_size)
                else:
                    idx = np.arange(epoch * batch_size, len(X) - 1)
                imgs = X[idx]
                z = encoder1.predict(imgs)
                imgs_ = generator.predict(z)
                z = np.random.normal(size=(batch_size, latent_dim))
                rand_imgs_ = generator.predict(z)
                if epoch ==0:
                    print('imgs[0]', imgs[0])
                    print('imgs_[0]', imgs_[0])
                imgs = imgs.squeeze(axis=-1)
                imgs_ = imgs_.squeeze(axis=-1)
                imgs = imgs.reshape((-1, raw_traj_len, RAW_NUM_VALS))
                imgs_ = imgs_.reshape((-1, raw_traj_len, RAW_NUM_VALS))
                rand_imgs_ = rand_imgs_.reshape((-1, raw_traj_len, RAW_NUM_VALS))
                # imgs = np.concatenate([imgs[:, :, 0:RAW_NUM_VALS], imgs[:, :, RAW_NUM_VALS:]], axis=1)
                # imgs_ = np.concatenate([imgs_[:, :, 0:RAW_NUM_VALS], imgs_[:, :, RAW_NUM_VALS:]], axis=1)
                orig_trajs.append(imgs)
                recon_trajs.append(imgs_)
                xy = torch.tensor(imgs)
                recon_xy = torch.tensor(imgs_)
                rand_xy = torch.tensor(rand_imgs_)

                batch_orig_angles = compute_angle(xy)
                batch_recon_angles = compute_angle(rand_xy)
                orig_angles.append(batch_orig_angles)
                recon_angles.append(batch_recon_angles)
                batch_orig_segment_length = compute_segment_length(xy)
                batch_recon_segment_length = compute_segment_length(rand_xy)
                orig_segment_length.append(batch_orig_segment_length)
                recon_segment_length.append(batch_recon_segment_length)
                mde_list.append(MDE(recon_xy, xy))
                if data_name == 'pkdd':
                    mask = (batch_recon_segment_length[:, :-1] > precede_segment_thres) & (
                                batch_recon_angles < angle_thres)
                elif data_name == 'tdrive':
                    mask = (batch_recon_angles[:, :-1] < angle_thres) & (batch_recon_angles[:, 1:] < angle_thres)
                else:
                    mask = [True]
                spatial_validity_mask.append(mask)
            # print('self.test.x_range, self.test.y_range', self.test.x_range, self.test.y_range)
            spatial_validity_mask = np.concatenate(spatial_validity_mask, axis=0).reshape(1, -1)[0]
            recon_trajs = np.concatenate(recon_trajs, axis=0)
            orig_trajs = np.concatenate(orig_trajs, axis=0)
            recon_angles = np.concatenate(recon_angles, axis=0)
            orig_angles = np.concatenate(orig_angles, axis=0)
            orig_segment_length = np.concatenate(orig_segment_length, axis=0)
            recon_segment_length = np.concatenate(recon_segment_length, axis=0)
            mde_list = np.array(mde_list)

            # get total length of each traj
            recon_angles = recon_angles.reshape(-1, 1)
            orig_angles = orig_angles.reshape(-1, 1)
            orig_total_length = orig_segment_length.sum(axis=-1).reshape(-1, 1)
            recon_total_length = recon_segment_length.sum(axis=-1).reshape(-1, 1)
            orig_segment_length = orig_segment_length.reshape(-1, 1)
            recon_segment_length = recon_segment_length.reshape(-1, 1)
            orig_2Dhist = compute_2Dhist_numpy(orig_trajs, x_range, y_range, x_bins_, y_bins_)
            recon_2Dhist = compute_2Dhist_numpy(recon_trajs, x_range, y_range, x_bins_, y_bins_)

            # compute spatial validity score
            validity_score = len(np.where(spatial_validity_mask)[0]) / len(spatial_validity_mask)
            print('spatial validity score : {}'.format(validity_score))
            # plot spatial constraints heatmap
            print('orig_trajs.min(), orig_trajs.max()', orig_trajs.min(), orig_trajs.max())
            print('recon_trajs.min(), recon_trajs.max()', recon_trajs.min(), recon_trajs.max())
            if data_name == 'pkdd':
                plots_pkdd(recon_trajs * params['scale'], data_log_folder=data_log_folder, file_prefix=file_prefix)
            elif data_name == 'tdrive':
                plots_tdrive(recon_trajs * params['scale'], data_log_folder=data_log_folder, file_prefix=file_prefix)

            print('Mean Distance Error: ', np.mean(mde_list))

            print('save angle distribution to {}/_angle_dist_{}'.format(data_log_folder, file_prefix))
            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
            ax.hist(orig_angles, bins=180, range=[0, 180])
            ax.set_xlabel('Angles ranged in [0, 180]')
            ax.set_ylabel('Normalized Counts')
            plt.tight_layout()
            file_name = '{}/_angle_dist_{}_real.png'.format(data_log_folder, file_prefix)
            plt.savefig(file_name, dpi=120)
            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
            ax.hist(recon_angles, bins=180, range=[0, 180])
            ax.set_xlabel('Angles ranged in [0, 180]')
            ax.set_ylabel('Normalized Counts')
            plt.tight_layout()
            file_name = '{}/_angle_dist_{}_fake.png'.format(data_log_folder, file_prefix)
            plt.savefig(file_name, dpi=120)
            plt.close()

            # recon_density, _ = np.histogram(recon_angles, bins=180, range=[0, 180], density=True)
            # orig_density, _ = np.histogram(orig_angles, bins=180, range=[0, 180], density=True)
            # distribution_score = distance.jensenshannon(orig_density, recon_density)
            # distribution_score = distance.jensenshannon(orig_angles, recon_angles)
            # distribution_score = MMD(orig_angles[:n_subset].reshape(-1, 1), recon_angles[:n_subset].reshape(-1, 1))
            # print('orig_angle_ratio: {:.4f} recon_angle_ratio: {:.4f} distribution score: {:.4f}'.format(
            #     orig_angle_ratio, recon_angle_ratio, distribution_score))
            distribution_score = MMD(orig_angles[:n_subset], recon_angles[:n_subset])
            print('angle distribution score: {:.4f}'.format(distribution_score))
            # distribution_score = MMD(orig_angles.reshape(-1, 1), recon_angles.reshape(-1, 1))
            # print('orig_angle_ratio: {:.4f} recon_angle_ratio: {:.4f} MMD: {:.4f}'.format(
            #     orig_angle_ratio, recon_angle_ratio, distribution_score))

            # segment lengths
            orig_segment_length = np.concatenate(orig_segment_length, axis=0).reshape(1, -1)[0]
            recon_segment_length = np.concatenate(recon_segment_length, axis=0).reshape(1, -1)[0]

            print('save segment length distribution to {}/segment_length_dist_{}'.format(data_log_folder, file_prefix))
            min_ = min(orig_segment_length.min(), recon_segment_length.min())
            max_ = max(orig_segment_length.max(), recon_segment_length.max())

            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
            ax.hist(orig_segment_length, bins=100, range=[min_, max_])
            ax.set_xlabel('Length ranged in [{}, {}]'.format(min_, max_))
            ax.set_ylabel('Normalized Counts')
            plt.tight_layout()
            file_name = '{}/_segment_length_dist_{}_real.png'.format(data_log_folder, file_prefix)
            plt.savefig(file_name, dpi=120)
            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
            ax.hist(recon_segment_length, bins=100, range=[min_, max_])
            ax.set_xlabel('Length ranged in [{}, {}]'.format(min_, max_))
            ax.set_ylabel('Normalized Counts')
            plt.tight_layout()
            file_name = '{}/_segment_length_dist_{}_fake.png'.format(data_log_folder, file_prefix)
            plt.savefig(file_name, dpi=120)
            plt.close()

            # recon_density, _ = np.histogram(recon_segment_length, bins=bins_, range=[min_, max_], density=True)
            # orig_density, _ = np.histogram(orig_segment_length, bins=bins_, range=[min_, max_], density=True)
            # distribution_score = distance.jensenshannon(recon_density, orig_density)
            # distribution_score = distance.jensenshannon(orig_segment_length, recon_segment_length)
            distribution_score = MMD(orig_segment_length[:n_subset].reshape(-1, 1),
                                     recon_segment_length[:n_subset].reshape(-1, 1))
            print('segment length distribution score: {:.4f}'.format(distribution_score))
            # distribution_score = MMD(orig_segment_length.reshape(1, -1), recon_segment_length.reshape(1, -1))
            # print('segment length MMD: {:.4f}'.format(distribution_score))

            # total length distribution
            orig_total_length = np.concatenate(orig_total_length, axis=0).reshape(1, -1)[0]
            recon_total_length = np.concatenate(recon_total_length, axis=0).reshape(1, -1)[0]

            print('save segment length distribution to {}/total_length_dist_{}'.format(data_log_folder, file_prefix))
            min_ = min(orig_total_length.min(), recon_total_length.min())
            max_ = max(orig_total_length.max(), recon_total_length.max())

            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
            ax.hist(orig_total_length, bins=100, range=[min_, max_])
            ax.set_xlabel('Length ranged in [{}, {}]'.format(min_, max_))
            ax.set_ylabel('Normalized Counts')
            plt.tight_layout()
            file_name = '{}/_total_length_dist_{}_real.png'.format(data_log_folder, file_prefix)
            plt.savefig(file_name, dpi=120)
            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
            ax.hist(recon_total_length, bins=100, range=[min_, max_])
            ax.set_xlabel('Length ranged in [{}, {}]'.format(min_, max_))
            ax.set_ylabel('Normalized Counts')
            plt.tight_layout()
            file_name = '{}/_total_length_dist_{}_fake.png'.format(data_log_folder, file_prefix)
            plt.savefig(file_name, dpi=120)
            plt.close()

            # distribution_score = distance.jensenshannon(orig_total_length, recon_total_length)
            distribution_score = MMD(orig_total_length[:n_subset].reshape(-1, 1),
                                     recon_total_length[:n_subset].reshape(-1, 1))
            print('total length distribution score: {:.4f}'.format(distribution_score))

            # 2d histgram
            print('save 2d histgram point count distribution to {}/_2dHistgram_dist_{}'.format(
                data_log_folder, file_prefix))
            # bins_ = 100
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            im = ax.imshow(np.log10(orig_2Dhist.T), cmap='Blues')
            # im = ax.imshow(orig_2Dhist.T, cmap='Blues')
            ax.set_xlabel('X ranged in [{}, {}]'.format(x_range[0], x_range[1]))
            ax.set_ylabel('Y ranged in [{}, {}]'.format(y_range[0], y_range[1]))
            ax.set_xlim(x_tick_range)
            ax.set_ylim(y_tick_range)
            # ax.set_xticks(x_ticks)
            # ax.set_xticklabels(x_tick_labels)
            # ax.set_yticks(y_ticks)
            # ax.set_yticklabels(y_tick_labels)
            cbar = ax.figure.colorbar(im, ax=ax, cmap="Blues")
            cbar.ax.set_ylabel('Counts')
            # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
            plt.tight_layout()
            file_name = '{}/_2dHistgram_dist_{}_real.png'.format(data_log_folder, file_prefix)
            plt.savefig(file_name, dpi=120)
            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            im = ax.imshow(np.log10(recon_2Dhist.T), cmap='Blues')
            # im = ax.imshow(recon_2Dhist.T, cmap='Blues')
            ax.set_xlabel('X ranged in [{}, {}]'.format(x_range[0], x_range[1]))
            ax.set_ylabel('Y ranged in [{}, {}]'.format(y_range[0], y_range[1]))
            ax.set_xlim(x_tick_range)
            ax.set_ylim(y_tick_range)
            # ax.set_xticks(x_ticks)
            # ax.set_xticklabels(x_tick_labels)
            # ax.set_yticks(y_ticks)
            # ax.set_yticklabels(y_tick_labels)
            cbar = ax.figure.colorbar(im, ax=ax, cmap="Blues")
            cbar.ax.set_ylabel('Counts')
            plt.tight_layout()
            file_name = '{}/_2dHistgram_dist_{}_fake.png'.format(data_log_folder, file_prefix)
            plt.savefig(file_name, dpi=120)
            plt.close()

            print(np.where(orig_2Dhist == orig_2Dhist.max()))
            print(np.where(recon_2Dhist == recon_2Dhist.max()))
            print('orig_2Dhist, recon_2Dhist', orig_2Dhist.min(), orig_2Dhist.max(),
                  recon_2Dhist.min(), recon_2Dhist.max())
            # distribution_score = distance.jensenshannon(orig_2Dhist, recon_2Dhist)
            # distribution_score = MMD(orig_2Dhist.reshape(-1, 1), recon_2Dhist.reshape(-1, 1))
            distribution_score = MMD(orig_trajs.reshape(-1, 2)[:n_subset], recon_trajs.reshape(-1, 2)[:n_subset])
            print('2d point histgram distribution score: {:.4f}'.format(distribution_score))