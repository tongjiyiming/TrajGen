import os
import platform
import matplotlib

if platform.system() == 'Linux': matplotlib.use('Agg')
if platform.system() == 'Darwin': matplotlib.use('TkAgg')

from datetime import time

import torch
torch.manual_seed(0)
import torchvision
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from functools import partial
from tqdm import *
from data_utils import *
from preprocess import *
from utils import *
from DVAE import *
from SVAE_y import *
from SVAE_z import *
from LSTM import *

from scipy.spatial import distance

__all__ = ['loss_fn', 'Trainer']

def spatial_constraint_pkdd(angle_thres, precede_segment_thres, seq):
    cosine_thres = np.cos(angle_thres / 180 * np.pi)
    # print('cosine_thres', cosine_thres, 'precede_segment_thres', precede_segment_thres)
    cosine_scores = compute_cosine(seq)
    segment_length = compute_segment_length_loss(seq)
    segment_length = segment_length[:, :-1]
    spatial_constraint = torch.relu(torch.relu(cosine_scores - cosine_thres) * torch.relu(segment_length - precede_segment_thres))
    # print('cosine_scores', cosine_scores.shape, cosine_scores)
    # print('segment_length', segment_length.shape, segment_length)
    # print('spatial_constraint', spatial_constraint.shape, spatial_constraint)
    spatial_constraint = torch.sum(spatial_constraint)
    return spatial_constraint

def spatial_constraint_tdrive(angle_thres, seq):
    cosine_thres = np.cos(angle_thres / 180 * np.pi)
    cosine_scores = compute_cosine(seq)
    first_cosine_scores = cosine_scores[:, :-1]
    second_cosine_scores = cosine_scores[:, 1:]
    spatial_constraint = torch.relu(torch.relu(first_cosine_scores - cosine_thres) * torch.relu(second_cosine_scores - cosine_thres))
    # print('cosine_thres', cosine_thres)
    # print('first_cosine_scores', first_cosine_scores.shape, first_cosine_scores)
    # print('second_cosine_scores', second_cosine_scores.shape, second_cosine_scores)
    # print('spatial_constraint', spatial_constraint.shape, spatial_constraint)
    spatial_constraint = torch.sum(spatial_constraint)
    return spatial_constraint

def loss_fn(original_seq, recon_seq, f_mean, f_logvar, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar,
            spatial_constraint_func, beta, alpha):
    batch_size = original_seq.size(0)
    recon_loss = F.mse_loss(recon_seq, original_seq, reduction='sum')
    total_loss = recon_loss

    kld_f = torch.tensor(0)
    if f_mean is not None:
        kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean, 2) - torch.exp(f_logvar))
        total_loss += alpha * kld_f

    spatial_constraint = torch.tensor(0)
    orig_spatial_constraint = torch.tensor(0)
    kld_z = torch.tensor(0)

    if spatial_constraint_func is not None:
        spatial_constraint = spatial_constraint_func(recon_seq)
        orig_spatial_constraint = spatial_constraint_func(original_seq)
        total_loss += beta * spatial_constraint
    if z_post_logvar is not None:
        z_post_var = torch.exp(z_post_logvar)
        z_prior_var = torch.exp(z_prior_logvar)
        kld_z = 0.5 * torch.sum(
            z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)
        total_loss += alpha * kld_z

    return total_loss / batch_size, kld_f / batch_size, kld_z / batch_size, recon_loss / batch_size, \
           spatial_constraint / batch_size, orig_spatial_constraint / batch_size

class Trainer(object):
    def __init__(self,model,train,test,trainloader,testloader, data_log_folder, model_name, file_prefix,
                 spatial_constraint_func, beta=1.0, alpha=1.0,
                 epochs=100,batch_size=64,learning_rate=0.001,nsamples=1, eval_epoches=10,isTest=False,jit=None,plot_bound=0.01,
                 sample_path='./sample', recon_path='./recon/', transfer_path = './transfer/',
                 checkpoints='model.pth', device=torch.device('cuda:0')):
        self.trainloader = trainloader
        self.isTest = isTest
        self.train = train
        self.test = test
        self.data_log_folder = data_log_folder
        self.jit = jit
        self.plot_bound = plot_bound
        # self.useConstraints = useConstraints
        self.spatial_constraint_func = spatial_constraint_func
        self.beta = beta
        self.alpha = alpha
        # self.cosine_thres = cosine_thres
        self.model_name = model_name
        self.file_prefix = file_prefix
        # self.traj_len = traj_len
        self.testloader = testloader
        self.start_epoch = 0
        self.epochs = epochs
        self.eval_epoches = eval_epoches
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.optimizer = optim.Adam(self.model.parameters(),self.learning_rate)
        self.samples = nsamples
        self.sample_path = sample_path
        self.recon_path = recon_path
        self.transfer_path = transfer_path
        # self.test_f_expand = test_f_expand
        self.epoch_losses = []

    def save_checkpoint(self,epoch):
        print("save checkpoint for epoch {}".format(epoch))
        torch.save({
            'epoch' : epoch,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'losses' : self.epoch_losses},
            self.checkpoints)
        
    def load_checkpoint(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            print(os.path.isfile(self.checkpoints))
            if 'cuda' in str(self.device):
                checkpoint = torch.load(self.checkpoints)
            elif 'cpu' in str(self.device):
                checkpoint = torch.load(self.checkpoints, map_location=lambda storage, loc: storage)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except Exception as e:
            print(e)
            print("Failed to load Checkpoint Exists At '{}'. Start Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def sample_traj_f(self, test_f_expand, test_zf):
        with torch.no_grad():
            _,_,test_z = self.model.sample_z(1, random_sampling=False)
            test_zf = torch.cat((test_z, test_f_expand), dim=2)
            recon_x = self.model.decode_trajs(test_zf)
            return recon_x

    def viz_compare(self, epoch):
        for i,dataitem in enumerate(self.testloader,1):
            ids, xy = dataitem
            xy = xy.type(torch.FloatTensor)
            xy = xy.to(self.device)
            _, _, _, _, _, _, _, _, recon_xy = self.model(xy)

            for j in range(10):
                id = ids[j]
                real_one_traj = xy[j]
                fake_one_traj = recon_xy[j]
                print('mean distance error of one traj:', MDE(fake_one_traj, real_one_traj))
                if 'cpu' in str(self.device):
                    real_one_traj = real_one_traj.data.numpy()
                    fake_one_traj = fake_one_traj.data.numpy()
                elif 'cuda' in str(self.device):
                    real_one_traj = real_one_traj.data.cpu().numpy()
                    fake_one_traj = fake_one_traj.data.cpu().numpy()
                min_x = min(real_one_traj[:, 0].min(), fake_one_traj[:, 0].min()) - self.plot_bound
                max_x = max(real_one_traj[:, 0].max(), fake_one_traj[:, 0].max()) + self.plot_bound
                min_y = min(real_one_traj[:, 1].min(), fake_one_traj[:, 1].min()) - self.plot_bound
                max_y = max(real_one_traj[:, 1].max(), fake_one_traj[:, 1].max()) + self.plot_bound
                fake_save_file = os.path.join(self.data_log_folder, '{}_traj_epoch_{}_row_{}_fake.png'.format(self.file_prefix, epoch, id))
                plot_traj(fake_one_traj, self.model.traj_len, (min_x, max_x), (min_y, max_y), show=False, save_file=fake_save_file, jit=self.jit)
                # if epoch == self.eval_epoches:
                read_save_file = os.path.join(self.data_log_folder, '{}_traj_epoch_{}_row_{}_real.png'.format(self.file_prefix, epoch, id))
                plot_traj(real_one_traj, self.model.traj_len, (min_x, max_x), (min_y, max_y), show=False, save_file=read_save_file, jit=self.jit)
            if i > 0: break

    def evaluate(self):
        self.model.eval()
        ### mean distance error
        mde_list = []
        for i,dataitem in enumerate(self.testloader,1):
            ids, xy = dataitem
            # xy = torch.tensor(xy, dtype=torch.float32).squeeze(dim=0)
            xy = xy.type(torch.FloatTensor)
            xy = xy.to(self.device)
            _, _, _, _, _, _, _, _, recon_xy = self.model(xy)

            mde_list.append(MDE(recon_xy, xy))
        print('Mean Distance Error of all:', np.mean(mde_list))
        return mde_list

    def post_evaluate(self, data_name, angle_thres, precede_segment_thres):
        self.model.eval()
        ### mean distance error
        mde_list = []
        recon_trajs = []
        rand_trajs = []
        orig_trajs = []
        recon_angles = []
        orig_angles = []
        recon_segment_length = []
        orig_segment_length = []
        spatial_validity_mask = []

        x_min = 0
        x_max = (self.train.params['x_max'] - self.train.params['x_min'] + self.train.params['offset']) / self.train.params['scale']
        x_max = int(x_max)
        y_min = 0
        y_max = (self.train.params['y_max'] - self.train.params['y_min'] + self.train.params['offset']) / self.train.params['scale']
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
            n_subset = len(self.test)
            # n_subset = 10000
            x_tick_range = [0, x_bins_]
            y_tick_range = [0, y_bins_]
            # x_tick_range = [750 * n_per_bin, 950 * n_per_bin]
            # y_tick_range = [850 * n_per_bin, 1050 * n_per_bin]
        elif data_name == 'gowalla':
            n_per_bin = 0.1
            x_bins_ = int(x_max * n_per_bin)
            y_bins_ = int(y_max * n_per_bin)
            n_subset = len(self.test)
            # n_subset = 10000
            # x_tick_range = [0, x_bins_]
            # y_tick_range = [0, y_bins_]
            x_tick_range = [1900 * n_per_bin, 2300 * n_per_bin]
            y_tick_range = [1800 * n_per_bin, 2100 * n_per_bin]
        else:
            n_per_bin = 1
            x_bins_ = x_max * n_per_bin
            y_bins_ = y_max * n_per_bin
            n_subset = len(self.test)
            # n_subset = 10000
            x_tick_range = [0, x_bins_]
            y_tick_range = [0, y_bins_]
            # x_tick_range = [750 * n_per_bin, 950 * n_per_bin]
            # y_tick_range = [850 * n_per_bin, 1050 * n_per_bin]
        print('x_range, y_range, x_bins_, y_bins_', x_range, y_range, x_bins_, y_bins_)
        save_file = '{}/{}evaluation.bin'.format(self.data_log_folder, self.file_prefix)
        # if os.path.isfile(save_file):
        #     print('load previous evaluation results from ', save_file)
        #     with open(save_file, 'rb') as f:
        #         spatial_validity_mask, recon_trajs, orig_trajs, recon_angles, orig_angles, \
        #         recon_segment_length, orig_segment_length, mde_list = pickle.load(f)
        #         f.close()
        # else:
        # for _ in range(10):
        for i,dataitem in tqdm(enumerate(self.testloader, 1)):
            ids, xy = dataitem
            # xy = torch.tensor(xy, dtype=torch.float32).squeeze(dim=0)
            xy = xy.type(torch.FloatTensor).to(self.device)
            # sample multiple latent variablesTraining is complete
            # for _ in range(10):
            _, _, _, _, _, _, _, _, recon_xy = self.model(xy)
            if self.model_name == 'lstm':
                rand_xy = recon_xy
            else:
                rand_xy = self.model.sample_traj(self.batch_size)

            if xy.is_cuda:
                orig_trajs.append(xy.data.cpu().numpy())
                recon_trajs.append(recon_xy.data.cpu().numpy())
                rand_trajs.append(rand_xy.data.cpu().numpy())
            else:
                orig_trajs.append(xy.data.numpy())
                recon_trajs.append(recon_xy.data.numpy())
                rand_trajs.append(rand_xy.data.numpy())

            mde_list.append(MDE(recon_xy, xy))
            batch_orig_angles = compute_angle(xy)
            batch_recon_angles = compute_angle(rand_xy)
            orig_angles.append(batch_orig_angles)
            recon_angles.append(batch_recon_angles)
            batch_orig_segment_length = compute_segment_length(xy)
            batch_recon_segment_length = compute_segment_length(rand_xy)
            orig_segment_length.append(batch_orig_segment_length)
            recon_segment_length.append(batch_recon_segment_length)
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
        rand_trajs = np.concatenate(rand_trajs, axis=0)
        orig_trajs = np.concatenate(orig_trajs, axis=0)
        recon_angles = np.concatenate(recon_angles, axis=0)
        orig_angles = np.concatenate(orig_angles, axis=0)
        orig_segment_length = np.concatenate(orig_segment_length, axis=0)
        recon_segment_length = np.concatenate(recon_segment_length, axis=0)
        mde_list = np.array(mde_list)
        with open(save_file, 'wb') as f:
            res = (spatial_validity_mask, recon_trajs, orig_trajs, recon_angles, orig_angles,
                   recon_segment_length, orig_segment_length, mde_list)
            pickle.dump(res, f)
            f.close()

        # plot spatial constraints heatmap
        print('orig_trajs.min(), orig_trajs.max()', orig_trajs.min(), orig_trajs.max())
        print('recon_trajs.min(), recon_trajs.max()', recon_trajs.min(), recon_trajs.max())
        print('rand_trajs.min(), rand_trajs.max()', rand_trajs.min(), rand_trajs.max())
        if data_name == 'pkdd':
            plots_pkdd(rand_trajs * self.train.params['scale'], data_log_folder=self.data_log_folder, file_prefix=self.file_prefix)
        elif data_name == 'tdrive':
            plots_tdrive(rand_trajs * self.train.params['scale'], data_log_folder=self.data_log_folder, file_prefix=self.file_prefix)

        # get total length of each traj
        recon_angles = recon_angles.reshape(-1, 1)
        orig_angles = orig_angles.reshape(-1, 1)
        orig_total_length = orig_segment_length.sum(axis=-1).reshape(-1, 1)
        recon_total_length = recon_segment_length.sum(axis=-1).reshape(-1, 1)
        orig_segment_length = orig_segment_length.reshape(-1, 1)
        recon_segment_length = recon_segment_length.reshape(-1, 1)
        orig_2Dhist = compute_2Dhist_numpy(orig_trajs, x_range, y_range, x_bins_, y_bins_)
        recon_2Dhist = compute_2Dhist_numpy(recon_trajs, x_range, y_range, x_bins_, y_bins_)
        # orig_2Dhist = np.zeros((x_bins_, y_bins_))
        # for i in range(len(orig_trajs)):
        #     xy = orig_trajs[i]
        #     recon_xy = recon_trajs[i]
        #     orig_2Dhist += compute_2Dhist_numpy(xy, x_range, y_range, x_bins_, y_bins_)
        #     recon_2Dhist += compute_2Dhist_numpy(recon_xy, x_range, y_range, x_bins_, y_bins_)

        print('Mean Distance Error: ', np.mean(mde_list))

        # compute spatial validity score
        spatial_validity_mask = spatial_validity_mask.reshape(1, -1)[0]
        validity_score = len(np.where(spatial_validity_mask)[0]) / len(spatial_validity_mask)
        print('spatial validity score : {}'.format(validity_score))

        print('save angle distribution to {}/_angle_dist_{}'.format(self.data_log_folder, self.file_prefix))
        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        ax.hist(orig_angles, bins=180, range=[0, 180])
        ax.set_xlabel('Angles ranged in [0, 180]')
        ax.set_ylabel('Normalized Counts')
        plt.tight_layout()
        file_name = '{}/_angle_dist_{}_real.png'.format(self.data_log_folder, self.file_prefix)
        plt.savefig(file_name, dpi=120)
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        ax.hist(recon_angles, bins=180, range=[0, 180])
        ax.set_xlabel('Angles ranged in [0, 180]')
        ax.set_ylabel('Normalized Counts')
        plt.tight_layout()
        file_name = '{}/_angle_dist_{}_fake.png'.format(self.data_log_folder, self.file_prefix)
        plt.savefig(file_name, dpi=120)
        plt.close()

        # recon_angle_ratio = (recon_angles > angle_thres).sum() / recon_angles.shape[0]
        # orig_angle_ratio = (orig_angles > angle_thres).sum() / orig_angles.shape[0]
        # recon_density, _ = np.histogram(recon_angles, bins=180, range=[0, 180], density=True)
        # orig_density, _ = np.histogram(orig_angles, bins=180, range=[0, 180], density=True)
        # distribution_score = distance.jensenshannon(orig_density, recon_density)
        # distribution_score = distance.jensenshannon(orig_angles, recon_angles)
        distribution_score = MMD(orig_angles[:n_subset], recon_angles[:n_subset])
        print('angle distribution score: {:.4f}'.format(distribution_score))
        # print('orig_angle_ratio: {:.4f} recon_angle_ratio: {:.4f} distribution score: {:.4f}'.format(
        #     orig_angle_ratio, recon_angle_ratio, distribution_score))
        # distribution_score = MMD(orig_angles, recon_angles)
        # print('orig_angle_ratio: {:.4f} recon_angle_ratio: {:.4f} MMD: {:.4f}'.format(
        #     orig_angle_ratio, recon_angle_ratio, distribution_score))

        print('save segment length distribution to {}/segment_length_dist_{}'.format(self.data_log_folder, self.file_prefix))
        min_ = min(orig_segment_length.min(), recon_segment_length.min())
        max_ = max(orig_segment_length.max(), recon_segment_length.max())

        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        ax.hist(orig_segment_length, bins=100, range=[min_, max_])
        ax.set_xlabel('Length ranged in [{}, {}]'.format(min_, max_))
        ax.set_ylabel('Normalized Counts')
        plt.tight_layout()
        file_name = '{}/_segment_length_dist_{}_real.png'.format(self.data_log_folder, self.file_prefix)
        plt.savefig(file_name, dpi=120)
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        ax.hist(recon_segment_length, bins=100, range=[min_, max_])
        ax.set_xlabel('Length ranged in [{}, {}]'.format(min_, max_))
        ax.set_ylabel('Normalized Counts')
        plt.tight_layout()
        file_name = '{}/_segment_length_dist_{}_fake.png'.format(self.data_log_folder, self.file_prefix)
        plt.savefig(file_name, dpi=120)
        plt.close()

        # recon_density, _ = np.histogram(recon_segment_length, bins=bins_, range=[min_, max_], density=True)
        # orig_density, _ = np.histogram(orig_segment_length, bins=bins_, range=[min_, max_], density=True)
        # distribution_score = distance.jensenshannon(recon_density, orig_density)
        # distribution_score = distance.jensenshannon(orig_segment_length, recon_segment_length)
        distribution_score = MMD(orig_segment_length[:n_subset], recon_segment_length[:n_subset])
        print('segment length distribution score: {:.4f}'.format(distribution_score))
        # distribution_score = MMD(orig_segment_length.reshape(1, -1), recon_segment_length.reshape(1, -1))
        # print('segment length MMD: {:.4f}'.format(distribution_score))

        print('save segment length distribution to {}/total_length_dist_{}'.format(self.data_log_folder, self.file_prefix))
        min_ = min(orig_total_length.min(), recon_total_length.min())
        max_ = max(orig_total_length.max(), recon_total_length.max())

        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        ax.hist(orig_total_length, bins=100, range=[min_, max_])
        ax.set_xlabel('Length ranged in [{}, {}]'.format(min_, max_))
        ax.set_ylabel('Normalized Counts')
        plt.tight_layout()
        file_name = '{}/_total_length_dist_{}_real.png'.format(self.data_log_folder, self.file_prefix)
        plt.savefig(file_name, dpi=120)
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        ax.hist(recon_total_length, bins=100, range=[min_, max_])
        ax.set_xlabel('Length ranged in [{}, {}]'.format(min_, max_))
        ax.set_ylabel('Normalized Counts')
        plt.tight_layout()
        file_name = '{}/_total_length_dist_{}_fake.png'.format(self.data_log_folder, self.file_prefix)
        plt.savefig(file_name, dpi=120)
        plt.close()

        # distribution_score = distance.jensenshannon(orig_total_length, recon_total_length)
        distribution_score = MMD(orig_total_length[:n_subset], recon_total_length[:n_subset])
        print('total length distribution score: {:.4f}'.format(distribution_score))

        # 2d histgram
        print('save 2d histgram point count distribution to {}/_2dHistgram_dist_{}'.format(
            self.data_log_folder, self.file_prefix))
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
        file_name = '{}/_2dHistgram_dist_{}_real.png'.format(self.data_log_folder, self.file_prefix)
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
        file_name = '{}/_2dHistgram_dist_{}_fake.png'.format(self.data_log_folder, self.file_prefix)
        plt.savefig(file_name, dpi=120)
        plt.close()

        print('orig_2Dhist argmax grid', np.where(orig_2Dhist == orig_2Dhist.max()))
        print('recon_2Dhist argmax grid', np.where(recon_2Dhist == recon_2Dhist.max()))
        print('orig_2Dhist min max, recon_2Dhist min max', orig_2Dhist.min(), orig_2Dhist.max(),
              recon_2Dhist.min(), recon_2Dhist.max())
        # distribution_score = distance.jensenshannon(orig_2Dhist, recon_2Dhist)
        # distribution_score = MMD(orig_2Dhist, recon_2Dhist)
        distribution_score = MMD(orig_trajs.reshape(-1, 2)[:n_subset], recon_trajs.reshape(-1, 2)[:n_subset])
        print('2d point histgram distribution score: {:.4f}'.format(distribution_score))

        return mde_list

    def recon_traj(self, ind, savefig):
        print('run case study of reconstructing traj :', ind)
        self.model.eval()
        self.model.random_sampling = True # to reparameterize f z with real random sampling
        _, xy = self.test[ind]
        xy_torch = torch.tensor(xy, dtype=torch.float32)
        xy_torch = xy_torch.unsqueeze(dim=0).to(self.device)

        min_x = xy[:, 0].min() - self.plot_bound
        max_x = xy[:, 0].max() + self.plot_bound
        min_y = xy[:, 1].min() - self.plot_bound
        max_y = xy[:, 1].max() + self.plot_bound
        read_save_file = os.path.join(self.data_log_folder,
                                      '{}_{}_traj_{}_real.png'.format('_case_study', self.file_prefix, ind))
        plot_traj(xy, self.model.traj_len, (min_x, max_x), (min_y, max_y), show=not savefig,
                  save_file=read_save_file, jit=self.jit, doNote=False)

        for epoch in range(5):
            _, _, f, _, _, z, _, _, recon_xy = self.model(xy_torch)
            if 'cpu' in str(self.device):
                recon_xy = recon_xy.data.numpy()
            else:
                recon_xy = recon_xy.data.cpu().numpy()
            print('mean distance error of one traj:', MDE_numpy(recon_xy[0], xy))
            # min_x = min(real_one_traj[:, 0].min(), fake_one_traj[:, 0].min()) - self.plot_bound
            # max_x = max(real_one_traj[:, 0].max(), fake_one_traj[:, 0].max()) + self.plot_bound
            # min_y = min(real_one_traj[:, 1].min(), fake_one_traj[:, 1].min()) - self.plot_bound
            # max_y = max(real_one_traj[:, 1].max(), fake_one_traj[:, 1].max()) + self.plot_bound
            fake_save_file = os.path.join(self.data_log_folder,
                                          '{}_{}_traj_{}_sample_{}_fake.png'.format('_case_study', self.file_prefix, ind, epoch))
            plot_traj(recon_xy[0], self.model.traj_len, (min_x, max_x), (min_y, max_y), show=not savefig,
                      save_file=fake_save_file, jit=self.jit, doNote=False)

    def sample_traj_fix_f(self):
        pass

    def train_model(self):
        self.model.train()
        times = []
        for epoch in range(self.start_epoch,self.epochs):
            t1 = time.time()
            losses = []
            kld_fs = []
            kld_zs = []
            print("Running Epoch : {}".format(epoch+1))
            for i,dataitem in enumerate(self.trainloader,1):
            # for i,dataitem in tqdm(enumerate(self.trainloader,1)):
                ids, xy = dataitem
                # update learning rate
                update_lr = self.learning_rate * (0.1 ** (epoch // 30))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = update_lr
                xy = xy.type(torch.FloatTensor)
                xy = xy.to(self.device)
                self.optimizer.zero_grad()
                f_mean, f_logvar, f, z_post_mean, z_post_logvar, z, z_prior_mean, z_prior_logvar, recon_xy = self.model(xy)
                loss, kld_f, kld_z, recon_loss, spatial_constraint, original_spatial_constraint = loss_fn(
                    xy, recon_xy, f_mean, f_logvar, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar,
                    self.spatial_constraint_func, self.beta, self.alpha)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                kld_fs.append(kld_f.item())
                kld_zs.append(kld_z.item())

                if self.isTest: break
            meanloss = np.mean(losses)
            meanf = np.mean(kld_fs)
            meanz = np.mean(kld_zs)
            self.epoch_losses.append(meanloss)
            print("Epoch {} : Average Loss: {} KL of f : {} KL of z : {} recon loss : {} spatial constraint : {} original spatial constraint : {}".format(
                epoch+1,meanloss, meanf, meanz, recon_loss, spatial_constraint, original_spatial_constraint))
            t2 = time.time()
            times.append(t2-t1)
            if (epoch+1) % self.eval_epoches == 0:
                self.save_checkpoint(epoch+1)
                self.model.eval()
                # visualize real and reconstructed one
                self.viz_compare(epoch+1)
                self.evaluate()
                self.model.train()
        print("Training is complete")
        return times

if __name__ == "__main__":
    # isTest = True
    isTest = False
    # reload = True
    reload = False

    # doEval = True
    doEval = False

    data_name = 'pol'
    # data_name = 'gowalla'
    # data_name = 'pkdd'
    # data_name = 'tdrive'

    # model_name = 'dvsae'
    model_name = 'fdvsae'
    # model_name = 'vsae_y'
    # model_name = 'vsae_z'
    # model_name = 'lstm'

    useConstraints = 'useConstraint'
    # useConstraints = 'noConstraint'

    device = torch.device('cuda:0')
    # device = torch.device('cpu')

    if isTest:
        print('testing run')
        data_name = 'pol'
        traj_len = 32
        angle_thres = 30
        batch_size = 2
        test_batch = 12
        epochs = 1
        eval_epoches = 1
        learning_rate = 0.02
        f_dim = 256
        z_dim = 32
        encode_dims = [48, 32]
        decode_dims = [64, 32]
        beta = 1.
        alpha = 1.0
        jit = 0.1
        plot_bound = 0.1
        step = 256
    elif data_name == 'pol':
        traj_len = 8
        batch_size = 128
        test_batch = 128
        epochs = 1
        eval_epoches = 10
        learning_rate = 0.0002
        f_dim = 256
        z_dim = 32
        encode_dims = [48, 32]
        decode_dims = [64, 32]
        beta = 1.
        alpha = 1.0
        angle_thres = 180
        jit = 0.1
        plot_bound = 0.1
        step = 256
    elif data_name == "gowalla":
        traj_len = 16
        batch_size = 128
        test_batch = 128
        epochs = 100
        eval_epoches = 10
        learning_rate = 0.002
        f_dim = 256
        z_dim = 32
        encode_dims = [48, 32]
        decode_dims = [64, 32]
        beta = 1.
        alpha = 1.0
        angle_thres = 180
        jit = 0.1
        plot_bound = 0.1
        step = 256
    elif data_name == 'pkdd':
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
    elif data_name == 'tdrive':
        traj_len = 32
        batch_size = 128
        test_batch = 128
        epochs = 200
        eval_epoches = 10
        learning_rate = 0.0002
        f_dim = 256
        z_dim = 64
        encode_dims = [48, 16]
        decode_dims = [128]
        beta = 100.0
        alpha = 1.
        jit = None
        plot_bound = 0.01

    random_sampling = True
    if doEval: random_sampling = False
    if useConstraints == 'noConstraint': angle_thres = 180

    traj_train = Traj(*load_dataset(data_name, isTrain=True))
    traj_test = Traj(*load_dataset(data_name, isTrain=False))
    traj_len = traj_train.params['traj_len']
    print("initialize dataset with total {} training traj {} testing traj".format(len(traj_train), len(traj_test)))

    train_loader = torch.utils.data.DataLoader(traj_train, batch_size=batch_size, shuffle=True, num_workers=6)
    test_loader = torch.utils.data.DataLoader(traj_test, batch_size=test_batch, shuffle=True, num_workers=6)
    # test_loader = torch.utils.data.DataLoader(traj_train, batch_size=test_batch, shuffle=True, num_workers=6)

    angle_thres = None
    precede_segment_thres = None
    spatial_constraint_func = None
    if data_name == 'pkdd':
        angle_thres = 30  # best equal or larger than 60 degree
        precede_segment_thres = 166 / traj_train.params['scale']
        if useConstraints == 'useConstraint':
            spatial_constraint_func = partial(spatial_constraint_pkdd, angle_thres, precede_segment_thres)
            print('use spatial constraint angle thres {} precede segment thres {} use function {}'.format(
                angle_thres, precede_segment_thres, spatial_constraint_func))
            file_prefix = '{}_{}_angle{}_precedSegment{}_{}_'.format(
                model_name, useConstraints, angle_thres, precede_segment_thres, data_name)
        else:
            file_prefix = '{}_{}_{}_'.format(
                model_name, useConstraints, data_name)
    else:
        angle_thres = 30  # two consecutive angles are not both less than 30 degree
        if useConstraints == 'useConstraint':
            spatial_constraint_func = partial(spatial_constraint_tdrive, angle_thres)
            print('use spatial constraint two consecutive angle thres {} use function {}'.format(
                angle_thres, spatial_constraint_func))
            file_prefix = '{}_{}_angle{}_{}_'.format(model_name, useConstraints, angle_thres, data_name)
        else:
            file_prefix = '{}_{}_{}_'.format(model_name, useConstraints, data_name)
    if isTest: file_prefix += 'test_'

    checkpoints_folder = "checkpoints"
    if not os.path.isdir(checkpoints_folder): os.mkdir(checkpoints_folder)
    checkpoints = '{}/{}.pth'.format(checkpoints_folder, file_prefix)

    log_folder = './logs/'
    if not os.path.isdir(log_folder): os.mkdir(log_folder)
    data_log_folder = os.path.join(log_folder, data_name)
    if not os.path.isdir(data_log_folder): os.mkdir(data_log_folder)

    if model_name == 'dvsae':
        model = DisentangledVAE(f_dim=f_dim, z_dim=z_dim, traj_len=traj_len, factorised=False, device=device,
                              encode_dims=encode_dims, decode_dims=decode_dims, random_sampling=random_sampling)
    elif model_name == 'fdvsae':
        model = DisentangledVAE(f_dim=f_dim, z_dim=z_dim, traj_len=traj_len, factorised=True, device=device,
                              encode_dims=encode_dims, decode_dims=decode_dims, random_sampling=random_sampling)
    elif model_name == 'vsae_y':
        model = SVAE_y(f_dim=f_dim, z_dim=z_dim, traj_len=traj_len, factorised=False, device=device,
                              encode_dims=encode_dims, decode_dims=decode_dims)
    elif model_name == 'vsae_z':
        model = SVAE_z(f_dim=f_dim, z_dim=z_dim, traj_len=traj_len, factorised=False, device=device,
                              encode_dims=encode_dims, decode_dims=decode_dims)
    elif model_name == 'lstm':
        model =LSTM(f_dim=f_dim, z_dim=z_dim, traj_len=traj_len, factorised=False, device=device,
                              encode_dims=encode_dims, decode_dims=decode_dims)

    trainer = Trainer(model, traj_train, traj_test, train_loader, test_loader, model_name=model_name, file_prefix=file_prefix,
                      data_log_folder=data_log_folder,batch_size=batch_size, epochs=epochs,
                      spatial_constraint_func=spatial_constraint_func, beta=beta, alpha=alpha,
                      eval_epoches=eval_epoches, isTest=isTest, jit=jit, plot_bound=plot_bound, checkpoints=checkpoints,
                      learning_rate=learning_rate, device=device)
    print('checkpoints', checkpoints)

    if doEval:
        trainer.load_checkpoint()
        print('angle threshold:', angle_thres)
        # trainer.post_evaluate(data_name=data_name, angle_thres=angle_thres, precede_segment_thres=precede_segment_thres)

        # trainer.recon_traj(np.random.choice(len(traj_test)), savefig=True)

        ### disentangle case study
        num = 10
        # z list
        test_z_list = []
        for epoch in range(num):
            _, _, test_z = trainer.model.sample_z(1, random_sampling=False)
            # for i in range(traj_len): test_z[0, i, 0] = epoch
            # for j in range(traj_len):
            # test_z = test_z * (epoch + 1) / 2
            # test_z = test_z + torch.rand((1, traj_len, z_dim), device=device)
            test_z = test_z + torch.rand((1, traj_len, z_dim), device=device) * 10
            # for i in range(1, z_dim, 2): test_z[0, 2, i] = (epoch - num // 2)*100
            test_z_list.append(test_z)
        # y list
        test_f_list = []
        for epoch in range(num):
            # ### manual assign y
            # test_f = torch.rand(1, f_dim, device=device) * 100
            # # test_f = torch.ones((1, f_dim), device=device) * 100
            # # for i in range(0, f_dim, 2): test_f[0, i] = epoch
            # # test_f[0, 0] = (epoch - 4) * 10000
            # test_f_list.append(test_f)

            ### use a real traj to encode y
            ind = np.random.choice(len(traj_test))
            print('test sample index:', ind)
            _, xy = traj_test[ind]
            xy_torch = torch.tensor(xy, dtype=torch.float32)
            xy_torch = xy_torch.unsqueeze(dim=0).to(trainer.device)
            _, _, f, _, _, z, _, _, recon_xy = trainer.model(xy_torch)
            test_f_list.append(f)

        with torch.no_grad():
            fig, axs = plt.subplots(num, num, figsize=(num*4, num*4))
            for i in range(num):
                for j in range(num):
                    test_f = test_f_list[i]
                    test_z = test_z_list[j]
                    test_f_expand = test_f.unsqueeze(1).expand(1, traj_len, f_dim)
                    test_zf = torch.cat((test_z, test_f_expand), dim=2)
                    recon_traj = trainer.model.decode_trajs(test_zf)
                    recon_traj = recon_traj.cpu().numpy()

                    # sample_traj_file = '{}/_case_study_{}_fix_y_{}_fix_z_{}.png'.format(data_log_folder, file_prefix, i, j)
                    # plot_one_traj(recon_traj[0, :, 0], recon_traj[0, :, 1], doNote=False, sample_traj_file=sample_traj_file,
                    #               # x_range=[40, 130],
                    #               # y_range=[40,130]
                    #               )

                    sample_traj_file = '{}/_case_study_{}.png'.format(data_log_folder, file_prefix)
                    ax = axs[i][j]
                    x = recon_traj[0, :, 0]
                    y = recon_traj[0, :, 1]
                    ax.plot(x, y, marker='+', markersize=12, markeredgecolor='red', linestyle='-', linewidth=4)
                    ax.set_xticks([])
                    ax.set_yticks([])
            plt.tight_layout()
            plt.box(on=None)
            plt.savefig(sample_traj_file, dpi=120)
            # plt.show()
            plt.close()

        print("end evaluation")
    else:
        if reload:
            trainer.load_checkpoint()
            trainer.train_model()
        else:
            t1 = time.time()
            trainer.train_model()
            t2 = time.time()
            print('average run time', t2-t1)
            with open('{}/_run_time_traj_len_{}_{}'.format(data_log_folder, traj_len, file_prefix), 'w') as f:
                f.write('average run time per epoch is {:.2f}'.format((t2 - t1)/epochs))
                f.close()
        print("end training")
