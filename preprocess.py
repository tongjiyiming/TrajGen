import pandas as pd
import geopandas as gpd
import numpy as np
from numpy.linalg.linalg import norm
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from adjustText import adjust_text
from multiprocessing import Pool
import pickle
import os
import time

from utils import *

def plot_one_traj(x, y, doNote=True, sample_traj_file=None, x_range=None, y_range=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(x, y, marker='+', markersize=8, markeredgecolor='red', linestyle='--', linewidth=0.5)
    if doNote:
        texts = []
        for i in range(len(x)):
            texts.append(plt.text(x[i], y[i], str(i+1), ))
        adjust_text(texts)
    # ax.grid()
    if x_range: ax.set_xlim(x_range)
    if y_range: ax.set_ylim(y_range)
    plt.tight_layout()
    plt.box(on=None)
    if sample_traj_file:
        plt.savefig(sample_traj_file, dpi=120)
        plt.close()
    else:
        plt.show()

# def preprocess_cut_by_distance(x, y):
#     post_points = np.c_[x, y]
#     start_points = post_points[:-1].copy()
#     end_points = post_points[1:].copy()
#     dist = norm(start_points - end_points, axis=1)
#     # dist = np.sqrt(np.square(x[:-1] - x[1:]) + np.square(y[:-1] - y[1:]))
#     print('dist', dist)
#     mask = (dist < dist_thres_low) | (dist > dist_thres_high)
#     mask = np.where(mask)[0]
#     print(mask)
#     # post_points = np.c_[x, y]
#     if len(mask) == 0:
#         return [post_points]
#     else:
#         post_cut_traj_list = []
#         for i in range(-1, len(mask)):
#             if i == -1:
#                 t = post_points[:mask[0]]
#             elif i == len(mask) - 1:
#                 t = post_points[mask[-1] + 1:]
#             else:
#                 t = post_points[mask[i] + 1:mask[i+1]+1]
#         return post_cut_traj_list

def preprocess_filter_points_pkdd(x, y):
    # pkdd is a complete trip, not need to cut with stay point
    traj_len = len(x)
    post_trajs = []
    i = 0
    traj = []
    while i < traj_len - 1:
        traj.append([x[i], y[i]])
        for j in range(i + 1, traj_len):
            # since there is not stay points, we only detect noise points with fixed i, and jump over them
            # example: 1--2--3-4-5--6--7, get 1--2--3--5--6--7
            dist = np.sqrt(np.square(x[i] - x[j]) + np.square(y[i] - y[j])) # use fixe i, not rolling j-1
            # print(dist)
            if dist > dist_thres_low and dist < dist_thres_high:
                break
        i = j
    # if the last point to its precedor is larger than dist_thres
    if dist > dist_thres_low and dist < dist_thres_high:
        traj.append([x[i], y[i]])
    post_trajs = [np.array(traj)]
    return post_trajs

def get_segment_length(points):
    first_points = points[:-1]
    second_points = points[1:]
    vectors = second_points - first_points
    segment_length = norm(vectors, axis=1)
    return segment_length

def get_angles(points):
    first_points = points[:-1]
    second_points = points[1:]
    vectors = second_points - first_points
    first_vectors = vectors[:-1]
    second_vectors = vectors[1:]
    DotProducts = first_vectors[:, 0] * second_vectors[:, 0] + first_vectors[:, 1] * second_vectors[:, 1]
    NormKronecker = norm(first_vectors, axis=1) * norm(second_vectors, axis=1)
    cosine_sim = - DotProducts / NormKronecker
    cosine_sim[np.where(cosine_sim > 1)] = 1
    cosine_sim[np.where(cosine_sim < -1)] = -1
    angles = np.arccos(cosine_sim) / np.pi * 180
    return angles

def preprocess_cut_traj(post_points):
    angles = get_angles(post_points)
    mask = np.where(angles < angle_thres)[0]
    if len(mask) == 0:
        return [post_points]
    else:
        mask += 1
        post_cut_traj_list = []
        for i in range(-1, len(mask)):
            if i == -1:
                post_cut_traj_list.append(post_points[:mask[0]+1])
            elif i == len(mask) - 1:
                post_cut_traj_list.append(post_points[mask[-1]:])
            else:
                post_cut_traj_list.append(post_points[mask[i]:mask[i+1]+1])
        return post_cut_traj_list

def preprocess_pkdd(ind):
    print(ind)
    c = np.array(eval(data.loc[ind, 'POLYLINE']))
    if c.shape[0] < 2: return []
    points_gpd = gpd.GeoSeries(gpd.points_from_xy(c[:, 0], c[:, 1]), crs=crs)
    points_gpd = points_gpd.to_crs(local_crs)
    # points_gpd = points_gpd[:80]
    # plot_one_traj(points_gpd.x, points_gpd.y, doNote=True)
    post_traj_list = preprocess_filter_points_pkdd(points_gpd.x, points_gpd.y)
    # for traj in post_traj_list:
    #     plot_one_traj(traj[:, 0], traj[:, 1], doNote=True)
    return post_traj_list

# def preprocess_filter_points_tdrive(x, y):
#     traj_len = len(x)
#     # post_points = []
#     post_trajs = []
#     i = 0
#     traj = []
#     while i < traj_len - 1:
#         traj.append([x[i], y[i]])
#         for j in range(i + 1, traj_len):
#             # example: 1--2--3-4-5--6--7, get 1--2--3, 5--6--7
#             dist = np.sqrt(np.square(x[j-1] - x[j]) + np.square(y[j-1] - y[j]))
#             # print(dist, i, j)
#             if dist > dist_thres_low and dist < dist_thres_high:
#                 break
#         if j >= i + 2: # detect a stay point
#             if len(traj) > 2: post_trajs.append(np.array(traj))
#             traj = []
#             i = j - 1
#         else:
#             i = j
#     # if the last point to its precedor is larger than dist_thres
#     if dist > dist_thres_low and dist < dist_thres_high:
#         traj.append([x[i], y[i]])
#         if len(traj) > 2: post_trajs.append(np.array(traj))
#     return post_trajs

def preprocess_filter_points_tdrive(x, y):
    traj_len = len(x)
    # post_points = []
    post_trajs = []
    i = 0
    traj = []
    while i < traj_len - 1:
        traj.append([x[i], y[i]])
        for j in range(i + 1, traj_len):
            # example: 1--2--3-4-5--6--7, get 1--2--3, 5--6--7
            dist = np.sqrt(np.square(x[i] - x[j]) + np.square(y[i] - y[j]))
            # print(dist, i, j)
            if dist > dist_thres_low and dist < dist_thres_high:
                break
        i = j
    # if the last point to its precedor is larger than dist_thres
    if dist > dist_thres_low and dist < dist_thres_high:
        traj.append([x[i], y[i]])
    post_trajs.append(np.array(traj))
    return post_trajs

def cut_by_time_tdrive(x, y, times):
    x_list, y_list, time_list = [], [], []
    tmp_x, tmp_y, tmp_t = [], [], []
    for i in range(1, len(times)):
        if times[i] - times[i-1] >= 0 and times[i] - times[i-1] <= time_thres_high:
            tmp_x.append(x[i-1])
            tmp_y.append(y[i-1])
            tmp_t.append(times[i-1])
            if i == len(times) - 1:
                tmp_x.append(x[i])
                tmp_y.append(y[i])
                tmp_t.append(times[i])
                x_list.append(tmp_x)
                y_list.append(tmp_y)
                time_list.append(tmp_t)
        else:
            if len(tmp_x) > 2:
                x_list.append(tmp_x)
                y_list.append(tmp_y)
                time_list.append(tmp_t)
            tmp_x, tmp_y, tmp_t = [], [], []
    return x_list, y_list, time_list

def remove_dist_noise_tdrive(x, y, times):
    traj_len = len(x)
    i = 0
    # print(i, traj_len)
    traj = []
    while i < traj_len - 1:
        traj.append([x[i], y[i]])
        for j in range(i + 1, min(i + 6, traj_len)):
            # example: 1--2--3-4-5--6--7, get 1--2--3, 5--6--7
            last_ = j-1
            dist = np.sqrt(np.square(x[last_] - x[j]) + np.square(y[last_] - y[j]))
            t = times[j] - times[last_]
            if t == 0: continue
            speed = dist / t * 3.6 # km per hour
            # print(dist, t, speed, i, j)
            if dist > dist_thres_low and dist < dist_thres_high and speed > speed_thres_low and speed < speed_thres_high:
                break
        i = j
    # print(i)
    # if the last point to its precedor is larger than dist_thres
    traj.append([x[i], y[i]])
    # if speed > speed_thres_low and speed < speed_thres_high:
    #     traj.append([x[i], y[i]])
    return np.array(traj)

def preprocess_filter_points_tdrive_with_time(x, y, times):
    x_list, y_list, time_list = cut_by_time_tdrive(x, y, times)
    post_trajs = [remove_dist_noise_tdrive(x_list[i], y_list[i], time_list[i]) for i in range(len(x_list))]
    # print(time_list[0])
    # post_trajs = [remove_dist_noise_tdrive(x_list[0], y_list[0], time_list[0])]
    # print('post_trajs', post_trajs)
    return post_trajs

def preprocess_tdrive(i):
    print(i)
    ID_col, raw_time_col, Y_col, X_col = 'ID', 'CheckinTime', 'Y', 'X'
    traj_file = os.path.join(path, '{}.txt'.format(i))
    # c = pd.read_csv(traj_file, header=None, names=[ID_col, raw_time_col, Y_col, X_col])
    c = pd.read_csv(traj_file, header=None, names=[ID_col, raw_time_col, X_col, Y_col], parse_dates=[raw_time_col])
    if c.shape[0] < 2: return []

    times = c[raw_time_col] - c[raw_time_col].min()
    times = np.array([t.seconds for t in times])
    points_gpd = gpd.GeoSeries(gpd.points_from_xy(c.loc[:, X_col], c.loc[:, Y_col]), crs=crs)
    points_gpd = points_gpd.to_crs(local_crs)
    mask = (np.isinf(points_gpd.x) | np.isinf(points_gpd.y) | (points_gpd.x < x_low) \
            | (points_gpd.x > x_high) | (points_gpd.y < y_low) | (points_gpd.y > y_high))
    mask = np.logical_not(mask)
    times = times[mask]
    points_gpd = points_gpd[mask]
    post_traj_list = preprocess_filter_points_tdrive_with_time(points_gpd.x.values, points_gpd.y.values, times)
    a = points_gpd.x[np.isinf(points_gpd.x)].shape[0]
    if a > 0: print('points_gpd.x[np.isinf(points_gpd.x)].shape', a)
    b = points_gpd.y[np.isinf(points_gpd.y)].shape[0]
    if b > 0: print('points_gpd.y[np.isinf(points_gpd.y)].shape', b)
    # points_gpd = points_gpd[:160]
    # times = times[:160]
    # plot_one_traj(points_gpd.x, points_gpd.y, doNote=False)
    # for traj in post_traj_list:
    #     plot_one_traj(traj[:, 0], traj[:, 1], doNote=False)
    print('post preprocess', i, len(post_traj_list))
    return post_traj_list
    # # post_points = preprocess_cut_by_distance(post_points[:, 0], post_points[:, 1])

def plots_pkdd(post_traj_list, data_log_folder, file_prefix):
    # post_traj_list shape: batch x traj_len x 2
    # # plot trip length distribution
    # traj_length_list = [len(x) for x in post_traj_list]
    # plt.hist(traj_length_list, bins=100, range=[0, 1000])
    # plt.title('after dist-cut trip length distribution')
    # plt.show()

    # plot angles distribution
    angles = []
    preceding_segments = []
    succeeding_segments = []
    both_segments = []
    for traj in post_traj_list:
        angles.extend(list(get_angles(traj)))
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
    dist_thres, angle_thres = 166, 150 # 60 km/h
    print(preceding_segments)
    print(angles)
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
    h, _, _ = np.histogram2d(preceding_segments, angles, bins=[500, 360], range=[[0, 500], [0, 180]])
    plt.figure(figsize=(4, 3))
    # ax = plt.imshow(h, cmap='Blues')
    ax = plt.imshow(np.log10(h), cmap='Blues')
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

def plots_tdrive(post_traj_list, data_log_folder, file_prefix):
    # plot angles distribution
    angles = []
    preceding_segments = []
    succeeding_segments = []
    both_segments = []
    for traj in post_traj_list:
        angles.extend(list(get_angles(traj)))
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
    angle_thres = 10
    ratio = len(np.where((angles[:-1] < angle_thres) & (angles[1:] < angle_thres))[0]) / len(angles)
    print('''ratio that preceding and suceeding angles are both smaller than {} is {}'''.format(
        angle_thres, ratio))
    angle_thres = 30
    ratio = len(np.where((angles[:-1] < angle_thres) & (angles[1:] < angle_thres))[0]) / len(angles)
    print('''ratio that preceding and suceeding angles are both smaller than {} is {}'''.format(
        angle_thres, ratio))

    angle_thres = 10
    ratio = len(np.where((angles[:-2] < angle_thres) & (angles[1:-1] < angle_thres) \
                         & (angles[2:] < angle_thres))[0]) / len(angles)
    print('''ratio that three consecutive angles are smaller than {} is {}'''.format(
        angle_thres, ratio))
    angle_thres = 30
    ratio = len(np.where((angles[:-2] < angle_thres) & (angles[1:-1] < angle_thres) \
                         & (angles[2:] < angle_thres))[0]) / len(angles)
    print('''ratio that three consecutive angles are smaller than {} is {}'''.format(
        angle_thres, ratio))

    dist_thres, angle_thres = 166, 10 # 100 km/h
    ratio = len(np.where((preceding_segments > dist_thres) & (succeeding_segments > dist_thres) \
                         & (angles < angle_thres))[0]) / len(preceding_segments)
    print('''ratio that preceding and suceeding segments are both larger than {}
          and angle are smaller than {} is {}'''.format(dist_thres, angle_thres, ratio))
    dist_thres, angle_thres = 111, 10 # 40 km/h
    ratio = len(np.where((preceding_segments > dist_thres) & (succeeding_segments > dist_thres) \
                         & (angles < angle_thres))[0]) / len(preceding_segments)
    print('''ratio that preceding and suceeding segments are both larger than {}
          and angle are smaller than {} is {}'''.format(dist_thres, angle_thres, ratio))
    dist_thres, angle_thres = 111, 30 # 40 km/h
    ratio = len(np.where((preceding_segments > dist_thres) & (succeeding_segments > dist_thres) \
                         & (angles < angle_thres))[0]) / len(preceding_segments)
    print('''ratio that preceding and suceeding segments are both larger than {}
          and angle are smaller than {} is {}'''.format(dist_thres, angle_thres, ratio))

    # preceding angle v.s. angle
    h, _, _ = np.histogram2d(angles[:-1], angles[1:], bins=[360, 360], range=[[0, 180], [0, 180]])
    plt.figure(figsize=(4, 3))
    ax = plt.imshow(np.log10(h), cmap='Blues')
    plt.xticks([i*60 for i in range(7)], [str(i*30) for i in range(7)])
    plt.yticks([i*60 for i in range(7)], [str(i*30) for i in range(7)])
    plt.xlabel('Preceding angles')
    plt.ylabel('Succeeding angles')
    plt.colorbar(ax)
    plt.tight_layout()
    plt.savefig('{}/_angle_to_angle_constraint_{}.png'.format(data_log_folder, file_prefix), dpi=120)
    plt.close()

    # preceding segment v.s. angle
    h, _, _ = np.histogram2d(preceding_segments, angles, bins=[500, 900], range=[[0, 500], [0, 180]])
    plt.figure(figsize=(4, 3))
    # ax = plt.imshow(h, cmap='Blues')
    ax = plt.imshow(np.log10(h), cmap='Blues')
    plt.xticks([i * 150 for i in range(7)], [str(i * 30) for i in range(7)])
    plt.xlabel('Angles (1 degree bin)')
    plt.ylabel('Preceding segment lengths (10 meter bin)')
    plt.colorbar(ax)
    plt.tight_layout()
    plt.savefig('{}/_preseding_segment_to_angle_constraint_{}.png'.format(data_log_folder, file_prefix), dpi=120)
    plt.close()

    # succeeding segment v.s. angle
    h, _, _ = np.histogram2d(succeeding_segments, angles, bins=[500, 900], range=[[0, 500], [0, 180]])
    plt.figure(figsize=(4, 3))
    # ax = plt.imshow(h, cmap='Blues')
    ax = plt.imshow(np.log10(h), cmap='Blues')
    plt.xticks([i * 150 for i in range(7)], [str(i * 30) for i in range(7)])
    plt.xlabel('Angles')
    plt.ylabel('Succeeding segment lengths (10 meter bin)')
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
    plt.ylabel('Sum of both segment lengths (10 meter bin)')
    plt.colorbar(ax)
    plt.tight_layout()
    plt.savefig('{}/_both_segment_to_angle_constraint_{}.png'.format(data_log_folder, file_prefix), dpi=120)
    plt.close()

    # angles distribution
    plt.hist(angles, bins=90, range=[0, 180])
    plt.title('after dist-cut angle distribution')
    plt.tight_layout()
    plt.savefig('{}/_angle_constraint_{}.png'.format(data_log_folder, file_prefix), dpi=120)
    plt.close()

if __name__ == "__main__":
    # isTest = True
    isTest = False

    crs = "EPSG:4326"

    data_name = 'pkdd'
    # data_name = 'tdrive'
    # data_name = 'pol'
    # data_name = 'gowalla'

    if data_name == 'pkdd':
        data = pd.read_csv('/media/liming/Liming1/llm_porto_results/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015/train.csv')
        local_crs = "EPSG:3763"
        dist_thres_low = 10
        dist_thres_high = 1000

        # remove not moving points
        print('preprocessing {} data'.format(len(data)))
        if isTest: post_traj_file = 'dataset/{}/{}_post_traj_file_test.bin'.format(data_name, data_name)
        else: post_traj_file = 'dataset/{}/{}_post_traj_file.bin'.format(data_name, data_name)
        if not os.path.isfile(post_traj_file):
            pool = Pool()
            t0 = time.time()
            if isTest: tmp_list = pool.map(preprocess_pkdd, range(1000))
            else: tmp_list = pool.map(preprocess_pkdd, range(len(data)))
            print('use time:', time.time() - t0)
            post_traj_list = []
            for x in tmp_list: post_traj_list.extend(x)
            post_traj_list = [x for x in post_traj_list if len(x) > 0]
            with open(post_traj_file, 'wb') as f:
                pickle.dump(post_traj_list, f)
                f.close()
        else:
            print('load a saved trajectory file: {}'.format(post_traj_file))
            with open(post_traj_file, 'rb') as f:
                post_traj_list = pickle.load(f)
                f.close()

        # post_traj_list = []
        # # post_traj_list.extend(preprocess_pkdd(np.random.randint(1, len(data)+1)))
        # # post_traj_list.extend(preprocess_pkdd(1482810))
        # post_traj_list.extend(preprocess_pkdd(2472))
        data_log_folder = './logs/pkdd/'
        if not os.path.isdir(data_log_folder):
            os.mkdir(data_log_folder)
        plots_pkdd(post_traj_list, data_log_folder=data_log_folder, file_prefix='pkdd_spatial_validity_real')

        # save to fixed length trajectories for training
        traj_len = 32
        train_ratio = 0.9
        if isTest:
            train_traj_file = 'dataset/{}/{}_training_traj_file_test.bin'.format(data_name, data_name)
            test_traj_file = 'dataset/{}/{}_testing_traj_file_test.bin'.format(data_name, data_name)
            params_file = 'dataset/{}/{}_params_test.bin'.format(data_name, data_name)
        else:
            train_traj_file = 'dataset/{}/{}_training_traj_file.bin'.format(data_name, data_name)
            test_traj_file = 'dataset/{}/{}_testing_traj_file.bin'.format(data_name, data_name)
            params_file = 'dataset/{}/{}_params.bin'.format(data_name, data_name)
        if not os.path.isfile(params_file):
            used_traj_list = []
            for traj in post_traj_list:
                l = len(traj)
                if l >= traj_len:
                    for i in range(l // traj_len):
                        used_traj_list.append(traj[i * traj_len:(i + 1) * traj_len])
            used_traj_list = np.array(used_traj_list)

            # get the x y ranges
            offset = 1000
            scale = 1000
            x_min = used_traj_list[:, :, 0].min()
            x_max = used_traj_list[:, :, 0].max()
            y_min = used_traj_list[:, :, 1].min()
            y_max = used_traj_list[:, :, 1].max()
            print('before scaling x_min, x_max, y_min, y_max', x_min, x_max, y_min, y_max)
            params = {'offset': offset, 'scale': scale, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
                      'traj_len': traj_len}
            used_traj_list[:, :, 0] = (used_traj_list[:, :, 0] - x_min + offset) / scale
            used_traj_list[:, :, 1] = (used_traj_list[:, :, 1] - y_min + offset) / scale
            x_min = used_traj_list[:, :, 0].min()
            x_max = used_traj_list[:, :, 0].max()
            y_min = used_traj_list[:, :, 1].min()
            y_max = used_traj_list[:, :, 1].max()
            print('after scaling x_min, x_max, y_min, y_max', x_min, x_max, y_min, y_max)

            train_num = int(train_ratio * len(used_traj_list))
            np.random.shuffle(used_traj_list)
            train_traj_list = np.array(used_traj_list[:train_num])
            test_traj_list = np.array(used_traj_list[train_num:])

            with open(params_file, 'wb') as f:
                pickle.dump(params, f)
                f.close()
            with open(train_traj_file, 'wb') as f:
                pickle.dump(train_traj_list, f)
                f.close()
            with open(test_traj_file, 'wb') as f:
                pickle.dump(test_traj_list, f)
                f.close()
        else:
            print('load a saved train/test trajectory file: {} / {}'.format(train_traj_file, test_traj_file))
            with open(params_file, 'rb') as f:
                params = pickle.load(f)
                f.close()
            with open(train_traj_file, 'rb') as f:
                train_traj_list = pickle.load(f)
                f.close()
            with open(test_traj_file, 'rb') as f:
                test_traj_list = pickle.load(f)
                f.close()

        # # cut by angles
        # if isTest: post_traj_file = 'dataset/{}/{}_post_traj_cut_angle_file_test.bin'.format(data_name, data_name)
        # else: post_traj_file = 'dataset/{}/{}_post_traj_cut_angle_file.bin'.format(data_name, data_name)
        # if not os.path.isfile(post_traj_file):
        #     pool = Pool()
        #     t0 = time.time()
        #     if isTest: tmp_list = pool.map(preprocess_cut_traj, post_traj_list[:1000])
        #     else: tmp_list = pool.map(preprocess_cut_traj, post_traj_list)
        #     print('use time:', time.time() - t0)
        #     post_traj_list = []
        #     for x in tmp_list:
        #         if len(x) > 0:
        #             post_traj_list.extend(x)
        #     with open(post_traj_file, 'wb') as f:
        #         pickle.dump(post_traj_list, f)
        #         f.close()
        # else:
        #     print('load a saved trajectory file: {}'.format(post_traj_file))
        #     with open(post_traj_file, 'rb') as f:
        #         post_traj_list = pickle.load(f)
        #         f.close()

    elif data_name == 'tdrive':
        data_name = 'tdrive'
        num_users = 10357
        path = 'dataset/tdrive/taxi_log_2008_by_id/'
        # local_crs = "EPSG:21460"
        local_crs = "EPSG:4796"
        x_low = 387941
        x_high = 527941
        y_low = 4370000
        y_high = 4500000

        # dist_thres_low = 100
        # dist_thres_high = 1000
        # time_thres_high = 1200
        # speed_thres_low = 10
        # speed_thres_high = 80
        dist_thres_low = 30
        dist_thres_high = 10000
        time_thres_high = 600
        speed_thres_low = 10
        speed_thres_high = 80
        # preprocess_tdrive(100)
        print('preprocessing {} data'.format(num_users))
        if isTest: post_traj_file = 'dataset/{}/{}_post_traj_file_test.bin'.format(data_name, data_name)
        else: post_traj_file = 'dataset/{}/{}_post_traj_file.bin'.format(data_name, data_name)
        if not os.path.isfile(post_traj_file):
            pool = Pool()
            t0 = time.time()
            if isTest: tmp_list = pool.map(preprocess_tdrive, range(1, 4))
            else: tmp_list = pool.map(preprocess_tdrive, range(1, num_users+1))
            print('use time:', time.time() - t0)
            post_traj_list = []
            for x in tmp_list:
                post_traj_list.extend(x)
            with open(post_traj_file, 'wb') as f:
                pickle.dump(post_traj_list, f)
                f.close()
        else:
            print('load a saved trajectory file: {}'.format(post_traj_file))
            with open(post_traj_file, 'rb') as f:
                post_traj_list = pickle.load(f)
                f.close()

        # post_traj_list = []
        # # post_traj_list.extend(preprocess_tdrive(np.random.randint(1, num_users)))
        # # post_traj_list.extend(preprocess_tdrive(2976))
        # post_traj_list.extend(preprocess_tdrive(141))
        # post_traj_list.extend(preprocess_tdrive(2472))

        plots_tdrive(post_traj_list, data_log_folder='./logs/tdrive/', file_prefix='tdrive_spatial_validity_real')

        # save to fixed length trajectories for training
        traj_len = 32
        train_ratio = 0.9
        if isTest:
            train_traj_file = 'dataset/{}/{}_training_traj_file_test.bin'.format(data_name, data_name)
            test_traj_file = 'dataset/{}/{}_testing_traj_file_test.bin'.format(data_name, data_name)
            params_file = 'dataset/{}/{}_params_test.bin'.format(data_name, data_name)
        else:
            train_traj_file = 'dataset/{}/{}_training_traj_file.bin'.format(data_name, data_name)
            test_traj_file = 'dataset/{}/{}_testing_traj_file.bin'.format(data_name, data_name)
            params_file = 'dataset/{}/{}_params.bin'.format(data_name, data_name)
        if not os.path.isfile(params_file):
            used_traj_list = []
            for traj in post_traj_list:
                l = len(traj)
                if l >= traj_len:
                    for i in range(0, l - traj_len + 1, traj_len // 2):
                        used_traj_list.append(traj[i:i + traj_len])
                    # for i in range(l - traj_len + 1):
                    #     used_traj_list.append(traj[i * traj_len:(i + 1) * traj_len])
            used_traj_list = np.array(used_traj_list)
            print(used_traj_list[-1].shape)

            # get the x y ranges
            offset = 1000
            scale = 1000
            x_min = used_traj_list[:, :, 0].min()
            x_max = used_traj_list[:, :, 0].max()
            y_min = used_traj_list[:, :, 1].min()
            y_max = used_traj_list[:, :, 1].max()
            print('before scaling x_min, x_max, y_min, y_max', x_min, x_max, y_min, y_max)
            params = {'offset':offset, 'scale':scale, 'x_min':x_min, 'x_max':x_max, 'y_min':y_min, 'y_max':y_max, 'traj_len':traj_len}
            used_traj_list[:,:,0] = (used_traj_list[:,:,0] - x_min + offset) / scale
            used_traj_list[:,:,1] = (used_traj_list[:,:,1] - y_min + offset) / scale
            x_min = used_traj_list[:, :, 0].min()
            x_max = used_traj_list[:, :, 0].max()
            y_min = used_traj_list[:, :, 1].min()
            y_max = used_traj_list[:, :, 1].max()
            print('after scaling x_min, x_max, y_min, y_max', x_min, x_max, y_min, y_max)

            train_num = int(train_ratio * len(used_traj_list))
            np.random.shuffle(used_traj_list)
            train_traj_list = np.array(used_traj_list[:train_num])
            test_traj_list = np.array(used_traj_list[train_num:])

            with open(params_file, 'wb') as f:
                pickle.dump(params, f)
                f.close()
            with open(train_traj_file, 'wb') as f:
                pickle.dump(train_traj_list, f)
                f.close()
            with open(test_traj_file, 'wb') as f:
                pickle.dump(test_traj_list, f)
                f.close()
        else:
            print('load a saved train/test trajectory file: {} / {}'.format(train_traj_file, test_traj_file))
            with open(params_file, 'rb') as f:
                params = pickle.load(f)
                f.close()
            with open(train_traj_file, 'rb') as f:
                train_traj_list = pickle.load(f)
                f.close()
            with open(test_traj_file, 'rb') as f:
                test_traj_list = pickle.load(f)
                f.close()
        # # cut by angles
        # if isTest: post_traj_file = 'dataset/{}/{}_post_traj_cut_angle_file_test.bin'.format(data_name, data_name)
        # else: post_traj_file = 'dataset/{}/{}_post_traj_cut_angle_file.bin'.format(data_name, data_name)
        # if not os.path.isfile(post_traj_file):
        #     pool = Pool()
        #     t0 = time.time()
        #     tmp_list = pool.map(preprocess_cut_traj, post_traj_list)
        #     print('use time:', time.time() - t0)
        #     post_traj_list = []
        #     for x in tmp_list:
        #         if len(x) > 0:
        #             post_traj_list.extend(x)
        #     with open(post_traj_file, 'wb') as f:
        #         pickle.dump(post_traj_list, f)
        #         f.close()
        # else:
        #     print('load a saved trajectory file: {}'.format(post_traj_file))
        #     with open(post_traj_file, 'rb') as f:
        #         post_traj_list = pickle.load(f)
        #         f.close()
    elif data_name == 'pol':
        if isTest:
            train_traj_file = 'dataset/{}/{}_training_traj_file_test.bin'.format(data_name, data_name)
            test_traj_file = 'dataset/{}/{}_testing_traj_file_test.bin'.format(data_name, data_name)
            params_file = 'dataset/{}/{}_params_test.bin'.format(data_name, data_name)
        else:
            train_traj_file = 'dataset/{}/{}_training_traj_file.bin'.format(data_name, data_name)
            test_traj_file = 'dataset/{}/{}_testing_traj_file.bin'.format(data_name, data_name)
            params_file = 'dataset/{}/{}_params.bin'.format(data_name, data_name)
        if not os.path.isfile(params_file):
            path = 'dataset/pol/pol_checkin.tsv'
            polyline_col = 'POLYLINE'
            ID_col = 'ID'
            X_col = 'X'
            Y_col = 'Y'
            df = pd.read_csv(path, delimiter='\t', usecols=['UserId', 'CheckinTime', 'X', 'Y'], parse_dates=['CheckinTime'])
            df.columns = [ID_col, 'CheckinTime', X_col, Y_col]

            num_users = np.unique(df[ID_col]).shape[0]
            print('preprocessing {} data'.format(num_users))

            traj_len = 32
            train_ratio = 0.9
            # get the x y ranges
            offset = 0
            scale = 100
            x_min, x_max = df[X_col].min(), df[X_col].max()
            y_min, y_max = df[Y_col].min(), df[Y_col].max()
            print('before scaling x_min, x_max, y_min, y_max', x_min, x_max, y_min, y_max)
            params = {'offset': offset, 'scale': scale, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
                      'traj_len': traj_len}
            df[X_col] = (df[X_col] - df[X_col].min()) / scale
            df[Y_col] = (df[Y_col] - df[Y_col].min()) / scale
            x_min, x_max = df[X_col].min(), df[X_col].max()
            y_min, y_max = df[Y_col].min(), df[Y_col].max()
            print('after scaling x_min, x_max, y_min, y_max', x_min, x_max, y_min, y_max)

            used_traj_list = []
            for user in range(num_users):
                user_checkins = df.query('{} == {}'.format(ID_col, user))
                user_total_traj = user_checkins[[X_col, Y_col]].values
                for i in range(0, len(user_total_traj) - traj_len + 1, traj_len):
                    used_traj_list.append(user_total_traj[i:i+traj_len])

            train_num = int(train_ratio * len(used_traj_list))
            np.random.shuffle(used_traj_list)
            train_traj_list = np.array(used_traj_list[:train_num])
            test_traj_list = np.array(used_traj_list[train_num:])
            with open(params_file, 'wb') as f:
                pickle.dump(params, f)
                f.close()
            with open(train_traj_file, 'wb') as f:
                pickle.dump(train_traj_list, f)
                f.close()
            with open(test_traj_file, 'wb') as f:
                pickle.dump(test_traj_list, f)
                f.close()
        else:
            print('load a saved train/test trajectory file: {} / {}'.format(train_traj_file, test_traj_file))
            with open(params_file, 'rb') as f:
                params = pickle.load(f)
                f.close()
            with open(train_traj_file, 'rb') as f:
                train_traj_list = pickle.load(f)
                f.close()
            with open(test_traj_file, 'rb') as f:
                test_traj_list = pickle.load(f)
                f.close()
    elif data_name == 'gowalla':
        if isTest:
            train_traj_file = 'dataset/{}/{}_training_traj_file_test.bin'.format(data_name, data_name)
            test_traj_file = 'dataset/{}/{}_testing_traj_file_test.bin'.format(data_name, data_name)
            params_file = 'dataset/{}/{}_params_test.bin'.format(data_name, data_name)
        else:
            train_traj_file = 'dataset/{}/{}_training_traj_file.bin'.format(data_name, data_name)
            test_traj_file = 'dataset/{}/{}_testing_traj_file.bin'.format(data_name, data_name)
            params_file = 'dataset/{}/{}_params.bin'.format(data_name, data_name)
        if not os.path.isfile(params_file):
            path = 'dataset/gowalla/loc-gowalla_totalCheckins.txt'
            polyline_col = 'POLYLINE'
            ID_col = 'ID'
            X_col = 'X'
            Y_col = 'Y'
            df = pd.read_csv(path, delimiter='\t', usecols=[0, 1, 2, 3], parse_dates=[1])
            df.columns = [ID_col, 'CheckinTime', X_col, Y_col]
            # df.columns = [ID_col, 'CheckinTime', Y_col, X_col]
            # San Antonio: 29.429939, -98.502053; Austin: 30.239107, -97.734383; Dallas: 32.915533, -96.756600; Houston: 29.811067, -94.812020
            x_min, x_max = 28, 33
            y_min, y_max = -100, -95
            df = df[(df[X_col] > x_min) & (df[X_col] < x_max) & (df[Y_col] > y_min) & (df[Y_col] < y_max)]
            # x_min, x_max = 28, 30
            # y_min, y_max = -100, -97
            # df = df[(df[X_col] > 28) & (df[X_col] < 30) & (df[Y_col] > -100) & (df[Y_col] < -97)]
            # df['time'] = (df['CheckinTime'].dt.hour * 60 + df['CheckinTime'].dt.minute) / (24 * 60)

            # local_crs = "EPSG:32039"
            # points_gpd = gpd.GeoSeries(gpd.points_from_xy(df.loc[:, X_col], df.loc[:, Y_col]), crs=crs)
            # points_gpd = points_gpd.to_crs(local_crs)
            # print('points_gpd.x', points_gpd.x)
            # df[X_col] = points_gpd.x
            # df[Y_col] = points_gpd.y

            unique_users = np.unique(df[ID_col])
            print('preprocessing {} users'.format(len(unique_users)))

            traj_len = 16
            train_ratio = 0.9
            # get the x y ranges
            # offset = 1000
            # scale = 1000
            offset = 0
            scale = 1.11e-3
            # x_min, x_max = df[X_col].min(), df[X_col].max()
            # y_min, y_max = df[Y_col].min(), df[Y_col].max()
            print('before scaling x_min, x_max, y_min, y_max', x_min, x_max, y_min, y_max)
            params = {'offset': offset, 'scale': scale, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
                      'traj_len': traj_len}
            df[X_col] = (df[X_col] - df[X_col].min()) / scale
            df[Y_col] = (df[Y_col] - df[Y_col].min()) / scale
            x_min, x_max = df[X_col].min(), df[X_col].max()
            y_min, y_max = df[Y_col].min(), df[Y_col].max()
            print('after scaling x_min, x_max, y_min, y_max', x_min, x_max, y_min, y_max)

            used_traj_list = []
            for user in unique_users:
                user_checkins = df.query('{} == {}'.format(ID_col, user))
                user_total_traj = user_checkins[[X_col, Y_col]].values
                for i in range(0, len(user_total_traj) - traj_len + 1, traj_len):
                    used_traj_list.append(user_total_traj[i:i+traj_len])

            train_num = int(train_ratio * len(used_traj_list))
            np.random.shuffle(used_traj_list)
            train_traj_list = np.array(used_traj_list[:train_num])
            test_traj_list = np.array(used_traj_list[train_num:])
            with open(params_file, 'wb') as f:
                pickle.dump(params, f)
                f.close()
            with open(train_traj_file, 'wb') as f:
                pickle.dump(train_traj_list, f)
                f.close()
            with open(test_traj_file, 'wb') as f:
                pickle.dump(test_traj_list, f)
                f.close()
        else:
            print('load a saved train/test trajectory file: {}  {}'.format(train_traj_file, test_traj_file))
            with open(params_file, 'rb') as f:
                params = pickle.load(f)
                f.close()
            with open(train_traj_file, 'rb') as f:
                train_traj_list = pickle.load(f)
                f.close()
            with open(test_traj_file, 'rb') as f:
                test_traj_list = pickle.load(f)
                f.close()
    print('train_traj_list.shape', train_traj_list.shape)
    print('test_traj_list.shape', test_traj_list.shape)
    print('params', params)

    x_min = 0
    x_max = (params['x_max'] - params['x_min'] + params['offset']) / params['scale']
    x_max = int(x_max)
    y_min = 0
    y_max = (params['y_max'] - params['y_min'] + params['offset']) / params['scale']
    y_max = int(y_max)
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)
    print('x_range, y_range', x_range, y_range)
    if data_name == 'pkdd':
        n_per_bin = 10
        x_bins_ = x_max * n_per_bin
        y_bins_ = y_max * n_per_bin
        n_subset = 20000
        x_tick_range = [60 * n_per_bin, 80 * n_per_bin]
        y_tick_range = [440 * n_per_bin, 470 * n_per_bin]
    elif data_name == 'tdrive':
        n_per_bin = 2
        x_bins_ = x_max * n_per_bin
        y_bins_ = y_max * n_per_bin
        # n_subset = len(self.test)
        # n_subset = 10000
        x_tick_range = [0, x_bins_]
        y_tick_range = [0, y_bins_]
        # x_tick_range = [15800 * n_per_bin, 16200 * n_per_bin]
        # y_tick_range = [4250 * n_per_bin, 4600 * n_per_bin]
    elif data_name == 'gowalla':
        n_per_bin = 0.1
        x_bins_ = int(x_max * n_per_bin)
        y_bins_ = int(y_max * n_per_bin)
        n_subset = len(train_traj_list)
        # n_subset = 10000
        # x_tick_range = [0, x_bins_]
        # y_tick_range = [0, y_bins_]
        x_tick_range = [1900 * n_per_bin, 2300 * n_per_bin]
        y_tick_range = [1800 * n_per_bin, 2100 * n_per_bin]
    else:
        n_per_bin = 1
        x_bins_ = x_max * n_per_bin
        y_bins_ = y_max * n_per_bin
        n_subset = len(train_traj_list)
        # n_subset = 10000
        x_tick_range = [0, x_bins_]
        y_tick_range = [0, y_bins_]
        # x_tick_range = [750 * n_per_bin, 950 * n_per_bin]
        # y_tick_range = [850 * n_per_bin, 1050 * n_per_bin]
    print('x_range, y_range, x_bins_, y_bins_', x_range, y_range, x_bins_, y_bins_)
    orig_2Dhist = compute_2Dhist_numpy(train_traj_list.reshape(-1, 2), x_range, y_range, x_bins_, y_bins_)
    # orig_2Dhist = compute_2Dhist_numpy(test_traj_list.reshape(-1, 2), x_range, y_range, x_bins_, y_bins_)
    print(np.where(orig_2Dhist == orig_2Dhist.max()))
    print('orig_2Dhist', orig_2Dhist.min(), orig_2Dhist.max())

    # 2d histgram
    file_name = 'dataset/{}/_2dHistgram_dist_{}'.format(data_name, 'testset_real')
    print('save 2d histgram point count distribution to', file_name)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(np.log10(orig_2Dhist.T), cmap='Blues')
    # im = ax.imshow(orig_2Dhist.T, cmap='Blues')
    ax.set_xlabel('X ranged in [{}, {}]'.format(x_tick_range[0], x_tick_range[1]))
    ax.set_ylabel('Y ranged in [{}, {}]'.format(y_tick_range[0], y_tick_range[1]))
    ax.set_xlim(x_tick_range)
    ax.set_ylim(y_tick_range)
    cbar = ax.figure.colorbar(im, ax=ax, cmap="Blues")
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    # plt.savefig(file_name, dpi=120)
    # plt.close()
    plt.show()
    print('finished execution!')