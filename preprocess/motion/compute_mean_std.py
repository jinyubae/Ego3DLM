"""Compute mean.npy / std.npy of the 279-D motion features.

The standard deviation is averaged within each feature group (root velocity, root rotation,
root angular velocity, joint rotations, joint angular velocities), as in HumanML3D.

Usage:
    python -m preprocess.motion.compute_mean_std --feat_dir <dataset>/new_joint_vecs --out_dir <dataset>
"""
import argparse
import os
from os.path import join as pjoin

import numpy as np


def mean_variance(data_dir, save_dir, joints_num):
    data_list = []
    for file in os.listdir(data_dir):
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(f'[skip] NaN in {file}')
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[0:3] = std[0:3].mean()
    std[3:9] = std[3:9].mean()
    std[9:15] = std[9:15].mean()
    std[15: 15 + (joints_num - 1) * 6] = std[15: 15 + (joints_num - 1) * 6].mean()
    std[15 + (joints_num - 1) * 6:] = std[15 + (joints_num - 1) * 6:].mean()
    assert 15 + (joints_num - 1) * 12 == std.shape[-1]

    np.save(pjoin(save_dir, 'mean.npy'), mean)
    np.save(pjoin(save_dir, 'std.npy'), std)
    return mean, std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', required=True, help='directory of <id>.npy motion features')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--joints_num', type=int, default=23)
    args = parser.parse_args()
    mean_variance(args.feat_dir, args.out_dir, args.joints_num)
