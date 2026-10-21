"""Step 4 - voxelize the lifted point features into the sparse scene representation used by Ego3DLM.

The points with a feature are permuted to the y-up frame used by the motion data ((x, y, z) -> (y, z, x)),
quantized into 0.15 m voxels and the features of each voxel are averaged:

    <out_dir>/voxelized_voxels/<scene>.npz   voxel (M, 3) int32, min_coords (3,), voxel_size
    <out_dir>/voxelized_features/<scene>.pt  (M, 1408) float32

The scene stays in the (permuted) world frame; the alignment to the ego frame of each sample is done on the fly
by the model from the head pose of the sample (see step 5).
"""
import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# (x, y, z) -> (y, z, x), same permutation as for the motion and the head poses
R = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
    ], dtype=np.float32)


def voxelize_sparse_mean(points, clip_feat_index, clip_features, voxel_size, min_points_per_voxel=1, ignore_index=-1):
    """Sparse voxelization with feature averaging.

    Returns voxel_coords (M, 3) int32, voxel_feats (M, D) float32 and min_coord (3,).
    Voxels are ordered by the first point that falls into them.
    """
    clip_feat_index = np.asarray(clip_feat_index, dtype=np.int64)
    assert points.shape[0] == clip_feat_index.shape[0]
    voxel_size = np.array([voxel_size] * 3)

    min_coord = points.min(axis=0)
    max_coord = points.max(axis=0)
    grid_shape = np.ceil((max_coord - min_coord) / voxel_size).astype(int)

    voxel_coords_all = np.floor((points - min_coord) / voxel_size).astype(int)
    valid_mask = np.all((voxel_coords_all >= 0) & (voxel_coords_all < grid_shape), axis=1)
    voxel_coords_all = voxel_coords_all[valid_mask]
    indices_all = clip_feat_index[valid_mask]

    vox2idx = defaultdict(list)
    for coord, idx in zip(map(tuple, voxel_coords_all), indices_all):
        vox2idx[coord].append(int(idx))

    K = clip_features.shape[0]
    sparse_coords, sparse_feats = [], []
    for vox, idx_list in vox2idx.items():
        if len(idx_list) < min_points_per_voxel:
            continue
        valid = [i for i in idx_list if (i != ignore_index) and (0 <= i < K)]
        if len(valid) == 0:
            continue
        sel = torch.as_tensor(valid, dtype=torch.long, device=clip_features.device)
        sparse_coords.append(vox)
        sparse_feats.append(clip_features[sel].mean(dim=0).detach().cpu().numpy().astype(np.float32))

    if len(sparse_coords) == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, clip_features.shape[1]), dtype=np.float32), min_coord
    return np.asarray(sparse_coords, dtype=np.int32), np.stack(sparse_feats, axis=0).astype(np.float32), min_coord


def main():
    parser = argparse.ArgumentParser(description="Voxelize lifted point features")
    parser.add_argument("--parsed_dir", required=True, help="root of the parsed Nymeria scenes (point clouds)")
    parser.add_argument("--proj_dir", required=True, help="output root of step 3")
    parser.add_argument("--scene", required=True, help="scene name")
    parser.add_argument("--out_dir", required=True, help="writes <out_dir>/voxelized_{voxels,features}/<scene>")
    parser.add_argument("--voxel_size", type=float, default=0.15)
    parser.add_argument("--min_points", type=int, default=1)
    args = parser.parse_args()

    scene_pc = np.load(os.path.join(args.parsed_dir, args.scene, 'pointcloud', 'scene_pc.npy'))
    proj_dir = Path(args.proj_dir) / args.scene
    pc_proj = np.load(sorted(proj_dir.glob("final_*.npz"))[0])
    clip_feat = torch.load(sorted(proj_dir.glob("final_*.pt"))[0], map_location='cpu')

    clip_feat_index = pc_proj['clip_feat_index']
    valid_mask = clip_feat_index != -1
    pc = scene_pc[valid_mask] @ R
    clip_feat_index = clip_feat_index[valid_mask]

    voxels, feats, min_coord = voxelize_sparse_mean(pc, clip_feat_index, clip_feat, args.voxel_size,
                                                    min_points_per_voxel=args.min_points, ignore_index=-1)

    voxel_path = os.path.join(args.out_dir, 'voxelized_voxels', f'{args.scene}.npz')
    feat_path = os.path.join(args.out_dir, 'voxelized_features', f'{args.scene}.pt')
    os.makedirs(os.path.dirname(voxel_path), exist_ok=True)
    os.makedirs(os.path.dirname(feat_path), exist_ok=True)
    np.savez(voxel_path, voxel=voxels, min_coords=min_coord, voxel_size=args.voxel_size)
    torch.save(torch.from_numpy(feats), feat_path)
    print(f"{args.scene}: {voxels.shape[0]} voxels")


if __name__ == "__main__":
    main()
