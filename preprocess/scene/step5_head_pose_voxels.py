"""Step 5 - head poses of every sample in the voxel grid of its scene.

For a sample id (``scene_info_with_scene_index.txt``: ``id,scene,segment``) the head poses
``<parsed_dir>/<scene>/head_pose/<segment>.npy`` (T+1, 4, 4) are cut to the T frames of the motion features,
permuted to the y-up frame ((x, y, z) -> (y, z, x)) and quantized with the voxel grid of step 4:

    <out_dir>/<id>.npz   voxel (T, 3) int32, rotation (T, 3, 3) float32, pose_yzx (T, 4, 4) float32

The model aligns the scene to the head pose of the last observed frame on the fly.
"""
import argparse
import os
from pathlib import Path

import numpy as np

R = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
    ], dtype=np.float32)


def rotate_pose_matrix(T, R):
    out = T.copy()
    out[..., :3, :3] = R.T[None, ...] @ out[..., :3, :3]
    out[..., :3, 3] = out[..., :3, 3] @ R
    return out


def voxelize_positions(positions, min_coords, voxel_size):
    size = np.asarray(voxel_size, dtype=np.float32)
    if size.ndim == 0:
        size = np.full(3, float(size), dtype=np.float32)
    origin = np.asarray(min_coords, dtype=np.float32)
    return np.round((positions - origin[None, :]) / size[None, :]).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Head poses in the scene voxel grid")
    parser.add_argument("--parsed_dir", required=True, help="root of the parsed Nymeria scenes")
    parser.add_argument("--scene_index", required=True, help="scene_info_with_scene_index.txt (id,scene,segment)")
    parser.add_argument("--voxel_dir", required=True, help="voxelized_voxels/ of step 4")
    parser.add_argument("--out_dir", required=True, help="output dir (global_head_poses/)")
    parser.add_argument("--ids", default=None, help="optional file whose first column restricts the ids")
    args = parser.parse_args()

    index = [l.strip().split(',') for l in open(args.scene_index) if l.strip()]
    if args.ids:
        keep = {l.strip().split(',')[0] for l in open(args.ids) if l.strip()}
        index = [e for e in index if e[0] in keep]

    os.makedirs(args.out_dir, exist_ok=True)
    grids, processed, skipped = {}, 0, 0
    for sid, scene, segment in index:
        if scene not in grids:
            voxel_file = Path(args.voxel_dir) / f'{scene}.npz'
            if not voxel_file.exists():
                grids[scene] = None
            else:
                with np.load(voxel_file) as d:
                    grids[scene] = (d['min_coords'], d['voxel_size'])
        if grids[scene] is None:
            skipped += 1
            continue

        poses = np.load(os.path.join(args.parsed_dir, scene, 'head_pose', f'{segment}.npy'))[:-1]
        poses = rotate_pose_matrix(poses, R)
        min_coords, voxel_size = grids[scene]
        np.savez_compressed(
            os.path.join(args.out_dir, f'{sid}.npz'),
            voxel=voxelize_positions(poses[:, :3, 3].astype(np.float32), min_coords, voxel_size),
            rotation=poses[:, :3, :3].astype(np.float32),
            pose_yzx=poses.astype(np.float32),
        )
        processed += 1
    print(f'processed={processed}, skipped (no scene voxels)={skipped}')


if __name__ == '__main__':
    main()
