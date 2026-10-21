"""Step 3 - lift the per-pixel features of step 2 onto the scene point cloud ("best view").

Every 3D point of ``<parsed_dir>/<scene>/pointcloud/scene_pc.npy`` that is visible in a projection frame takes
the feature of the frame in which it is closest to the ego head (``full_body_pose[:, 6]``). Features are
de-duplicated into a codebook (cosine similarity > ``--sim_threshold`` reuses an existing entry):

    <out_dir>/<scene>/final_pc_data_<last_frame>frames.npz   clip_feat_index (N,), dist (N,), frame_updated (N,)
    <out_dir>/<scene>/final_clip_feat_<last_frame>frames.pt  (K, 1408) codebook
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

IMG_SIZE = (1408, 1408)   # resolution of the parsed egocentric frames
FEAT_SIZE = (512, 512)    # resolution of the step-2 feature maps
HEAD_JOINT = 6


def scale_pixel_coords(pixel_coords, orig_size, target_size=(512, 512)):
    """(N, 2) pixel coordinates from orig_size to target_size, rounded to int."""
    if isinstance(pixel_coords, np.ndarray):
        pixel_coords = torch.from_numpy(pixel_coords)
    H_raw, W_raw = orig_size
    H_tgt, W_tgt = target_size
    x = pixel_coords[:, 0] * (W_tgt / W_raw)
    y = pixel_coords[:, 1] * (H_tgt / H_raw)
    return torch.stack([x, y], dim=1).round().long()


def feature_at(feature_map, scaled_coords, device, dtype=torch.float16):
    """Feature at a (row, col) location; zeros outside the map."""
    H, W, D = feature_map.shape
    y, x = map(int, scaled_coords[0])
    if not (0 <= x < W and 0 <= y < H):
        return torch.zeros((D,), dtype=dtype, device=device)
    return feature_map[y, x]


def lift_frame(frame, feature_map, codebook, pc, pc_data, projected_points, ego_head, device, sim_threshold=0.999):
    """Update pc_data / codebook with the points of one frame that are closer to the head than before."""
    feature_map = feature_map.to(device=device, dtype=torch.float32)
    H, W, _ = feature_map.shape

    for idx_projected, idx_point in enumerate(projected_points['indices']):
        dist = np.linalg.norm(ego_head - pc[idx_point])
        pixel_coords = np.array([tuple(map(int, projected_points["pts_2d"][idx_projected]))])  # (row, col)
        scaled_coords = scale_pixel_coords(pixel_coords, IMG_SIZE, (H, W))

        if dist <= pc_data["dist"][idx_point]:
            pc_data["dist"][idx_point] = dist
            feat = feature_at(feature_map, scaled_coords, device).to(device=device, dtype=torch.float32)
            if torch.all(feat == 0):
                continue

            if codebook is None:
                codebook = feat.unsqueeze(0)
                matched_index = 0
            else:
                sims = F.cosine_similarity(codebook, feat.unsqueeze(0).expand_as(codebook), dim=1)
                best_sim, best_key = sims.max(dim=0)
                if float(best_sim) > sim_threshold:
                    matched_index = int(best_key)
                else:
                    matched_index = codebook.shape[0]
                    codebook = torch.cat([codebook, feat.unsqueeze(0)], dim=0)

            pc_data["clip_feat_index"][idx_point] = matched_index
            pc_data["frame_updated"][idx_point] = int(frame)

    return pc_data, codebook


def main():
    parser = argparse.ArgumentParser(description="Lift per-pixel features onto the scene point cloud")
    parser.add_argument("--parsed_dir", required=True, help="root of the parsed Nymeria scenes")
    parser.add_argument("--scene", required=True, help="scene name")
    parser.add_argument("--feat_dir", required=True, help="output root of step 2")
    parser.add_argument("--out_dir", required=True, help="results are written to <out_dir>/<scene>/")
    parser.add_argument("--sim_threshold", type=float, default=0.999)
    parser.add_argument("--frames", type=int, nargs="+", default=None, help="subset of projection frame keys")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_dir = os.path.join(args.parsed_dir, args.scene)
    feat_dir = os.path.join(args.feat_dir, args.scene)
    save_dir = os.path.join(args.out_dir, args.scene)

    projected = np.load(os.path.join(scene_dir, 'projection', 'projections.npy'), allow_pickle=True).item()
    ego_head = np.load(os.path.join(scene_dir, 'full_body_pose', 'full_body_pose.npy'))[:, HEAD_JOINT, :]  # (T, 3)
    pc = np.load(os.path.join(scene_dir, 'pointcloud', 'scene_pc.npy'))  # (N, 3)
    N = pc.shape[0]
    pc_data = {
        "clip_feat_index": np.full((N,), -1, dtype=np.int64),
        "dist": np.full((N,), np.inf, dtype=np.float32),
        "frame_updated": np.full((N,), -1, dtype=np.int64),
    }

    frames = sorted(projected.keys())
    if args.frames is not None:
        frames = [f for f in frames if f in set(args.frames)]

    codebook = None
    for frame in tqdm(frames, desc=args.scene):
        feat_path = os.path.join(feat_dir, f"{frame:04d}.pt")
        if not os.path.exists(feat_path):
            print(f'[skip] no features for frame {frame}')
            continue
        pc_data, codebook = lift_frame(frame, torch.load(feat_path), codebook, pc, pc_data, projected[frame],
                                       ego_head[frame], device, args.sim_threshold)

    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, f"final_pc_data_{frame:04d}frames.npz"), **pc_data)
    torch.save(codebook, os.path.join(save_dir, f"final_clip_feat_{frame:04d}frames.pt"))
    print(f"{args.scene}: {int((pc_data['clip_feat_index'] >= 0).sum())}/{N} points, codebook {len(codebook)}")


if __name__ == "__main__":
    main()
