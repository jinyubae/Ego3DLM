"""Compute the Ego3DLM motion representation from parsed Nymeria segments.

For every narrated segment (``<parsed_dir>/<scene>/body_pose/NNNN.npy``, 23 Xsens joints at 10 FPS)
this script writes

    <out_dir>/new_joint_vecs/<scene>/NNNN.npy   (T-1, 279) motion features
    <out_dir>/new_joints/<scene>/NNNN.npy       (T, 23, 3) joints recovered from the features
    <out_dir>/three_points/<scene>/NNNN.npy     (T, 3, 4, 4) head / left wrist / right wrist poses
    <out_dir>/skel_offsets.npy                  (23, 3) target skeleton offsets

The 279-D feature is
    root linear velocity (3) | root 6D rotation (6) | root 6D angular velocity (6)
    | joint 6D rotations (22*6) | joint 6D angular velocities (22*6).

Every sequence is retargeted to a common skeleton, put on the floor, moved to the XZ origin and rotated
to face +Z at its first frame. The three-point poses are expressed in the same canonical frame.

Usage:
    python -m preprocess.motion.motion_representation \
        --parsed_dir /path/to/parsed_nymeria --out_dir /path/to/motion_scene \
        --skel_offsets datasets/ego3dlm/skel_offsets.npy
"""
import argparse
import os
from os.path import join as pjoin

import numpy as np
import torch
from tqdm import tqdm

from .common.quaternion import (cont6d_to_quat, cont6d_to_quat_np, qbetween_np, qinv, qinv_np, qmul, qmul_np,
                                qrot, qrot_np, quaternion_to_cont6d, quaternion_to_cont6d_np, quaternion_to_matrix)
from .common.skeleton import Skeleton
from .common.param_util_xsens import KINEMATIC_CHAIN, RAW_OFFSETS

# Xsens joint indices
l_idx1, l_idx2 = 16, 17                  # lower-leg joints used for the scale ratio
face_joint_indx = [15, 19, 7, 11]        # hips and shoulders used for the facing direction
HEAD, LEFT_WRIST, RIGHT_WRIST = 6, 14, 10
joints_num = 23

# Nymeria (z-up) -> SMPL-style (y-up) axis permutation: (x, y, z) -> (y, z, x)
R = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
    ], dtype=np.float32)

n_raw_offsets = torch.from_numpy(RAW_OFFSETS)
kinematic_chain = KINEMATIC_CHAIN


def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    # scale ratio from the leg length
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len

    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    # inverse kinematics on the source skeleton, forward kinematics on the target skeleton
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints, scale_rt


def process_file(positions, T34, tgt_offsets):
    """positions: (T, 23, 3) joints, T34: (T, 3, 4, 4) head/wrist poses, both already in the y-up frame."""
    positions, scale_rt = uniform_skeleton(positions, tgt_offsets)

    # put on floor
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # XZ at origin
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # all sequences initially face +Z
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)
    R_world = quaternion_to_matrix(torch.from_numpy(root_quat_init))

    # three-point poses in the canonical frame (translation = canonical head / wrist joints)
    R_world_tps = R_world[:, [HEAD, LEFT_WRIST, RIGHT_WRIST], ...]
    three_points_translation = positions[:, [HEAD, LEFT_WRIST, RIGHT_WRIST], :]
    y0 = float(positions[0, 0, 1])
    if not torch.is_tensor(T34):
        T34 = torch.from_numpy(T34)
    T34_out = T34.clone()
    T34_out[..., :3, 3] = torch.as_tensor(three_points_translation, dtype=T34_out.dtype, device=T34_out.device)
    T34_out[..., 1, 3] -= y0
    T34_out[..., :3, :3] = R_world_tps @ T34_out[..., :3, :3]

    global_positions = positions.copy()

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()
        # root linear velocity (T-1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        # root angular velocity (T-1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    r_6d_velocity = quaternion_to_cont6d_np(r_velocity)

    root_6d_rotation = cont_6d_params[:, 0]
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    # joint angular velocity as a relative rotation between consecutive frames
    j_quat = cont6d_to_quat_np(cont_6d_params[:, 1:])
    rot_velocity = qmul_np(j_quat[1:], qinv_np(j_quat[:-1]))
    rot_6d_velocity = quaternion_to_cont6d_np(rot_velocity)
    rot_6d_velocity = rot_6d_velocity.reshape(len(rot_6d_velocity), -1)

    data = np.concatenate([
        velocity,                  # (T-1, 3)     root translation velocity
        root_6d_rotation[:-1],     # (T-1, 6)     root 6D rotation
        r_6d_velocity,             # (T-1, 6)     root 6D rotation velocity
        rot_data[:-1],             # (T-1, 22*6)  joint 6D rotations
        rot_6d_velocity            # (T-1, 22*6)  joint 6D rotation velocities
    ], axis=-1).astype(np.float32, copy=False)

    return data, global_positions, positions, T34_out.detach().cpu().numpy()


def unpack_data(data, joints_num):
    J = joints_num
    i = 0
    v_root_local = data[..., i:i + 3];               i += 3
    root6d_t = data[..., i:i + 6];                   i += 6
    root6d_delta = data[..., i:i + 6];               i += 6
    joints6d_t = data[..., i:i + (J - 1) * 6];       i += (J - 1) * 6
    joints6d_delta = data[..., i:i + (J - 1) * 6]
    return v_root_local, root6d_t, root6d_delta, joints6d_t, joints6d_delta


@torch.no_grad()
def recover_cont6d_and_rootpos(data, joints_num, device):
    """Invert the 279-D feature into per-frame 6D rotations (T, J, 6) and root positions (T, 3)."""
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, device=device, dtype=torch.float32)
    v_root_local, root6d_t, root6d_delta, joints6d_t, joints6d_delta = unpack_data(data, joints_num)

    Tm1 = data.shape[-2]

    # root rotation: R_{t+1} = dR_t * R_t
    rquat_t = cont6d_to_quat(root6d_t).view(Tm1, 4)
    rquat_d = cont6d_to_quat(root6d_delta).view(Tm1, 4)
    rquat_tp1 = qmul(rquat_d, rquat_t)

    # root position: v_root_local[t] is expressed in the (t+1) frame
    v_global = qrot(qinv(rquat_tp1), v_root_local)
    root_pos = torch.zeros(Tm1 + 1, 3, device=device)
    root_pos[1:] = torch.cumsum(v_global, dim=0)

    # joint rotations
    joints6d_t = joints6d_t.view(Tm1, -1, 6)
    joints6d_delta = joints6d_delta.view(Tm1, -1, 6)
    jquat_t = cont6d_to_quat(joints6d_t)
    jquat_d = cont6d_to_quat(joints6d_delta)
    jquat_tp1 = qmul(jquat_d, jquat_t)
    joints6d_tp1 = quaternion_to_cont6d(jquat_tp1)

    # frames 0 .. T-2, then the last frame from the propagated rotations
    cont6d_t = torch.cat([root6d_t.view(Tm1, 1, 6), joints6d_t], dim=1)
    root6d_last = quaternion_to_cont6d(rquat_tp1[-1:])
    cont6d_last = torch.cat([root6d_last[:, None], joints6d_tp1[-1:]], dim=1)
    cont6d_all = torch.cat([cont6d_t, cont6d_last], dim=0)

    return cont6d_all, root_pos


@torch.no_grad()
def recover_positions(data, joints_num, skeleton, device):
    cont6d_all, root_pos = recover_cont6d_and_rootpos(data, joints_num, device)
    return skeleton.forward_kinematics_cont6d(cont6d_all, root_pos)  # (T, J, 3)


def rotate_joints(joints, R):
    return joints @ R


def rotate_pose_matrix(T34, R):
    out = T34.copy()
    out[..., :3, :3] = R.T[None, ...] @ out[..., :3, :3]
    out[..., :3, 3] = out[..., :3, 3] @ R
    return out


def target_skeleton_offsets(args):
    if args.skel_offsets:
        return torch.from_numpy(np.load(args.skel_offsets))
    example = np.load(args.reference_motion)
    example = example.reshape(len(example), -1, 3)
    example = torch.from_numpy(rotate_joints(example, R))
    return Skeleton(n_raw_offsets, kinematic_chain, 'cpu').get_offsets_joints(example[0])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parsed_dir", required=True, help="root of the parsed Nymeria scenes")
    p.add_argument("--out_dir", required=True, help="output root (per-scene folders are created inside)")
    ref = p.add_mutually_exclusive_group(required=True)
    ref.add_argument("--skel_offsets", help="precomputed target skeleton offsets (skel_offsets.npy of the release)")
    ref.add_argument("--reference_motion", help="body_pose .npy whose first frame defines the target skeleton "
                                                "(Ego3DLM: 20230830_s0_jason_vaughn_act3_00fg0l/body_pose/0001.npy)")
    p.add_argument("--scenes", nargs="+", default=None, help="subset of scenes (default: all)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tgt_offsets = target_skeleton_offsets(args)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')  # used to recover the joints from the features
    tgt_skel.set_offsets_joints(tgt_offsets)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(pjoin(args.out_dir, 'skel_offsets.npy'), tgt_offsets)

    scenes = args.scenes or sorted(d for d in os.listdir(args.parsed_dir)
                                   if os.path.isdir(pjoin(args.parsed_dir, d, 'body_pose')))
    num_frames = 0
    for scene in tqdm(scenes):
        scene_dir = pjoin(args.parsed_dir, scene)
        out = {k: pjoin(args.out_dir, k, scene) for k in ('new_joints', 'new_joint_vecs', 'three_points')}
        for d in out.values():
            os.makedirs(d, exist_ok=True)

        for pose_file in sorted(os.listdir(pjoin(scene_dir, 'body_pose'))):
            joints = np.load(pjoin(scene_dir, 'body_pose', pose_file))
            tps = np.stack([np.load(pjoin(scene_dir, f'{k}_pose', pose_file))
                            for k in ('head', 'left_wrist', 'right_wrist')], axis=1)
            try:
                data, _, _, T34_out = process_file(rotate_joints(joints, R), rotate_pose_matrix(tps, R), tgt_offsets)
                positions_rst = recover_positions(torch.from_numpy(data).float(), joints_num, tgt_skel, device)
                positions_rst = positions_rst.cpu().numpy()
                assert not np.isnan(positions_rst).any(), 'recovered positions contain NaN'
                assert not np.isnan(data).any(), 'motion representation contains NaN'
            except Exception as e:
                print(f'[skip] {scene}/{pose_file}: {e}')
                continue

            np.save(pjoin(out['new_joints'], pose_file), positions_rst)
            np.save(pjoin(out['new_joint_vecs'], pose_file), data)
            np.save(pjoin(out['three_points'], pose_file), T34_out)
            num_frames += data.shape[0]

    print(f'scenes: {len(scenes)}, frames: {num_frames}')


if __name__ == "__main__":
    main()
