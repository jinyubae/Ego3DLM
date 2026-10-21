import numpy as np
import torch
from .skeleton import Skeleton
from .quaternion import (cont6d_to_quat, cont6d_to_quat_np, qbetween_np, qinv, qinv_np, qmul, qmul_np, qrot,
                                 qrot_np, quaternion_to_cont6d, quaternion_to_cont6d_np)
from .skeleton_params import KINEMATIC_CHAIN, RAW_OFFSETS


# right lower leg
l_idx1, l_idx2 = 16, 17
r_hip, l_hip = 15, 19

face_joint_indx = [15, 19, 7, 11] # [15, 19, 8, 12]
joints_num = 23


def uniform_skeleton(positions, target_offset):
    kinematic_chain = KINEMATIC_CHAIN
    n_raw_offsets = torch.from_numpy(RAW_OFFSETS)
    
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)

    
    return new_joints


def process_file(positions, example_data):
    
    '''Uniform Skeleton'''
    
    kinematic_chain = KINEMATIC_CHAIN
    n_raw_offsets = torch.from_numpy(RAW_OFFSETS)
    
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    '''New ground truth positions'''
    global_positions = positions.copy()


    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    
    r_6d_velocity = quaternion_to_cont6d_np(r_velocity)
    # positions = get_rifke(positions)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    root_6d_rotation = cont_6d_params[:, 0]
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1) 
    
    j_quat = cont6d_to_quat_np(cont_6d_params[:, 1:])
    rot_velocity = qmul_np(j_quat[1:], qinv_np(j_quat[:-1]))
    rot_6d_velocity = quaternion_to_cont6d_np(rot_velocity)
    rot_6d_velocity = rot_6d_velocity.reshape(len(rot_6d_velocity), -1)
    

    data = np.concatenate([
        velocity,                  # (T-1, 3), root translation velocity
        root_6d_rotation[:-1],     # (T-1, 6), root 6d rotation
        r_6d_velocity,             # (T-1, 6), root 6d rotation velocity
        rot_data[:-1],             # (T-1, 22*6), joint 6d angles
        rot_6d_velocity               # (T-1, 22*6), joint 6d angle velocity
    ], axis=-1).astype(np.float32, copy=False)
    
    return data, global_positions, positions


def unpack_data(data, joints_num):
    """
    data: (T-1, D)
    returns:
      v_root_local   (T-1, 3)
      root6d_t       (T-1, 6)
      root6d_delta   (T-1, 6)
      joints6d_t     (T-1, (J-1)*6)
      joints6d_delta (T-1, (J-1)*6)
    """
    J = joints_num
    i = 0
    v_root_local   = data[..., i:i+3];                i += 3
    root6d_t       = data[..., i:i+6];                i += 6
    root6d_delta   = data[..., i:i+6];                i += 6
    joints6d_t     = data[..., i:i+(J-1)*6];          i += (J-1)*6
    joints6d_delta = data[..., i:i+(J-1)*6]           # i += ...
    return v_root_local, root6d_t, root6d_delta, joints6d_t, joints6d_delta


def recover_cont6d_and_rootpos(data, joints_num):
    """
    Reconstruct:
      cont6d_all: (B, T, J, 6)
      root_pos:   (B, T, 3)
    Assumes the helpers you already use in your project.
    """
    v_root_local, root6d_t, root6d_delta, joints6d_t, joints6d_delta = unpack_data(data, joints_num)
    
    Tm1 = data.shape[-2]
    B, Tm1, _ = data.shape

    '''1) Root Rotation'''
    rquat_t = cont6d_to_quat(root6d_t)          # (T-1,4)
    rquat_d = cont6d_to_quat(root6d_delta)      # (T-1,4)
    rquat_tp1 = qmul(rquat_d, rquat_t)          # R_{t+1} = ΔR_t * R_t
    
    '''2) Root Position'''
    #    v_root_local[t] was encoded in (t+1) frame -> bring to global with R_{t+1}^{-1}
    v_global = qrot(qinv(rquat_tp1), v_root_local)           # (T-1,3)
    root_pos = torch.zeros(B, Tm1 + 1, 3, device=data.device)
    root_pos[:, 1:] = torch.cumsum(v_global, dim=1)             # p_{t+1} = Σ v_global

    '''3) Joint Rotation'''
    joints6d_t   = joints6d_t.view(B, Tm1, -1, 6)                                   # (T-1, J-1, 6)
    joints6d_delta = joints6d_delta.view(B, Tm1, -1, 6)
    
    #! original (yujin)
    # joints6d_tp1 = joints6d_t + joints6d_delta.view(Tm1, -1, 6)                  # simple Euler step
    #! alternative
    jquat_t = cont6d_to_quat(joints6d_t)                     # (T-1,4)
    jquat_d = cont6d_to_quat(joints6d_delta)                 # (T-1,4)
    jquat_tp1 = qmul(jquat_d, jquat_t)                       # R_{t+1} = ΔR_t * R_t
    joints6d_tp1 = quaternion_to_cont6d(jquat_tp1)
    
    '''4) Root & Joint Rotation for Frames 0...T-2'''
    cont6d_t = torch.cat([root6d_t.view(B, Tm1, 1, 6), joints6d_t], dim=2)           # (T-1, J, 6)

    '''5) Build the last frame (T-1): use R_{t+1} from rquat_tp1 and joints6d_{t+1}'''
    root6d_last = quaternion_to_cont6d(rquat_tp1[:, -1:])                            # -> (1,6) ndarray
    cont6d_last = torch.cat([root6d_last[:, :, None], joints6d_tp1[:, -1:]], dim=2)      # (1,J,6)

    # 6) stitch all frames
    cont6d_all = torch.cat([cont6d_t, cont6d_last], dim=1)                        # (T, J, 6)

    return cont6d_all, root_pos


def recover_positions(data, joints_num, offsets):
    cont6d_all, root_pos = recover_cont6d_and_rootpos(data, joints_num)
    B, T, J, _ = cont6d_all.shape
    # FK expects (T, J, 6) + (T, 3)
    skeleton = Skeleton(torch.from_numpy(RAW_OFFSETS), KINEMATIC_CHAIN, data.device)
    skeleton.set_offset(torch.from_numpy(offsets))
    positions = skeleton.forward_kinematics_cont6d(cont6d_all.view(-1, J, 6), root_pos.view(-1, 3))
    return positions.view(B, T, J, 3)  # (T, J, 3)

