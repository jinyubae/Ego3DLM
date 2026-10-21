import torch.nn as nn
from .base import BaseLosses, PassThroughLoss
from ..data.motion.quaternion import cont6d_to_quat, qinv, qmul, quaternion_to_cont6d


class PQVAELosses(BaseLosses):
    """Losses of the motion tokenizer (Stage 0): feature and joint reconstruction, consistency of the
    rotation velocities with the rotations, and the VQ commitment loss."""

    def __init__(self, cfg, stage, num_joints, unpack_features, **kwargs):
        self.stage = stage
        self.unpack_features = unpack_features

        losses = []
        params = {}
        if stage == "vae":
            losses = ["recons_feature", "recons_joint", "recons_velocity", "vq_commit"]
            params = {
                'recons_feature': cfg.LOSS.LAMBDA_FEATURE,
                'recons_joint': cfg.LOSS.LAMBDA_JOINT,
                'recons_velocity': cfg.LOSS.LAMBDA_VELOCITY,
                'vq_commit': cfg.LOSS.LAMBDA_COMMIT,
            }
        losses_func = {loss: PassThroughLoss if loss == "vq_commit" else nn.SmoothL1Loss for loss in losses}

        super().__init__(cfg, losses, params, losses_func, num_joints,
                         **kwargs)

    def update(self, rs_set):
        '''Update the losses'''
        total: float = 0.0

        if self.stage in ["vae"]:
            total += self._update_loss("recons_feature", rs_set['m_rst'], rs_set['m_ref'])
            total += self._update_loss("recons_joint", rs_set['joints_rst'], rs_set['joints_ref'])

            nfeats = rs_set['m_rst'].shape[-1]
            if nfeats != 279:
                raise NotImplementedError("Velocity not implemented for nfeats = {})".format(nfeats))
            root3d_vel, root6d_rot, root6d_vel, joint6d_rot, joint6d_vel = self.unpack_features(rs_set['m_rst'])

            # root rotation velocity
            rquat_t = cont6d_to_quat(root6d_rot)
            rquat_d_from_t = qmul(rquat_t[:, 1:].contiguous(), qinv(rquat_t[:, :-1].contiguous()))
            root6d_vel_from_rot = quaternion_to_cont6d(rquat_d_from_t)

            # joint rotation velocity
            jquat_t = cont6d_to_quat(joint6d_rot)
            jquat_d_from_t = qmul(jquat_t[:, 1:].contiguous(), qinv(jquat_t[:, :-1].contiguous()))
            joint6d_vel_from_rot = quaternion_to_cont6d(jquat_d_from_t)

            total += self._update_loss("recons_velocity", root6d_vel[:, :-1], root6d_vel_from_rot.detach())
            total += self._update_loss("recons_velocity", joint6d_vel[:, :-1], joint6d_vel_from_rot.detach())

            total += self._update_loss("vq_commit", rs_set['loss_commit'],
                                       rs_set['loss_commit'])

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total
