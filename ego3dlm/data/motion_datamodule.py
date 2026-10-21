import numpy as np
import torch
import os
from os.path import join as pjoin
from .word_vectorizer import WordVectorizer
from .motion.motion_process import recover_positions
from . import BaseDataModule
from .datasets import MotionEvalDataset, MotionVQDataset, MotionTokenizeDataset
from .collate import motion_collate


class MotionDataModule(BaseDataModule):
    """Motion data of the motion tokenizer: Stage 0 (TRAIN.STAGE 'vae') and tokenization ('token')."""

    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=motion_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)

        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'xsens'
        self.name = "nymeria"
        self.njoints = 23

        # Path to the dataset
        data_root = cfg.DATASET.NYMERIA.ROOT
        self.hparams.data_root = data_root

        # Mean and std of the dataset (also used to renormalize the features for the evaluator)
        dis_data_root = cfg.DATASET.NYMERIA.MEAN_STD_PATH
        self.hparams.mean = np.load(pjoin(dis_data_root, "mean.npy"))
        self.hparams.std = np.load(pjoin(dis_data_root, "std.npy"))
        self.hparams.mean_eval = np.load(pjoin(dis_data_root, "mean.npy"))
        self.hparams.std_eval = np.load(pjoin(dis_data_root, "std.npy"))

        # Length of the dataset
        self.hparams.max_motion_length = cfg.DATASET.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.MIN_MOTION_LEN
        self.hparams.unit_length = cfg.DATASET.UNIT_LEN

        # Additional parameters
        self.hparams.debug = cfg.DEBUG
        self.hparams.w_vectorizer = WordVectorizer(cfg.DATASET.WORD_VECTORIZER_PATH)

        # Dataset switch
        if cfg.TRAIN.STAGE == "vae":
            self.hparams.win_size = 32
            self.Dataset = MotionVQDataset
            self.DatasetEval = MotionEvalDataset
        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = MotionTokenizeDataset
            self.DatasetEval = MotionTokenizeDataset
        else:
            raise ValueError(f"Unsupported stage {cfg.TRAIN.STAGE}")

        # Get additional info of the dataset
        self._sample_set = self.get_sample_set(overrides={"split": "test", "tiny": True})
        self.nfeats = self._sample_set.nfeats
        self.skel_offsets = np.load(os.path.join(data_root, 'skel_offsets.npy'))
        cfg.DATASET.NFEATS = self.nfeats

    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_positions(features, self.njoints, self.skel_offsets)

    def unpack_features(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean

        B, T, _ = features.shape

        i = 0
        root3d_vel = features[..., i:i+3]; i += 3
        root6d_rot = features[..., i:i+6]; i += 6
        root6d_vel = features[..., i:i+6]; i += 6
        joint6d_rot = features[..., i:i+6*(self.njoints-1)]; i += 6*(self.njoints-1)
        joint6d_vel = features[..., i:i+6*(self.njoints-1)]; i += 6*(self.njoints-1)

        return root3d_vel, root6d_rot, root6d_vel, joint6d_rot.view(B,T,-1,6), joint6d_vel.view(B,T,-1,6)
