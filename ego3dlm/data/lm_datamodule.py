import numpy as np
import torch
import os
from os.path import join as pjoin
from .word_vectorizer import WordVectorizer
from .motion.motion_process import (process_file, recover_positions)
from . import BaseDataModule
from .datasets import (InstructDataset, SceneQADataset,
                      InstructEvalDataset, SceneQAEvalDataset)
from .collate import instruct_collate, scene_qa_collate


class LMDataModule(BaseDataModule):
    """Data of the LM stages: Stage I (TRAIN.STAGE 'lm_pretrain') and Stages II / III ('lm_instruct', 'lm_grpo')."""

    def __init__(self, cfg, **kwargs):

        if cfg.TRAIN.STAGE == 'lm_pretrain':
            super().__init__(collate_fn=scene_qa_collate)
        elif cfg.TRAIN.STAGE in ['lm_instruct', 'lm_grpo']:
            super().__init__(collate_fn=instruct_collate)
        else:
            raise ValueError(f"Unsupported stage {cfg.TRAIN.STAGE}")
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
        self.hparams.code_path = cfg.DATASET.CODE_PATH
        if cfg.TRAIN.STAGE == 'lm_pretrain':
            self.Dataset = SceneQADataset
            self.DatasetEval = SceneQAEvalDataset
        else:
            self.Dataset = InstructDataset
            self.DatasetEval = InstructEvalDataset

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

    def joints2feats(self, joints):
        example_data = np.load(os.path.join(self.hparams.data_root, 'new_joints', '000000.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(joints, example_data)[0]
        return features

    def renorm_for_evaluator(self, features):
        # renorm to the evaluator normalization
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features
