import random
import numpy as np
from .motion import MotionDataset
from os.path import join as pjoin
import os
import torch
import json
from .scene_qa import _parse_obstacle_label_to_text


class SceneQAEvalDataset(MotionDataset):
    """Stage I validation samples: semantic scene QA pairs and the obstacle state of the last frame,
    with the 3D scene and the video features."""

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        
        num_3d_points = kwargs['cfg'].DATASET.NUM_3D_POINTS
        
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        self.w_vectorizer = w_vectorizer

        # Read video and text
        video_feat_dir = pjoin(data_root, 'video_features')
        video_text_dir = pjoin(data_root, 'video_texts')
        head_pose_voxel_dir = pjoin(data_root, 'global_head_poses')
        # obstacle labels directory (optional; when set, only clips with labels are loaded)
        obstacle_labels_dir = kwargs['cfg'].DATASET.NYMERIA.get('OBSTACLE_LABELS_PATH', None)
        
        name_list = list(self.name_list)
        for name in self.name_list:
            # Load obstacle label if available; None if missing (handled in __getitem__)
            obstacle_answer = None
            if obstacle_labels_dir is not None:
                label_path = pjoin(obstacle_labels_dir, f'{name}.txt')
                if os.path.exists(label_path):
                    obstacle_answer = _parse_obstacle_label_to_text(label_path)

            # Read video feature path
            video_feat_path = pjoin(video_feat_dir, f'{name}.pt')
            head_pose_voxel_rot = np.load(pjoin(head_pose_voxel_dir, f'{name}.npz'))
            global_head_voxel = head_pose_voxel_rot['voxel']
            global_head_rot = head_pose_voxel_rot['rotation']
            if not os.path.exists(video_feat_path):
                continue
            
            scene_qa = json.load(open(pjoin(video_text_dir, name, 'qa_dataset.final.filtered.json'), 'r'))
            if len(scene_qa) == 0:
                del self.data_dict[name]
                name_list.remove(name)
                continue
            scene_qa_data = []
            for qa in scene_qa:
                scene_qa_dict = {}
                scene_qa_dict['question'] = qa['question']
                scene_qa_dict['answer'] = qa['answer']
                scene_qa_data.append(scene_qa_dict)
            self.data_dict[name]['scene_qa'] = scene_qa_data
            self.data_dict[name]['video_feat_path'] = video_feat_path
            self.data_dict[name]['global_head_voxel'] = global_head_voxel
            self.data_dict[name]['global_head_rotation'] = global_head_rot
            self.data_dict[name]['obstacle_answer'] = obstacle_answer
        self.name_list = tuple(name_list)

        # Read scene feature
        scene_feature_dir = pjoin(data_root, '3d_features_scene', 'voxelized_features')
        scene_voxel_dir = pjoin(data_root, '3d_features_scene', 'voxelized_voxels')
        self.scene_list = [scene.split('.')[0] for scene in sorted(os.listdir(scene_feature_dir))]
        
        self.scene_info = {}
        with open(pjoin(data_root, 'scene_info.txt'), 'r') as f :
            for line in f.readlines() :
                self.scene_info[line.split(',')[0]] = line.split(',')[-1].strip()
                
        self.scene_feat_dict = {}
        for scene in self.scene_list:
            scene_feature = torch.load(os.path.join(scene_feature_dir, scene+'.pt'))
            scene_voxel = np.load(os.path.join(scene_voxel_dir, scene+'.npz'))['voxel']
            min_coord = np.load(os.path.join(scene_voxel_dir, scene+'.npz'))['min_coords']
            
            assert scene_voxel.shape[0] == scene_feature.shape[0], f'scene_voxel shape ({scene_voxel.shape[0]}) != scene_feature shape ({scene_feature.shape[0]})'
            
            if scene_feature.shape[0] > num_3d_points:
                idxes = torch.sort(torch.randperm(scene_feature.shape[0])[:num_3d_points])[1]
                scene_feature = scene_feature[idxes]
                scene_voxel = scene_voxel[idxes]
                valid_len = num_3d_points
            else:
                valid_len = scene_feature.shape[0]
                scene_feature = torch.cat([scene_feature, torch.zeros(num_3d_points - scene_feature.shape[0], 1408)], dim=0)
                scene_voxel = np.concatenate([scene_voxel, np.zeros((num_3d_points - scene_voxel.shape[0], 3), dtype=scene_voxel.dtype)], axis=0)
            
            pc_mask = torch.zeros(num_3d_points, device=scene_feature.device, dtype=torch.long)
            pc_mask[:valid_len] = 1
            self.scene_feat_dict[scene] = {
                'pc_embed' : scene_feature,
                'pc' : scene_voxel,
                'pc_mask': pc_mask, 
                'pc_len': valid_len,
                'min_coord' : min_coord
            }        
                
                
    def __getitem__(self, item):
        # Get text data
        idx = self.pointer + item
        data_index = self.name_list[idx]
        data = self.data_dict[data_index]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        
        global_head_voxel = data['global_head_voxel']
        global_head_rot = data['global_head_rotation']
        
        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        # Text
        max_text_len = 20
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        
        # Random crop
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        
        # Z Normalization
        motion = (motion - self.mean) / self.std
        
        scene = self.scene_feat_dict[self.scene_info[data_index]]
        pc = scene['pc']
        pc_embed = scene['pc_embed']
        pc_mask = scene['pc_mask']
        pc_len = scene['pc_len']
        min_coord = scene['min_coord']
        
        scene_qa_list = data['scene_qa']
        scene_qa_data = random.choice(scene_qa_list)
        scene_q = scene_qa_data['question']
        scene_a = scene_qa_data['answer']
        
        video_feat_path = data['video_feat_path']
        video_embed = torch.load(video_feat_path)
        video_lengths = video_embed.shape[0]

        obstacle_answer = data.get('obstacle_answer', None) or ''

        return caption, motion, m_length, word_embeddings, pos_one_hots, sent_len, "_".join(
            tokens), all_captions, None, pc, pc_embed, pc_mask, pc_len, video_embed, video_lengths, scene_q, scene_a, None, global_head_voxel, global_head_rot, min_coord, obstacle_answer
