import codecs as cs
import os
import pickle
import random

import numpy as np
import rich
import torch
from os.path import join as pjoin
from rich.progress import track
from torch.utils import data

from .instruct import _parse_obstacle_label_to_cot_text


class _InstructEvalBase(data.Dataset):
    """Motion / narration pairs of consecutive segments of one recording (base of InstructEvalDataset)."""

    def __init__(
        self,
        data_root,
        scene_info,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        unit_length=4,
        fps=10,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):

        # restrian the length of motion and text
        self.max_length = 20
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length

        # Data mean and std
        self.mean = mean
        self.std = std

        # Data path
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        future_motion_dir = motion_dir
        split_file = pjoin(data_root, split + '.txt')
        text_dir = pjoin(data_root, 'texts')
        video_feat_dir = pjoin(data_root, 'video_features')
        head_pose_voxel_dir = pjoin(data_root, 'global_head_poses')
        _has_head_pose = os.path.isdir(head_pose_voxel_dir)

        # Data id list
        self.id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        # Debug mode
        if tiny or debug:
            enumerator = enumerate(self.id_list)
            maxdata = 100
            subset = '_tiny'
        else:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading Nymeria {split}",
                ))
            maxdata = 1e10
            subset = ''

        new_name_list = []
        length_list = []
        data_dict = {}

        # Fast loading from the cache of a previous run (one cache per dataset class)
        cache = pjoin(data_root, 'tmp', f'{split}{subset}_{type(self).__name__}')
        if os.path.exists(f'{cache}_data.pkl'):
            if tiny or debug:
                with open(f'{cache}_data.pkl', 'rb') as file:
                    data_dict = pickle.load(file)
            else:
                with rich.progress.open(f'{cache}_data.pkl', 'rb',
                                        description=f"Loading Nymeria {split}") as file:
                    data_dict = pickle.load(file)
            with open(f'{cache}_index.pkl', 'rb') as file:
                name_list = pickle.load(file)
            for name in name_list:
                length_list.append(data_dict[name]['length'])

        else:
            for i, name in enumerator:
                if len(new_name_list) > maxdata:
                    break
                if i + 1 >= len(self.id_list):
                    break
                try:
                    next_name = self.id_list[i + 1]
                    if self.scene_info[name] not in self.scene_list :
                        continue
                    if int(name.split('.')[0]) + 1 != int(next_name.split('.')[0]) :
                        continue
                    if self.scene_info[name] != self.scene_info[next_name] :
                        continue
                    
                    motion = np.load(pjoin(motion_dir, name + ".npy"))
                    future_motion = np.load(pjoin(future_motion_dir, next_name + ".npy"))
                    video_feat_path = pjoin(video_feat_dir, f'{name}.pt')
                    
                    if _has_head_pose:
                        head_pose_voxel_rot = np.load(pjoin(head_pose_voxel_dir, f'{name}.npz'))
                        global_head_voxel = head_pose_voxel_rot['voxel']
                        global_head_rot = head_pose_voxel_rot['rotation']
                    else:
                        global_head_voxel = np.zeros((1, 3), dtype=np.int32)
                        global_head_rot = np.zeros((1, 3, 3), dtype=np.float32)
                    if not os.path.exists(video_feat_path):
                        continue
                    
                    if (len(motion)) < self.min_motion_length or (len(motion)
                                                                  >= self.max_motion_length):
                        continue

                    # Read text
                    text_data = []
                    flag = False
                    with cs.open(pjoin(text_dir, name + '.txt')) as f:
                        lines = f.readlines()
                        for line in lines:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                motion_new = motion[int(f_tag *
                                                        fps):int(to_tag * fps)]
                                if (len(motion_new)
                                    ) < self.min_motion_length or (
                                        len(motion_new) >= 200):
                                    continue
                                new_name = random.choice(
                                    'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in new_name_list:
                                    new_name = random.choice(
                                        'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                name_count = 1
                                while new_name in data_dict:
                                    new_name += '_' + name_count
                                    name_count += 1
                                data_dict[new_name] = {
                                    'motion': motion_new,
                                    "length": len(motion_new),
                                    'text': [text_dict]
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(motion_new))

                    with cs.open(pjoin(text_dir, next_name + '.txt')) as f:
                        lines = f.readlines()[0]
                        future_text = lines.split('#')[0].strip()
                        
                    if flag and len(motion) > min_motion_length : 
                        data_dict[name] = {
                            'motion': motion,
                            'future_motion': future_motion,
                            "length": len(motion),
                            "future_length": len(future_motion),
                            'text': text_data,
                            'future_text': future_text,
                            'video_feat_path': video_feat_path,
                            'global_head_voxel': global_head_voxel,
                            'global_head_rotation': global_head_rot
                        }
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except:
                    pass

            name_list, length_list = zip(
                *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

            if tmpFile:
                os.makedirs(pjoin(data_root, 'tmp'), exist_ok=True)
                with open(f'{cache}_data.pkl', 'wb') as file:
                    pickle.dump(data_dict, file)
                with open(f'{cache}_index.pkl', 'wb') as file:
                    pickle.dump(name_list, file)

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.nfeats = data_dict[name_list[0]]['motion'].shape[1]
        self.reset_max_len(self.max_length)
        
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.name_list) - self.pointer


class InstructEvalDataset(_InstructEvalBase):
    """Stage II / III validation (and evaluation) samples: motion features of a segment and of the next
    (future) segment with their narrations, the 3D scene, the video features, the three-point poses and the
    scene-reasoning (CoT) text."""

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

        self.w_vectorizer = w_vectorizer
        scene_feature_dir = pjoin(data_root, '3d_features_scene', 'voxelized_features')
        scene_voxel_dir = pjoin(data_root, '3d_features_scene', 'voxelized_voxels')
        self.scene_list = [scene.split('.')[0] for scene in sorted(os.listdir(scene_feature_dir))]

        self.data_root = data_root
        self.three_points_dir = pjoin(data_root, 'three_points')

        self.scene_info = {}
        with open(pjoin(data_root, 'scene_info.txt'), 'r') as f :
            for line in f.readlines() :
                self.scene_info[line.split(',')[0]] = line.split(',')[-1].strip()
        
        super().__init__(data_root, self.scene_info, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        # Drop samples without future_motion (the parent adds segmented entries, from texts with
        # time tags, without the future fields); keep length_arr in sync and refresh the pointer.
        keep_mask = np.array(
            ['future_motion' in self.data_dict[n] for n in self.name_list],
            dtype=bool,
        )
        if not keep_mask.all():
            dropped = int((~keep_mask).sum())
            self.name_list = [n for n, k in zip(self.name_list, keep_mask) if k]
            self.length_arr = self.length_arr[keep_mask]
            self.reset_max_len(self.max_length)
            print(f"[Eval] dropped {dropped} samples without future_motion; "
                  f"remaining={len(self.name_list)}")

        obstacle_labels_dir = kwargs['cfg'].DATASET.NYMERIA.get('OBSTACLE_LABELS_PATH', None)
        obstacle_cot_cache = {}
        for name in self.name_list:
            base_name = self._resolve_base_name(name)
            obstacle_cot_answer = None
            if obstacle_labels_dir is not None:
                if base_name not in obstacle_cot_cache:
                    label_path = pjoin(obstacle_labels_dir, f'{base_name}.txt')
                    if os.path.exists(label_path):
                        obstacle_cot_cache[base_name] = _parse_obstacle_label_to_cot_text(label_path)
                    else:
                        obstacle_cot_cache[base_name] = None
                obstacle_cot_answer = obstacle_cot_cache[base_name]
            self.data_dict[name]['obstacle_cot_answer'] = obstacle_cot_answer
        
        self.sequence_meta = {}
        for data_index in self.name_list:
            self.sequence_meta[data_index] = {
                'base_name': self._resolve_base_name(data_index),
            }

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
                scene_voxel = np.concatenate([scene_voxel, np.zeros((num_3d_points - scene_voxel.shape[0], 3))], axis=0)
            
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
        motion = data["motion"]
        future_motion = data["future_motion"]
        m_length = data["length"]
        future_m_length = data["future_length"]
        text_list = data["text"]
        future_text = data["future_text"]
        
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

        # Crop to a multiple of the token length
        m_length = (m_length // self.unit_length) * self.unit_length
        future_m_length = (future_m_length // self.unit_length) * self.unit_length
        idx = 0
        motion = motion[idx:idx + m_length]
        future_motion = future_motion[idx:idx + future_m_length]
        future_m_tokens_len = future_motion.shape[0]

        # Z Normalization
        motion = (motion - self.mean) / self.std
        future_motion = (future_motion - self.mean) / self.std

        base_name = self.sequence_meta[data_index]['base_name']
        scene = self.scene_feat_dict[self.scene_info[base_name]]

        pc = scene['pc']
        pc_embed = scene['pc_embed']
        pc_mask = scene['pc_mask']
        pc_len = scene['pc_len']
        min_coord = scene['min_coord']

        three_points_path = pjoin(self.three_points_dir, f'{base_name}.npy')
        three_points = self._get_three_points_array(three_points_path)
        three_points = self._slice_sequence(three_points, idx, m_length)
        three_points_len = three_points.shape[0]

        video_feat_path = data['video_feat_path']
        video_embed = torch.load(video_feat_path)
        video_lengths = video_embed.shape[0]
        obstacle_cot_answer = data.get('obstacle_cot_answer', None) or ''
        
        scene_name = self.scene_info[base_name]
        data_idx = base_name

        return (
            caption,
            motion,
            m_length,
            word_embeddings,
            pos_one_hots,
            sent_len,
            "_".join(tokens),
            all_captions,
            None,
            pc,
            pc_embed,
            pc_mask,
            pc_len,
            three_points,
            three_points_len,
            video_embed, 
            video_lengths,
            future_text,
            future_motion,
            future_m_tokens_len,
            None,
            None,
            global_head_voxel, 
            global_head_rot,
            min_coord,
            obstacle_cot_answer,
            scene_name,
            data_idx
        )

    def _resolve_base_name(self, data_index):
        if data_index in self.scene_info:
            return data_index
        if '_' in data_index:
            candidate = data_index.split('_', 1)[1]
            if candidate in self.scene_info:
                return candidate
        return data_index

    def _get_three_points_array(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'three_points file not found: {path}')
        return np.load(path)


    def _slice_sequence(self, array, frame_start, frame_count, pad_value=None):
        if frame_count <= 0:
            return self._ensure_length(array[:0], 0, pad_value)
        total_frames = array.shape[0]
        start = max(frame_start, 0)
        if start >= total_frames:
            start = max(total_frames - frame_count, 0)
        end = start + frame_count
        clip = array[start:end]
        if clip.shape[0] < frame_count:
            clip = self._ensure_length(clip, frame_count, pad_value=pad_value)
        return clip

    def _ensure_length(self, array, expected_frames, pad_value=None):
        current = array.shape[0]
        if current == expected_frames:
            return array
        if current > expected_frames:
            return array[:expected_frames]
        if current == 0:
            pad_value = 0 if pad_value is None else pad_value
            pad_shape = (expected_frames,) + array.shape[1:]
            return np.full(pad_shape, pad_value, dtype=array.dtype)
        pad_count = expected_frames - current
        if pad_value is None:
            pad_block = np.repeat(array[-1:, ...], pad_count, axis=0)
        else:
            pad_shape = (pad_count,) + array.shape[1:]
            pad_block = np.full(pad_shape, pad_value, dtype=array.dtype)
        return np.concatenate([array, pad_block], axis=0)


