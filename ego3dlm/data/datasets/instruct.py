import re
import random
import os
import numpy as np
import codecs as cs
import torch
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json


def _parse_obstacle_label_to_cot_text(filepath):
    """Parse an obstacle label file and return a short deterministic CoT string for the last frame."""
    with open(filepath, 'r') as f:
        text = f.read()

    frame_blocks = re.findall(r'\[Frame \d+\](.*?)(?=\[Frame |\Z)', text, re.DOTALL)
    if not frame_blocks:
        return None
    last_frame = frame_blocks[-1]

    def parse_field(name):
        m = re.search(rf'{name}\s*=\s*(\S+)', last_frame)
        return m.group(1) if m else None

    free_front = parse_field('FREE_FRONT')
    free_left = parse_field('FREE_LEFT')
    free_right = parse_field('FREE_RIGHT')
    col_front = parse_field('COLLIDE_STEP_FRONT')
    col_left = parse_field('COLLIDE_STEP_LEFT')
    col_right = parse_field('COLLIDE_STEP_RIGHT')
    best_dir = (parse_field('BEST_DIR') or 'UNKNOWN').lower()

    def state_str(collide, free):
        status = "blocked" if collide == 'YES' else "clear"
        return f"{status} with {free} clearance"

    cot_text = (
        f"Front is {state_str(col_front, free_front)}. "
        f"Left is {state_str(col_left, free_left)}. "
        f"Right is {state_str(col_right, free_right)}. "
        f"The optimal direction to move is {best_dir}."
    )
    return cot_text

class InstructDataset(data.Dataset):
    """Stage II / III training samples: motion tokens of a segment and of the next (future) segment, their
    narrations, the 3D scene, the video features, the three-point poses and the scene-reasoning (CoT) text."""

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        unit_length=4,
        tiny=False,
        debug=False,
        code_path='TOKENS',
        **kwargs,
    ):
        self.unit_length = unit_length
        self.data_root = data_root
        self.three_points_dir = pjoin(data_root, 'three_points')

        self.num_3d_points = kwargs['cfg'].DATASET.NUM_3D_POINTS
        obstacle_labels_dir = kwargs['cfg'].DATASET.NYMERIA.get('OBSTACLE_LABELS_PATH', None)

        # Data mean and std
        self.mean = mean
        self.std = std

        # Data path
        split = 'train'
        split_file = pjoin(data_root, split + '.txt')
        pose_dir = pjoin(data_root, 'new_joint_vecs')
        motion_dir = pjoin(data_root, code_path)
        future_pose_dir = pose_dir
        future_motion_dir = motion_dir
        text_dir = pjoin(data_root, 'texts')
        video_feat_dir = pjoin(data_root, 'video_features')
        scene_feature_dir = pjoin(data_root, '3d_features_scene', 'voxelized_features')
        scene_voxel_dir = pjoin(data_root, '3d_features_scene', 'voxelized_voxels')
        head_pose_voxel_dir = pjoin(data_root, 'global_head_poses')
        _has_head_pose = os.path.isdir(head_pose_voxel_dir)
        self.scene_list = [scene.split('.')[0] for scene in sorted(os.listdir(scene_feature_dir))]
        
        self.scene_info = {}
        with open(pjoin(data_root, 'scene_info.txt'), 'r') as f :
            for line in f.readlines() :
                self.scene_info[line.split(',')[0]] = line.split(',')[-1].strip()
        
        instructions = pjoin(data_root, 'template_instructions_stp2mt_cot.json')

        # Data id list
        self.id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        # Debug mode
        if tiny or debug:
            enumerator = enumerate(self.id_list)
            maxdata = 100
        else:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading Nymeria {split}",
                ))
            maxdata = 1e10

        new_name_list = []
        data_dict = {}

        # Fast loading
        for i, name in enumerator:
            if len(new_name_list) > maxdata:
                break
            if i + 1 >= len(self.id_list):
                break
            try:
                # Load motion tokens
                next_name = self.id_list[i + 1]
                if self.scene_info[name] not in self.scene_list :
                    continue
                if int(name.split('.')[0]) + 1 != int(next_name.split('.')[0]) :
                        continue
                if self.scene_info[name] != self.scene_info[next_name] :
                    continue

                obstacle_cot_answer = None
                if obstacle_labels_dir is not None:
                    label_path = pjoin(obstacle_labels_dir, f'{name}.txt')
                    if os.path.exists(label_path):
                        obstacle_cot_answer = _parse_obstacle_label_to_cot_text(label_path)
                
                m_token_list = np.load(pjoin(motion_dir, f'{name}.npy'))
                video_feat_path = pjoin(video_feat_dir, f'{name}.pt')
                
                past_pose = np.load(pjoin(pose_dir, f'{name}.npy'))
                fut_pose = np.load(pjoin(future_pose_dir, f'{next_name}.npy'))
                
                if _has_head_pose:
                    head_pose_voxel_rot = np.load(pjoin(head_pose_voxel_dir, f'{name}.npz'))
                    global_head_voxel = head_pose_voxel_rot['voxel']
                    global_head_rot = head_pose_voxel_rot['rotation']
                else:
                    global_head_voxel = np.zeros((1, 3), dtype=np.int32)
                    global_head_rot = np.zeros((1, 3, 3), dtype=np.float32)
                
                if not os.path.exists(video_feat_path):
                    continue
                future_m_token_list = np.load(pjoin(future_motion_dir, f'{next_name}.npy'))
                with cs.open(pjoin(text_dir, next_name + '.txt')) as f:
                    next_lines = f.readlines()
                    future_text = next_lines[0].split('#')[0].strip() if len(next_lines) > 0 else ''
                # Read text
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
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
                                total_tokens = int(m_token_list.shape[1])
                                start_token = int(round(f_tag))
                                end_token = int(round(to_tag))
                                start_token = max(min(start_token, total_tokens), 0)
                                end_token = max(min(end_token, total_tokens), 0)

                                if end_token <= start_token:
                                    continue

                                m_token_list_new = [
                                    tokens[start_token:end_token]
                                    for tokens in m_token_list
                                ]

                                if len(m_token_list_new) == 0:
                                    continue

                                new_name = '%s_%f_%f' % (name, f_tag,
                                                            to_tag)

                                data_dict[new_name] = {
                                    'm_token_list': m_token_list_new,
                                    'text': [text_dict],
                                    'token_start': start_token,
                                    'token_end': start_token + m_token_list_new[0].shape[0],
                                    'base_name': name,
                                    'future_m_token_list': future_m_token_list,
                                    'future_text': future_text,
                                    'video_feat_path': video_feat_path,
                                    'past_pose': past_pose,
                                    'future_pose': fut_pose,
                                    'global_head_voxel': global_head_voxel,
                                    'global_head_rotation': global_head_rot,
                                    'obstacle_cot_answer': obstacle_cot_answer,
                                    
                                }
                                new_name_list.append(new_name)
                        except:
                            pass
                if flag and m_token_list.shape[1] >= int(min_motion_length // unit_length) and m_token_list.shape[1] <= int(max_motion_length // unit_length) :
                    data_dict[name] = {
                        'm_token_list': m_token_list,
                        'future_m_token_list': future_m_token_list,
                        'text': text_data,
                        'future_text': future_text,
                        'token_start': 0,
                        'token_end': m_token_list.shape[1],
                        'base_name': name,
                        'video_feat_path': video_feat_path,
                        'past_pose': past_pose,
                        'future_pose': fut_pose,
                        'global_head_voxel': global_head_voxel,
                        'global_head_rotation': global_head_rot,
                        'obstacle_cot_answer': obstacle_cot_answer,
                        
                    }
                    new_name_list.append(name)
            except Exception as e:
                print(e)
                pass

        self.data_dict = data_dict
        self.name_list = new_name_list
        self.sequence_meta = {}
        for data_index in self.name_list:
            entry = self.data_dict[data_index]
            self.sequence_meta[data_index] = {
                'base_name': entry['base_name'],
                'token_start': entry['token_start'],
                'frame_start': entry['token_start'] * self.unit_length,
            }

        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])
                
        self.scene_feat_dict = {}
        for scene in self.scene_list:
            scene_feature = torch.load(os.path.join(scene_feature_dir, scene+'.pt'))
            scene_voxel = np.load(os.path.join(scene_voxel_dir, scene+'.npz'))['voxel']
            min_coord = np.load(os.path.join(scene_voxel_dir, scene+'.npz'))['min_coords']
            assert scene_voxel.shape[0] == scene_feature.shape[0], f'scene_voxel shape ({scene_voxel.shape[0]}) != scene_feature shape ({scene_feature.shape[0]})'
            
            if scene_feature.shape[0] > self.num_3d_points:
                idxes = torch.sort(torch.randperm(scene_feature.shape[0])[:self.num_3d_points])[1]
                scene_feature = scene_feature[idxes]
                scene_voxel = scene_voxel[idxes]
                valid_len = self.num_3d_points
            else:
                valid_len = scene_feature.shape[0]
                scene_feature = torch.cat([scene_feature, torch.zeros(self.num_3d_points - scene_feature.shape[0], 1408)], dim=0)
                scene_voxel = np.concatenate([scene_voxel, np.zeros((self.num_3d_points - scene_voxel.shape[0], 3), dtype=scene_voxel.dtype)], axis=0)
            
            pc_mask = torch.zeros(self.num_3d_points, device=scene_feature.device, dtype=torch.long)
            pc_mask[:valid_len] = 1
            self.scene_feat_dict[scene] = {
                'pc_embed' : scene_feature,
                'pc' : scene_voxel,
                'pc_mask': pc_mask, 
                'pc_len': valid_len,
                'min_coord' : min_coord
            }
                

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
            if pad_value is None:
                pad_value = 0
            pad_shape = (expected_frames,) + array.shape[1:]
            return np.full(pad_shape, pad_value, dtype=array.dtype)
        pad_count = expected_frames - current
        if pad_value is None:
            pad_block = np.repeat(array[-1:, ...], pad_count, axis=0)
        else:
            pad_shape = (pad_count,) + array.shape[1:]
            pad_block = np.full(pad_shape, pad_value, dtype=array.dtype)
        return np.concatenate([array, pad_block], axis=0)


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        data_idx = item % len(self.name_list)
        task_idx = np.random.choice(len(self.tasks), p=[1/len(self.tasks)]*len(self.tasks))

        data_index = self.name_list[data_idx]
        data = self.data_dict[data_index]
        
        m_token_list = data['m_token_list']
        future_m_token = data['future_m_token_list'][0]
        text_list = data['text']
        future_text = data['future_text']
        
        past_pose = data['past_pose']
        fut_pose = data['future_pose']
        
        global_head_voxel = data['global_head_voxel']
        global_head_rot = data['global_head_rotation']

        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data['caption']
        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]
        m_tokens_len = m_tokens.shape[0]
        future_m_tokens_len = future_m_token.shape[0]

        tasks = self.tasks[task_idx]
        scene = self.scene_feat_dict[self.scene_info[data_index]]
        
        pc = scene['pc']
        pc_embed = scene['pc_embed']
        pc_mask = scene['pc_mask']
        pc_len = scene['pc_len']
        min_coord = scene['min_coord']

        # three-point poses of the frames of the motion tokens
        meta = self.sequence_meta[data_index]
        clip_start = meta['frame_start']
        frame_count = m_tokens_len * self.unit_length
        three_points_path = pjoin(self.three_points_dir, f"{meta['base_name']}.npy")
        three_points = self._get_three_points_array(three_points_path)
        three_points = self._slice_sequence(three_points, clip_start, frame_count)
        three_points_len = three_points.shape[0]

        video_feat_path = data['video_feat_path']
        video_embed = torch.load(video_feat_path)
        video_lengths = video_embed.shape[0]
        obstacle_cot_answer = data.get('obstacle_cot_answer', None) or ''

        return \
            caption,\
            m_tokens, m_tokens_len,\
            None, None, None, None,\
            all_captions,\
            tasks,\
            pc, pc_embed, pc_mask, pc_len,\
            three_points, three_points_len, \
            video_embed, video_lengths,\
            future_text, future_m_token, future_m_tokens_len, \
            past_pose, fut_pose, \
            global_head_voxel, global_head_rot, min_coord, \
            obstacle_cot_answer


'''
caption, #0
m_tokens, m_tokens_len, #1, 2
None, None, None, None, #3, 4, 5, 6
all_captions,  #7
tasks, #8
pc, pc_embed, pc_mask, pc_len, #9, 10, 11, 12
three_points, three_points_len, #13, 14
video_embed, video_lengths, #15, 16
caption_future, m_tokens_future, m_tokens_len_future #17, 18, 19
past_pose, fut_pose # 20, 21
global_head_voxel, global_head_rot, min_coord # 22, 23, 24
obstacle_cot_answer # 25
'''
