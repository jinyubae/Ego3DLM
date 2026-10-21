import random
import os
import numpy as np
import codecs as cs
import torch
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json
import re

def _parse_obstacle_label_to_text(filepath):
    """Parse an obstacle label .txt file and return the obstacle state of the last frame."""
    with open(filepath, 'r') as f:
        text = f.read()

    # Find all [Frame NNN] blocks; take the last one
    frame_blocks = re.findall(r'\[Frame \d+\](.*?)(?=\[Frame |\Z)', text, re.DOTALL)
    if not frame_blocks:
        return None
    last_frame = frame_blocks[-1]

    def parse_field(name):
        m = re.search(rf'{name}\s*=\s*(\S+)', last_frame)
        return m.group(1) if m else None

    free_front  = parse_field('FREE_FRONT')
    free_left   = parse_field('FREE_LEFT')
    free_right  = parse_field('FREE_RIGHT')
    col_front   = parse_field('COLLIDE_STEP_FRONT')
    col_left    = parse_field('COLLIDE_STEP_LEFT')
    col_right   = parse_field('COLLIDE_STEP_RIGHT')
    best_dir    = (parse_field('BEST_DIR') or 'UNKNOWN').lower()

    def blocked_str(collide, free):
        if collide == 'YES':
            return f"blocked ({free} clearance)"
        return f"clear ({free} clearance)"

    answer = (
        f"Front: {blocked_str(col_front, free_front)}. "
        f"Left: {blocked_str(col_left, free_left)}. "
        f"Right: {blocked_str(col_right, free_right)}. "
        f"Best direction: {best_dir}."
    )
    return answer

class SceneQADataset(data.Dataset):
    """Stage I training samples: a semantic scene QA pair or the spatial (obstacle) state of the last frame,
    with the 3D scene and the video features."""

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        min_motion_length,
        unit_length=4,
        fps=20,
        tiny=False,
        debug=False,
        code_path='TOKENS',
        **kwargs,
    ):
        self.unit_length = unit_length
        self.num_3d_points = kwargs['cfg'].DATASET.NUM_3D_POINTS
        # obstacle labels directory (optional; when set, only clips with labels are loaded)
        obstacle_labels_dir = kwargs['cfg'].DATASET.NYMERIA.get('OBSTACLE_LABELS_PATH', None)

        # Data mean and std
        self.mean = mean
        self.std = std

        # Data path
        split = 'train'
        split_file = pjoin(data_root, split + '.txt')
        motion_dir = pjoin(data_root, code_path)
        text_dir = pjoin(data_root, 'texts')
        pose_dir = pjoin(data_root, 'new_joint_vecs')
        video_feat_dir = pjoin(data_root, 'video_features')
        video_text_dir = pjoin(data_root, 'video_texts')
        scene_feature_dir = pjoin(data_root, '3d_features_scene', 'voxelized_features')
        scene_voxel_dir = pjoin(data_root, '3d_features_scene', 'voxelized_voxels')
        head_pose_voxel_dir = pjoin(data_root, 'global_head_poses')

        self.scene_list = [scene.split('.')[0] for scene in sorted(os.listdir(scene_feature_dir))]
        
        self.scene_info = {}
        with open(pjoin(data_root, 'scene_info.txt'), 'r') as f :
            for line in f.readlines() :
                self.scene_info[line.split(',')[0]] = line.split(',')[-1].strip()
        
        instructions = pjoin(data_root, 'template_pretrain_with_sceneqa_obstacle.json')

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
            try:
                # Load motion tokens
                if self.scene_info[name] not in self.scene_list :
                    continue
                # Load obstacle label if available; None if missing (handled in __getitem__)
                obstacle_answer = None
                if obstacle_labels_dir is not None:
                    label_path = pjoin(obstacle_labels_dir, f'{name}.txt')
                    if os.path.exists(label_path):
                        obstacle_answer = _parse_obstacle_label_to_text(label_path)
                m_token_list = np.load(pjoin(motion_dir, f'{name}.npy'))
                past_pose = np.load(pjoin(pose_dir, f'{name}.npy'))
                
                head_pose_voxel_rot = np.load(pjoin(head_pose_voxel_dir, f'{name}.npz'))
                global_head_voxel = head_pose_voxel_rot['voxel']
                global_head_rot = head_pose_voxel_rot['rotation']
                
                # Read video feature path
                video_feat_path = pjoin(video_feat_dir, f'{name}.pt')
                if not os.path.exists(video_feat_path):
                    continue
                # Read video text
                scene_qa = json.load(open(pjoin(video_text_dir, name, 'qa_dataset.final.filtered.json'), 'r'))
                if len(scene_qa) == 0:
                    continue
                scene_qa_data = []
                for qa in scene_qa:
                    scene_qa_dict = {}
                    scene_qa_dict['question'] = qa['question']
                    scene_qa_dict['answer'] = qa['answer']
                    scene_qa_data.append(scene_qa_dict)
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
                                m_token_list_new = [
                                    tokens[int(f_tag * fps / unit_length
                                                ):int(to_tag * fps /
                                                        unit_length)]
                                    for tokens in m_token_list
                                    if int(f_tag * fps / unit_length) <
                                    int(to_tag * fps / unit_length)
                                ]

                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = '%s_%f_%f' % (name, f_tag,
                                                            to_tag)

                                data_dict[new_name] = {
                                    'm_token_list': m_token_list_new,
                                    'text': [text_dict],
                                    'scene_qa': scene_qa_data,
                                    'video_feat_path': video_feat_path,
                                    'past_pose': past_pose,
                                    'global_head_voxel': global_head_voxel,
                                    'global_head_rotation': global_head_rot,
                                    'obstacle_answer': obstacle_answer,
                                }
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag and m_token_list.shape[1] >= int(min_motion_length // unit_length) :
                    data_dict[name] = {
                        'm_token_list': m_token_list,
                        'text': text_data,
                        'scene_qa': scene_qa_data,
                        'video_feat_path': video_feat_path,
                        'past_pose': past_pose,
                        'global_head_voxel': global_head_voxel,
                        'global_head_rotation': global_head_rot,
                        'obstacle_answer': obstacle_answer,
                    }
                    new_name_list.append(name)
            except:
                pass

        self.data_dict = data_dict
        self.name_list = new_name_list
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])
        
        self.task_probs = np.array(kwargs['cfg'].model.params.task_probs, dtype=np.float32)
        self.task_probs /= self.task_probs.sum()

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
                
    def __len__(self):
        return len(self.name_list) * len(self.tasks)

    def __getitem__(self, item):
        data_idx = item % len(self.name_list)
        task_idx = np.random.choice(len(self.tasks), p=self.task_probs)

        data_index = self.name_list[data_idx]
        data = self.data_dict[data_index]
        m_token_list, text_list = data['m_token_list'], data['text']

        past_pose = data['past_pose']
        
        global_head_voxel = data['global_head_voxel']
        global_head_rot = data['global_head_rotation']
        
        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data['caption']
        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        coin = np.random.choice([False, False, True])

        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]

        m_tokens_len = m_tokens.shape[0]

        tasks = self.tasks[task_idx]
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

        # For obstacleqa task, skip samples that have no obstacle label
        if tasks.get('class') == 'obstacleqa' and not obstacle_answer:
            return None
        
        return caption, m_tokens, m_tokens_len, None, None, None, None, all_captions, tasks, pc, pc_embed, pc_mask, pc_len, video_embed, video_lengths, scene_q, scene_a, past_pose, global_head_voxel, global_head_rot, min_coord, obstacle_answer
