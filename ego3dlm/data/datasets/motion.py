import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin


class MotionDataset(data.Dataset):
    """Motion features and narrations of a split (base of the Stage 0 datasets)."""

    def __init__(
        self,
        data_root,
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
        split_file = pjoin(data_root, split + '.txt')
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')

        # Data id list
        self.id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())
        # Debug mode
        if tiny or debug:
            enumerator = enumerate(self.id_list)
            maxdata = 10
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
            for idx, name in enumerator:
                if len(new_name_list) > maxdata:
                    break
                try:
                    motion = np.load(pjoin(motion_dir, name + ".npy"))
                    if (len(motion)) < self.min_motion_length or (len(motion)
                                                                  >= 1000):
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

                    if flag and len(motion) > min_motion_length:
                        data_dict[name] = {
                            'motion': motion,
                            "length": len(motion),
                            'text': text_data
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

