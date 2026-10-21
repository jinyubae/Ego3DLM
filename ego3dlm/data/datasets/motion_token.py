import numpy as np
from torch.utils import data
import codecs as cs
from os.path import join as pjoin


class MotionTokenizeDataset(data.Dataset):
    """All motions of a split, one per item, for scripts/tokenize_motion.py."""

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        
        # Data mean and std
        self.mean = mean
        self.std = std
        
        # Data path
        split_file = pjoin(data_root, split + '.txt')
        motion_dir = pjoin(data_root, 'new_joint_vecs')

        # Data id list
        self.id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())
                
        new_name_list = []
        length_list = []
        data_dict = {}
        for name in self.id_list:
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                if (len(motion)) <  self.min_motion_length or (len(motion) >= self.max_motion_length):
                    continue

                data_dict[name] = {'motion': motion,
                                'length': len(motion),
                                'name': name}
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list
        self.nfeats = motion.shape[-1]
    
    
    def __len__(self):
        return len(self.data_dict)  
        
    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length
        motion = motion[:m_length]

        # Z Normalization
        motion = (motion - self.mean) / self.std

        return name, motion, m_length, True, True, True, True, True, True
