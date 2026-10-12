import codecs as cs
import json
import random
from os.path import join as pjoin

import numpy as np
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


#################################################################################
#                                  Collate Function                             #
#################################################################################
def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    return default_collate(batch)


#################################################################################
#                                      Datasets                                 #
#################################################################################
class SnapAEDataset(data.Dataset):
    def __init__(
        self, mean, std, motion_dir, window_size, split_file, window_stride=10
    ):
        self.data = []
        self.lengths = []
        id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            if "#" in name:
                motion_name = name.split("#")[0]
                start_frame = int(name.split("#")[1])
                end_frame = int(name.split("#")[2])
            else:
                motion_name = name
                start_frame = 0
                end_frame = -1
            motion = np.load(pjoin(motion_dir, motion_name + ".npy"))
            motion = motion[start_frame:end_frame]
            if motion.shape[0] < window_size:
                continue
            if np.isnan(motion).any() or np.isinf(motion).any():
                continue
            self.lengths.append(motion.shape[0] - window_size)
            self.data.append(motion)
        self.cumsum = np.cumsum([0] + self.lengths)
        self.window_size = window_size
        self.window_stride = window_stride
        self.mean = mean
        self.std = std

        self.indices = self._create_indices()
        print(
            "Total number of motions {}, snippets {}".format(
                len(self.data), len(self.indices)
            )
        )

    def _create_indices(self):
        # Create a list of tuples (sample_index, time_index) for data retrieval
        indices = []
        for i, sample in enumerate(self.data):
            for start_idx in range(
                0, len(sample) - self.window_size + 1, self.window_stride
            ):
                indices.append((i, start_idx))
        return indices

    def __len__(self):
        # Return the total number of windows
        return len(self.indices)

    def __getitem__(self, idx):
        # Retrieve a window of data based on the index
        sample_idx, time_idx = self.indices[idx]
        sample = self.data[sample_idx]
        motion = sample[time_idx : time_idx + self.window_size]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class SnapText2MotionDataset(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        motion_dir,
        text_file,
        unit_length,
        max_motion_length,
        max_text_length,
        min_motion_length=128,
        is_mesh=False,
        return_id=False,
    ):
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_length
        self.unit_length = unit_length
        self.min_motion_length = min_motion_length
        self.return_id = return_id

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        with open(text_file) as f:
            d = json.load(f)

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            motion_name = name.split("#")[0]
            motion = np.load(pjoin(motion_dir, motion_name + ".npy"))
            motion = np.expand_dims(motion, axis=1)
            if np.isnan(motion).any() or np.isinf(motion).any():
                continue
            start_frame = int(name.split("#")[1])
            end_frame = int(name.split("#")[2])
            motion = motion[start_frame:end_frame]
            if len(motion) < self.min_motion_length:
                continue
            text_data = d[name]["manual"] + d[name]["gpt"]
            data_dict[name] = {
                "motion": motion,
                "length": len(motion),
                "text": text_data,
            }
            new_name_list.append(name)
            length_list.append(len(motion))

        name_list, length_list = new_name_list, length_list
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        else:
            return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        else:
            return data * std + mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        sample_id = self.name_list[idx]
        data = self.data_dict[sample_id]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        caption = random.choice(text_list)
        m_length = (
            len(motion)
            if len(motion) < self.max_motion_length
            else self.max_motion_length
        )

        m_length = (m_length // self.unit_length) * self.unit_length

        start_idx = random.randint(0, len(motion) - m_length)
        motion = motion[start_idx : start_idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros(
                        (
                            self.max_motion_length - m_length,
                            motion.shape[1],
                            motion.shape[2],
                        )
                    ),
                ],
                axis=0,
            )

        if self.return_id:
            return caption, motion, m_length, sample_id
        return caption, motion, m_length
