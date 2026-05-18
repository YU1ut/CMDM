"""
# This code is based on https://github.com/neu-vi/ACMDM
"""

import codecs as cs
import random
from os.path import join as pjoin

import numpy as np
import torch
from einops import rearrange
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from utils.glove import GloVe


#################################################################################
#                                  Collate Function                             #
#################################################################################
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


#################################################################################
#                                      Datasets                                 #
#################################################################################
class AEDataset(data.Dataset):
    def __init__(
        self, mean, std, motion_dir, window_size, split_file, window_stride=10
    ):
        self.data = []
        self.lengths = []
        if "vec" in motion_dir:
            self.use_vec = True
        else:
            self.use_vec = False
        id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if len(motion.shape) == 2 and not self.use_vec:  # B,L,J,3
                    motion = np.expand_dims(motion, axis=0)
                if motion.shape[0] < window_size:
                    continue
                if np.isnan(motion).any() or np.isinf(motion).any():
                    continue
                self.lengths.append(motion.shape[0] - window_size)
                self.data.append(motion)
            except Exception as e:
                pass
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


class Text2MotionDataset(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        dataset_name,
        motion_dir,
        text_dir,
        unit_length,
        max_motion_length,
        max_text_length,
        evaluation=False,
        is_mesh=False,
    ):
        self.evaluation = evaluation
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_length
        self.unit_length = unit_length
        min_motion_len = 40 if dataset_name == "t2m" else 24
        if "vec" in motion_dir:
            self.use_vec = True
        else:
            self.use_vec = False

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if len(motion.shape) == 2 and not self.use_vec:
                    motion = np.expand_dims(motion, axis=0)
                if self.use_vec:
                    motion = np.expand_dims(motion, axis=1)
                if is_mesh:
                    if (len(motion)) < min_motion_len:
                        continue
                else:
                    if (len(motion)) < min_motion_len or (len(motion) >= 200):
                        continue
                if np.isnan(motion).any() or np.isinf(motion).any():
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20) : int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (
                                    len(n_motion) >= 200
                                ):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                                while new_name in data_dict:
                                    new_name = (
                                        random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                        + "_"
                                        + name
                                    )
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass
        if self.evaluation:
            self.w_vectorizer = GloVe("./glove", "our_vab")
            name_list, length_list = zip(
                *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
            )
        else:
            name_list, length_list = new_name_list, length_list
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        if self.evaluation:
            self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

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
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if self.evaluation:
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                sent_len = len(tokens)
                tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[: self.max_text_len]
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

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]

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
        elif m_length > self.max_motion_length:
            idx = random.randint(0, m_length - self.max_motion_length)
            motion = motion[idx : idx + self.max_motion_length]
            m_length = self.max_motion_length
        if self.evaluation:
            return (
                word_embeddings,
                pos_one_hots,
                caption,
                sent_len,
                motion,
                m_length,
                "_".join(tokens),
            )
        else:
            return caption, motion, m_length
