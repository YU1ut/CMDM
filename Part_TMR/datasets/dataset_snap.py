"""SnapMoGen caption and motion dataset for Part-TMR."""

import codecs as cs
import json
import random
from os.path import join as pjoin

import numpy as np
from torch.utils import data
from tqdm import tqdm

from .utils_snap import whole2parts


class SnapMoGenTextMotionDataset(data.Dataset):
    def __init__(
        self,
        cfg,
        mean,
        std,
        split_file,
        eval_mode=False,
        fps=None,
    ):
        self.cfg = cfg
        self.eval_mode = eval_mode
        self.max_motion_length = cfg.dataset.max_motion_length
        window_size = cfg.dataset.get("window_size")
        self.window_size = None if window_size is None else int(window_size)
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError("dataset.window_size must be positive")
        self.fps = fps

        with open(cfg.dataset.text_path) as f:
            caption_db = json.load(f)

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                split_tokens = name.split("#")
                motion_name = split_tokens[0]
                motion = np.load(pjoin(cfg.dataset.motion_dir, motion_name + ".npy"))
                if len(motion.shape) != 2:
                    continue
                if np.isnan(motion).any() or np.isinf(motion).any():
                    continue

                if len(split_tokens) >= 3:
                    start_frame = int(split_tokens[1])
                    end_frame = int(split_tokens[2])
                    motion = motion[start_frame:end_frame]

                if len(motion) == 0:
                    continue
                if self.window_size is not None and len(motion) < self.window_size:
                    continue
                if name not in caption_db:
                    continue

                text_data = []
                captions = caption_db[name].get("manual", []) + caption_db[name].get(
                    "gpt", []
                )
                for idx, caption in enumerate(captions):
                    if eval_mode and idx >= 1:
                        continue
                    if not isinstance(caption, str):
                        continue
                    caption = caption.strip()
                    if not caption:
                        continue
                    text_data.append({"caption": caption, "tokens": []})

                if not text_data:
                    continue

                data_dict[name] = {
                    "motion": motion,
                    "length": len(motion),
                    "text": text_data,
                }
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                continue

        if not new_name_list:
            raise ValueError(f"No valid SnapMoGen samples found in split: {split_file}")

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = name_list

        for key, item in tqdm(data_dict.items()):
            motion = data_dict[key]["motion"]
            motion = (motion - self.mean) / self.std
            motion = np.array(motion, dtype=np.float32)
            data_dict[key]["pre_motion"] = motion
            data_dict[key]["length"] = motion.shape[0]

    def real_len(self):
        return len(self.data_dict)

    def __len__(self):
        return self.real_len() * self.cfg.dataset.times

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["pre_motion"], data["length"], data["text"]

        if self.eval_mode:
            caption = text_list[0]["caption"]
        else:
            text_data = random.choice(text_list)
            caption = text_data["caption"]

        clip_length = self.window_size or self.max_motion_length
        if m_length >= clip_length:
            crop_start = (
                random.randint(0, len(motion) - clip_length)
                if not self.eval_mode
                else 0
            )
            motion = motion[crop_start : crop_start + clip_length]
            m_length = clip_length
        elif self.cfg.preprocess.padding:
            padding_len = clip_length - m_length
            d_feat = motion.shape[1]
            padding_zeros = np.zeros((padding_len, d_feat), dtype=np.float32)
            motion = np.concatenate((motion, padding_zeros), axis=0)

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = whole2parts(
            motion,
            mode="snapmogen",
        )

        return Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, caption, m_length, item
