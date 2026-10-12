"""SnapMoGen body-part conversion used by Part-TMR."""

import numpy as np
import torch

from utils.parts_utils_snap import whole2parts_snap_batch


def whole2parts(motion, mode="snapmogen", window_size=None):
    if mode != "snapmogen":
        raise ValueError(f"Unsupported motion mode: {mode}")
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    if motion.ndim != 2 or motion.shape[-1] != 296:
        raise ValueError(f"Expected (frames, 296) Snap motion, got {motion.shape}")
    if window_size is not None and motion.shape[0] != window_size:
        raise ValueError(f"Expected {window_size} frames, got {motion.shape[0]}")
    return [part.squeeze(0) for part in whole2parts_snap_batch(motion)]
