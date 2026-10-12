"""Shared helpers for controlled SNAPMoGen training experiments."""

import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def resolve_snap_split(data_root, split_name):
    """Resolve a split name under data_split_info, or an explicit path."""
    split_path = Path(split_name)
    if not split_path.is_absolute() and split_path.parent == Path("."):
        split_path = Path(data_root) / "data_split_info" / split_path
    if not split_path.is_file():
        raise FileNotFoundError(f"SNAPMoGen split not found: {split_path}")
    return str(split_path)


@contextmanager
def seeded_evaluation(seed, device):
    """Give every checkpoint the same evaluation draws without altering training RNG."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def load_vae_for_inference(vae, state_dict):
    """Load all inference weights while tolerating old feature-loss projections.

    Projection layers are used only by the VAE training loss. Older SNAP
    checkpoints project both features to 64 dimensions, so their projection
    weights cannot fit the current reverse_proj layers.
    """
    def is_projection(name):
        return name.startswith(("linear_proj.", "linear_proj_z.", "linear_proj_aux."))

    target_state = vae.state_dict()
    for name, tensor in target_state.items():
        if is_projection(name):
            continue
        if name not in state_dict:
            raise RuntimeError(f"VAE checkpoint is missing inference weight: {name}")
        if state_dict[name].shape != tensor.shape:
            raise RuntimeError(
                f"VAE checkpoint shape mismatch for {name}: "
                f"{tuple(state_dict[name].shape)} vs {tuple(tensor.shape)}"
            )
    unexpected = [name for name in state_dict if name not in target_state]
    if any(not is_projection(name) for name in unexpected):
        raise RuntimeError(f"Unexpected VAE checkpoint weights: {unexpected}")
    inference_state = {
        name: tensor for name, tensor in state_dict.items() if not is_projection(name)
    }
    missing, unexpected = vae.load_state_dict(inference_state, strict=False)
    if any(not is_projection(name) for name in missing + unexpected):
        raise RuntimeError(
            f"Could not load VAE inference weights: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return [name for name in state_dict if is_projection(name)]


def latent_sample_quality(latent, max_abs_threshold):
    """Return a per-sample keep mask, maxima, and finite mask."""
    flat = latent.reshape(latent.shape[0], -1)
    finite = torch.isfinite(flat).all(dim=1)
    max_abs = flat.abs().amax(dim=1)
    keep = finite & (max_abs <= max_abs_threshold)
    return keep, max_abs, finite


def take_samples(values, indices):
    if torch.is_tensor(values):
        return values[indices]
    positions = indices.tolist()
    return [values[position] for position in positions]


def should_evaluate_epoch(epoch, base_every, window_start, window_end, window_every):
    interval = window_every if window_start <= epoch <= window_end else base_every
    return epoch % interval == 0
