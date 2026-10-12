import os
import random
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from models.Causal_DiT_snap import dit_mlarge
from models.VAE_snap import vae
from utils.config_utils import apply_config_to_cfg, load_yaml_config
from utils.datasets_snap import SnapText2MotionDataset, collate_fn
from utils.eval_utils_snap import evaluation_dit_snap
from utils.snap_config.load_config import load_config
from utils.snap_evaluator.evaluator_wrapper import EvaluatorWrapper
from utils.snap_training import load_vae_for_inference

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SNAP_FULL_DIM = 296
SNAP_EVAL_FEATURE_DIM = 148
DIT_BUILDERS = {"dit_mlarge": dit_mlarge}


def build_dit_model(cfg: DictConfig, **kwargs):
    dit_arch = str(cfg.get("dit_arch", "dit_mlarge"))
    try:
        dit_builder = DIT_BUILDERS[dit_arch]
    except KeyError as exc:
        supported_arches = ", ".join(sorted(DIT_BUILDERS))
        raise ValueError(
            f"Unsupported dit_arch={dit_arch}. Supported values: {supported_arches}"
        ) from exc
    print(f"Using DiT architecture: {dit_arch}")
    return dit_builder(**kwargs)


class SnapFirstDimsDataset(Dataset):
    def __init__(self, base_dataset: Dataset, feature_dim: int):
        self.base_dataset = base_dataset
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample_items = list(sample)
        sample_items[1] = sample_items[1][..., : self.feature_dim]
        return tuple(sample_items)


def resolve_snap_feature_dim(
    cfg: DictConfig,
    *,
    default_use_feature_dim: bool = False,
    require_eval_dim: bool = False,
) -> int:
    use_feature_dim = bool(cfg.get("snap_use_feature_dim", default_use_feature_dim))
    if use_feature_dim:
        dim_pose = int(cfg.get("snap_feature_dim", SNAP_EVAL_FEATURE_DIM))
    else:
        dim_pose = SNAP_FULL_DIM

    if not 0 < dim_pose <= SNAP_FULL_DIM:
        raise ValueError(
            f"snap_feature_dim must be in [1, {SNAP_FULL_DIM}], got {dim_pose}"
        )
    if require_eval_dim and dim_pose < SNAP_EVAL_FEATURE_DIM:
        raise ValueError(
            f"snap_feature_dim must be >= {SNAP_EVAL_FEATURE_DIM} for SNAP metrics, "
            f"got {dim_pose}"
        )
    return dim_pose


@hydra.main(version_base=None, config_path="../conf", config_name="eval_ae_dit_snap")
def main(cfg: DictConfig) -> None:
    dit_config_path = pjoin(
        cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "config.yaml"
    )
    dit_config = load_yaml_config(dit_config_path)
    if dit_config:
        cfg = apply_config_to_cfg(
            cfg,
            dit_config,
            (
                "dataset_name",
                "output_dim",
                "snap_max_motion_length",
                "snap_unit_length",
                "num_frame_per_block",
                "snap_eval_batch_size",
                "snap_text_encoder_type",
                "snap_text_encoder_name",
                "snap_text_max_length",
                "snap_text_encoder_local_files_only",
                "snap_feature_dim",
                "snap_use_feature_dim",
            ),
        )
        print(f"Loaded DiT config from {dit_config_path}")
    else:
        raise ValueError(
            f"No saved DiT config found at {dit_config_path}. Using Hydra config."
        )
    ae_config_path = pjoin(
        cfg.checkpoints_dir, cfg.dataset_name, cfg.ae_name, "config.yaml"
    )
    ae_config = load_yaml_config(ae_config_path)
    if ae_config:
        cfg = apply_config_to_cfg(
            cfg,
            ae_config,
            ("output_dim", "snap_feature_dim", "snap_use_feature_dim"),
        )
        print(f"Loaded AE config from {ae_config_path}")
    else:
        print(f"No saved AE config found at {ae_config_path}. Using Hydra config.")

    torch.backends.cudnn.benchmark = False
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if cfg.dataset_name == "snap":
        data_root = f"{cfg.dataset_dir}/SnapMoGen/"
        dim_pose = resolve_snap_feature_dim(cfg, require_eval_dim=True)
    else:
        raise NotImplementedError("eval_ae_dit_snap.py only supports dataset_name=snap")

    motion_dir = pjoin(data_root, "renamed_feats")
    text_file = pjoin(data_root, "all_caption_clean.json")
    mean = np.load(pjoin(data_root, "meta_data/mean.npy"))
    std = np.load(pjoin(data_root, "meta_data/std.npy"))

    max_motion_length = int(cfg.get("snap_max_motion_length", 320))
    print(f"max_motion_length: {max_motion_length}")
    unit_length = int(cfg.get("snap_unit_length", 4))
    eval_batch_size = int(cfg.get("snap_eval_batch_size", 100))
    split_file = pjoin(data_root, "data_split_info/test_ids.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"SNAPMoGen test split not found: {split_file}")

    use_m_length = cfg.get("use_m_length", False)

    eval_dataset = SnapText2MotionDataset(
        mean,
        std,
        split_file,
        motion_dir,
        text_file,
        unit_length,
        max_motion_length,
        max_text_length=cfg.get("snap_text_max_length", 120),
        min_motion_length=cfg.get("snap_min_motion_length", 128),
    )
    if dim_pose < SNAP_FULL_DIM:
        eval_dataset = SnapFirstDimsDataset(eval_dataset, feature_dim=dim_pose)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        shuffle=True,
    )

    ae = vae(
        input_width=dim_pose,
        output_emb_width=cfg.output_dim,
        hidden_size=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
        clip_range=[-30, 20],
    )
    cfg.ae_ckpt_name = cfg.get("ae_ckpt_name", "latest.tar")
    ae_ckpt_path = pjoin(
        cfg.checkpoints_dir,
        cfg.dataset_name,
        cfg.ae_name,
        "model",
        cfg.ae_ckpt_name,
    )
    if not os.path.exists(ae_ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {ae_ckpt_path}")
    ae_ckpt = torch.load(ae_ckpt_path, map_location="cpu")
    print (f"Loading AE checkpoint from {ae_ckpt_path} at epoch {ae_ckpt['ep']}")
    load_vae_for_inference(ae, ae_ckpt["ae"])

    snap_text_max_length = int(cfg.get("snap_text_max_length", 120))
    snap_text_encoder_local_files_only = cfg.get(
        "snap_text_encoder_local_files_only", False
    )
    snap_text_encoder_type = cfg.get(
        "snap_text_encoder_type", cfg.get("snap_text_encoder_name", "snap_t5")
    )

    ema_model = build_dit_model(cfg,
        input_dim=ae.output_emb_width,
        cond_mode="text",
        max_length=max_motion_length // ae.down_scale,
        uncertainty_scale=cfg.uncertainty_scale,
        sampling_timesteps=cfg.sampling_timesteps,
        num_frame_per_block=5
        if "mix" in cfg.name
        else int(cfg.get("num_frame_per_block", 5)),
        text_encoder_type=snap_text_encoder_type,
        text_encoder_max_length=snap_text_max_length,
        text_encoder_local_files_only=snap_text_encoder_local_files_only,
    )

    dit_ckpt_name = cfg.get("dit_ckpt_name", "latest.tar")
    dit_ckpt_path = pjoin(
        cfg.checkpoints_dir,
        cfg.dataset_name,
        cfg.name,
        "model",
        dit_ckpt_name,
    )
    if not os.path.exists(dit_ckpt_path):
        raise FileNotFoundError(f"DiT checkpoint not found: {dit_ckpt_path}")

    dit_ckpt = torch.load(dit_ckpt_path, map_location="cpu")
    print (f"Loading DiT checkpoint from {dit_ckpt_path} at epoch {dit_ckpt['ep']}")
    missing_keys, unexpected_keys = ema_model.load_state_dict(
        dit_ckpt["ema_model"], strict=False
    )
    assert len(unexpected_keys) == 0
    assert all([k.startswith("clip_model.") for k in missing_keys])

    device = torch.device(f"cuda:{cfg.gpu}" if cfg.gpu != "cpu" else "cpu")

    eval_cfg_path = pjoin(
        data_root,
        "SnapMoGen",
        "checkpoint_dir",
        "snapmogen",
        "evaluator",
        "eval_klde-5_late-5_nlayer6_norm",
        "evaluator.yaml",
    )
    print("Loading SNAP evaluator config from", eval_cfg_path)
    eval_cfg = load_config(eval_cfg_path)
    eval_cfg.exp.root_ckpt_dir = pjoin(data_root, "SnapMoGen", "checkpoint_dir")
    eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)

    ema_model.eval()
    ema_model.to(device)
    ae.eval()
    ae.to(device)

    out_dir = cfg.get("eval_output_dir") or pjoin(
        cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "eval"
    )
    os.makedirs(out_dir, exist_ok=True)
    sampler = str(cfg.get("sampler", "euler"))
    log_path = pjoin(out_dir, f"eval_{cfg.cfg}_{sampler}.log")

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []

    repeat_time = 10
    for _ in range(repeat_time):
        metric = evaluation_dit_snap(
            eval_loader,
            ema_model,
            ae,
            eval_wrapper,
            device,
            cond_scale=float(cfg.cfg),
            down_scale=ae.down_scale,
            use_m_length=use_m_length,
            sampler=sampler,
        )
        fid.append(metric[0])
        div.append(metric[1])
        top1.append(metric[2])
        top2.append(metric[3])
        top3.append(metric[4])
        matching.append(metric[5])

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)

    with open(log_path, "w") as f:
        protocol = (
            f"VAE checkpoint: {ae_ckpt_path}\n"
            f"DiT checkpoint: {dit_ckpt_path}\n"
            f"Training split: {dit_config.get('snap_train_split', 'unknown')}\n"
            f"Evaluation split: {split_file}\n"
            f"Maximum motion length: {max_motion_length}\n"
            f"Feature dimension: {dim_pose}\n"
            f"Guidance: {cfg.cfg}\n"
            f"Evaluation seed: {cfg.seed}\n"
            f"Passes: {repeat_time}\n"
        )
        print(protocol, file=f, flush=True)
        print("final result")
        print("final result", file=f, flush=True)

        msg_final = (
            f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid, ddof=1) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div, ddof=1) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1, ddof=1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2, ddof=1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3, ddof=1) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching, ddof=1) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
        )

        print(msg_final)
        print(msg_final, file=f, flush=True)


if __name__ == "__main__":
    main()
