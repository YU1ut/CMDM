import os
import random
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from models.VAE_snap import vae
from utils.config_utils import apply_config_to_cfg, load_yaml_config
from utils.datasets_snap import SnapText2MotionDataset
from utils.eval_utils_snap import evaluation_ae_snap
from utils.snap_codes.bvh_io import load as bvh_load
from utils.snap_codes.common.skeleton import Skeleton
from utils.snap_codes.motion_process_bvh import recover_pos_from_rot
from utils.snap_config.load_config import load_config
from utils.snap_evaluator.evaluator_wrapper import EvaluatorWrapper
from utils.snap_training import load_vae_for_inference

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SNAP_FULL_DIM = 296
SNAP_EVAL_FEATURE_DIM = 148


class NoOpWriter:
    def log(self, *_args, **_kwargs):
        return None


def forward_kinematic_func(dataset, data, skeleton):
    mean = torch.from_numpy(dataset.mean[: data.shape[-1]]).to(data.device)
    std = torch.from_numpy(dataset.std[: data.shape[-1]]).to(data.device)
    motions = (data * std) + mean
    global_pos = recover_pos_from_rot(
        motions.float(),
        joints_num=24,
        skeleton=skeleton,
    )
    return global_pos


class SnapFirstDimsDataset(Dataset):
    def __init__(self, base_dataset: Dataset, feature_dim: int):
        self.base_dataset = base_dataset
        self.feature_dim = feature_dim
        if hasattr(base_dataset, "mean"):
            self.mean = base_dataset.mean
        if hasattr(base_dataset, "std"):
            self.std = base_dataset.std

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


@hydra.main(version_base=None, config_path="../conf", config_name="eval_ae_snap")
def main(cfg: DictConfig) -> None:
    config_path = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "config.yaml")
    train_config = load_yaml_config(config_path)
    if train_config:
        cfg = apply_config_to_cfg(
            cfg,
            train_config,
            (
                "dataset_name",
                "output_dim",
                "snap_max_motion_length",
                "snap_unit_length",
                "snap_eval_batch_size",
                "snap_feature_dim",
                "snap_use_feature_dim",
            ),
        )
        print(f"Loaded config from {config_path}")
    else:
        print(f"No saved config found at {config_path}. Using Hydra config.")

    torch.backends.cudnn.benchmark = False
    os.environ["OMP_NUM_THREADS"] = "1"
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.dataset_name == "snap":
        data_root = f"{cfg.dataset_dir}/SnapMoGen/"
        dim_pose = resolve_snap_feature_dim(cfg, require_eval_dim=True)
    else:
        raise NotImplementedError("eval_ae_snap_v5.py only supports dataset_name=snap")

    motion_dir = pjoin(data_root, "renamed_feats")
    text_file = pjoin(data_root, "all_caption_clean.json")
    mean = np.load(pjoin(data_root, "meta_data/mean.npy"))
    std = np.load(pjoin(data_root, "meta_data/std.npy"))

    max_motion_length = int(cfg.get("snap_max_motion_length", 320))
    unit_length = int(cfg.get("snap_unit_length", 4))
    eval_batch_size = int(cfg.get("snap_eval_batch_size", 100))
    split_file = pjoin(data_root, "data_split_info/test_ids.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"SNAPMoGen test split not found: {split_file}")

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

    model_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "model")
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
        root_dim=4,
        vel_dim=148,
    )

    ae_ckpt_name = cfg.get("ae_ckpt_name", "latest.tar")
    ae_ckpt_path = pjoin(model_dir, ae_ckpt_name)
    if not os.path.exists(ae_ckpt_path):
        raise FileNotFoundError(f"AE checkpoint not found: {ae_ckpt_path}")

    checkpoint = torch.load(ae_ckpt_path, map_location="cpu")
    load_vae_for_inference(ae, checkpoint["ae"])
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

    template_anim = bvh_load(pjoin(data_root, "renamed_bvhs", "m_ep2_00086.bvh"))
    skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=device)

    out_dir = cfg.get("eval_output_dir") or pjoin(
        cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "eval"
    )
    os.makedirs(out_dir, exist_ok=True)

    ae.eval()
    ae.to(device)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mpjpe = []

    writer = NoOpWriter()
    repeat_time = int(cfg.get("repeat_time", 10))
    for i in range(repeat_time):
        (
            best_fid,
            best_div,
            best_top1,
            best_top2,
            best_top3,
            best_matching,
            best_mpjpe,
        ) = (
            1000,
            0,
            0,
            0,
            0,
            100,
            100,
        )
        (
            best_fid,
            best_div,
            best_top1,
            best_top2,
            best_top3,
            best_matching,
            best_mpjpe,
            _,
        ) = evaluation_ae_snap(
            model_dir,
            eval_loader,
            ae,
            writer,
            i,
            eval_wrapper=eval_wrapper,
            device=device,
            best_fid=best_fid,
            best_div=best_div,
            best_top1=best_top1,
            best_top2=best_top2,
            best_top3=best_top3,
            best_matching=best_matching,
            best_mpjpe=best_mpjpe,
            skeleton=skeleton,
            forward_kinematic_func=forward_kinematic_func,
            save=False,
            draw=False,
        )

        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        mpjpe.append(best_mpjpe.cpu().numpy())

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mpjpe = np.array(mpjpe)

    with open(pjoin(out_dir, "eval.log"), "w") as f:
        print("final result")
        print("final result", file=f, flush=True)

        msg_final = (
            f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"\tMPJPE: {np.mean(mpjpe):.3f}, conf. {np.std(mpjpe) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
        )

        print(msg_final)
        print(msg_final, file=f, flush=True)


if __name__ == "__main__":
    main()
