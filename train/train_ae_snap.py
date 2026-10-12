import os
import random
import time
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

import hydra
import numpy as np
import torch
import torch.optim as optim
from einops import rearrange
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

import wandb
from Part_TMR.models.motion_aux_snap import MotionAuxModel
from models.VAE_snap import vae
from utils.config_utils import save_args_yaml
from utils.datasets_snap import SnapAEDataset, SnapText2MotionDataset
from utils.eval_utils_snap import evaluation_ae_snap
from utils.parts_utils_snap import whole2parts_snap_batch
from utils.snap_config.load_config import load_config
from utils.snap_evaluator.evaluator_wrapper import EvaluatorWrapper
from utils.snap_training import resolve_snap_split, seeded_evaluation
from utils.train_utils import def_value, print_current_loss, save, update_lr_warm_up
from utils.wandb_utils import namespace_to_dict

from utils.snap_codes.motion_process_bvh import recover_pos_from_rot
from utils.snap_codes.common.skeleton import Skeleton
from utils.snap_codes.bvh_io import load as bvh_load

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SNAP_FULL_DIM = 296
SNAP_EVAL_FEATURE_DIM = 148


def forward_kinematic_func(dataset, data, skeleton):
    mean = torch.from_numpy(dataset.mean[: data.shape[-1]]).to(data.device)
    std = torch.from_numpy(dataset.std[: data.shape[-1]]).to(data.device)
    motions = (data * std) + mean
    global_pos = recover_pos_from_rot(motions.float(), joints_num=24, skeleton=skeleton)
    return global_pos


class SnapFirstDimsDataset(Dataset):
    def __init__(self, base_dataset: Dataset, feature_dim: int):
        self.base_dataset = base_dataset
        self.feature_dim = feature_dim
        if hasattr(base_dataset, "mean"):
            self.mean = base_dataset.mean[:feature_dim]
        if hasattr(base_dataset, "std"):
            self.std = base_dataset.std[:feature_dim]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if isinstance(sample, tuple):
            sample_items = list(sample)
            sample_items[1] = sample_items[1][..., : self.feature_dim]
            return tuple(sample_items)
        return sample[..., : self.feature_dim]


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


def pad_motion_to_snap_full_dim(
    motion: torch.Tensor, target_dim: int = SNAP_FULL_DIM
) -> torch.Tensor:
    if motion.shape[-1] == target_dim:
        return motion
    padded = torch.zeros(
        *motion.shape[:-1], target_dim, device=motion.device, dtype=motion.dtype
    )
    padded[..., : motion.shape[-1]] = motion
    return padded


@hydra.main(version_base=None, config_path="../conf", config_name="train_ae_snap")
def main(cfg: DictConfig) -> None:
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    if cfg.dataset_name == "snap":
        data_root = f"{cfg.dataset_dir}/SnapMoGen/"
        dim_pose = resolve_snap_feature_dim(cfg, require_eval_dim=True)
    else:
        raise NotImplementedError("train_ae_snap.py only supports dataset_name=snap")

    motion_dir = pjoin(data_root, "renamed_feats")
    text_file = pjoin(data_root, "all_caption_clean.json")
    mean = np.load(pjoin(data_root, "meta_data/mean.npy"))
    std = np.load(pjoin(data_root, "meta_data/std.npy"))

    train_split_file = resolve_snap_split(
        data_root, cfg.get("snap_train_split", "train_ids_clean.txt")
    )
    val_split_file = resolve_snap_split(
        data_root, cfg.get("snap_val_split", "val_ids.txt")
    )
    print(f"Training split: {train_split_file}")
    print(f"Validation split: {val_split_file}")

    train_dataset = SnapAEDataset(
        mean, std, motion_dir, cfg.window_size, train_split_file, window_stride=10
    )
    val_dataset = SnapAEDataset(
        mean,
        std,
        motion_dir,
        cfg.window_size,
        val_split_file,
        window_stride=10,
    )
    if dim_pose < SNAP_FULL_DIM:
        train_dataset = SnapFirstDimsDataset(train_dataset, feature_dim=dim_pose)
        val_dataset = SnapFirstDimsDataset(val_dataset, feature_dim=dim_pose)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    max_motion_length = int(cfg.get("snap_max_motion_length", 320))
    unit_length = int(cfg.get("snap_unit_length", 4))
    eval_batch_size = int(cfg.get("snap_eval_batch_size", 100))
    eval_dataset = SnapText2MotionDataset(
        mean,
        std,
        val_split_file,
        motion_dir,
        text_file,
        unit_length,
        max_motion_length,
        max_text_length=int(cfg.get("snap_text_max_length", 120)),
        min_motion_length=int(cfg.get("snap_min_motion_length", 128)),
    )
    if dim_pose < SNAP_FULL_DIM:
        eval_dataset = SnapFirstDimsDataset(eval_dataset, feature_dim=dim_pose)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=cfg.num_workers,
        drop_last=False,
        shuffle=False,
    )
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    run_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name)
    model_dir = pjoin(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    save_args_yaml(cfg, pjoin(run_dir, "config.yaml"))

    kl_base_weight = cfg.kl_weight
    if cfg.use_aux_model and cfg.vf_weight <= 0:
        raise ValueError("use_aux_model=true requires vf_weight > 0")
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
        root_loss_weight=cfg.root_loss_weight,
        vel_loss_weight=cfg.vel_loss_weight,
        vf_weight=cfg.vf_weight,
        adaptive_vf=cfg.use_adaptive_vf,
        kl_weight=kl_base_weight,
        reverse_proj=cfg.get("reverse_proj", True),
        nll_reduction=cfg.get("nll_reduction", "sum"),
        vel_loss_mode=cfg.get("vel_loss_mode", "feature_prefix"),
    )

    pc_vae = sum(param.numel() for param in ae.parameters())
    print("Total parameters of all models: {}M".format(pc_vae / 1000_000))

    device = torch.device(f"cuda:{cfg.gpu}" if cfg.gpu != "cpu" else "cpu")

    if cfg.use_aux_model:
        if not os.path.isfile(cfg.snap_aux_model_path):
            raise FileNotFoundError(
                f"SNAP auxiliary model not found: {cfg.snap_aux_model_path}"
            )
        tmr_model = MotionAuxModel(
            motion_embedding_dims=512,
            projection_dims=256,
            mode="snapmogen",
        )

        print("Loading Part-TMR model from", cfg.snap_aux_model_path)
        state_dict = torch.load(
            cfg.snap_aux_model_path, map_location="cpu", weights_only=True
        )
        tmr_model.load_part_tmr_state_dict(state_dict)
        del state_dict

        for param in tmr_model.parameters():
            param.requires_grad = False

        tmr_model.to(device)
        tmr_model.eval()

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

    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    wandb_mode = "disabled" if cfg.entity == "your_wandb_entity" else "online"
    wandb.init(
        name=cfg.name,
        project=cfg.project_name,
        entity=cfg.entity,
        config=namespace_to_dict(cfg),
        mode=wandb_mode,
    )
    criterion = torch.nn.SmoothL1Loss()

    ae.to(device)
    optimizer = optim.AdamW(
        ae.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.milestones, gamma=cfg.lr_decay
    )
    epoch = 0
    it = 0
    if cfg.is_continue:
        checkpoint = torch.load(pjoin(model_dir, "latest.tar"), map_location=device)
        ae.load_state_dict(checkpoint["ae"])
        optimizer.load_state_dict(checkpoint["opt_ae"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch, it = checkpoint["ep"] + 1, checkpoint["total_it"]
        print("Load model epoch:%d iterations:%d" % (epoch, it))

    start_time = time.time()
    total_iters = cfg.epoch * len(train_loader)
    print(f"Total Epochs: {cfg.epoch}, Total Iters: {total_iters}")
    print(
        "Iters Per Epoch, Training: %04d, Validation: %03d"
        % (len(train_loader), len(val_loader))
    )

    current_lr = cfg.lr
    logs = defaultdict(def_value, OrderedDict())

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe = (
        1000,
        0,
        0,
        0,
        0,
        100,
        100,
    )
    if cfg.is_continue and "best_metrics" in checkpoint:
        best_metrics = checkpoint["best_metrics"]
        best_fid = best_metrics["fid"]
        best_div = best_metrics["diversity"]
        best_top1 = best_metrics["top1"]
        best_top2 = best_metrics["top2"]
        best_top3 = best_metrics["top3"]
        best_matching = best_metrics["matching"]
        best_mpjpe = best_metrics["mpjpe"]
    eval_seed = int(cfg.get("eval_seed", cfg.seed))

    while epoch < cfg.epoch:
        ae.train()
        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < cfg.warm_up_iter:
                current_lr = update_lr_warm_up(it, cfg.warm_up_iter, optimizer, cfg.lr)

            motions = batch_data.detach().to(device).float()

            if cfg.use_aux_model:
                motions_for_aux = motions
                if dim_pose < SNAP_FULL_DIM:
                    motions_for_aux = pad_motion_to_snap_full_dim(motions_for_aux)
                motions_parts = whole2parts_snap_batch(motions_for_aux)
                motions_parts = [motion.to(device).float() for motion in motions_parts]
                with torch.no_grad():
                    aux_feature = tmr_model.encode_motion(
                        motions_parts, calc_mean=False
                    )
                    aux_feature = aux_feature.detach()
                if (
                    aux_feature.ndim != 3
                    or aux_feature.shape[0] < ae.down_scale
                    or aux_feature.shape[0] % ae.down_scale != 0
                    or aux_feature.shape[1] != motions.shape[0]
                    or aux_feature.shape[2] != 256
                ):
                    raise ValueError(
                        "Unexpected SNAP auxiliary feature shape: "
                        f"{tuple(aux_feature.shape)}"
                    )
            else:
                aux_feature = None

            motions = rearrange(motions, "b l d -> b l 1 d")

            # a naive way of writing gradient accumulation
            accum = cfg.accum
            accum_bs = cfg.batch_size // accum
            if cfg.kl_warmup_iter > 0:
                kl_scale = min(1.0, it / cfg.kl_warmup_iter)
            else:
                kl_scale = 1.0
            current_kl_weight = kl_base_weight * kl_scale

            for kk in range(accum):
                _, loss, loss_items = ae(
                    motions[kk * accum_bs : (kk + 1) * accum_bs],
                    aux_feature=aux_feature[:, kk * accum_bs : (kk + 1) * accum_bs]
                    if aux_feature is not None
                    else None,
                    need_loss=True,
                    kl_weight_override=current_kl_weight,
                    return_loss_components=True,
                )
                logs["lr"] += (optimizer.param_groups[0]["lr"]) / accum
                logs["loss"] += loss.item() / accum
                for tag, value in loss_items.items():
                    logs[tag] += value.item() / accum
                loss = loss / accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at iteration {it}")
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            logs["grad_norm"] += grad_norm.item()

            optimizer.step()
            if it >= cfg.warm_up_iter:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if it % cfg.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    wandb.log({f"Train/{tag}": value / cfg.log_every}, step=it)
                    mean_loss[tag] = value / cfg.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(
                    start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i
                )

        #################################################################################
        #                                      Eval Loop                                #
        #################################################################################
        print("Validation time:")
        ae.eval()
        val_loss_rec = []
        val_loss = []
        with seeded_evaluation(eval_seed, device), torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                motions = batch_data.detach().to(device).float()
                motions = rearrange(motions, "b l d -> b l 1 d")
                pred_motion = ae(motions)

                loss_rec = criterion(pred_motion, motions)
                loss = loss_rec

                val_loss.append(loss.item())
                val_loss_rec.append(loss_rec.item())

        wandb.log({f"Val/loss": sum(val_loss) / len(val_loss)}, step=it)
        wandb.log({f"Val/loss_rec": sum(val_loss_rec) / len(val_loss_rec)}, step=it)
        print(
            "Validation Loss: %.5f, Reconstruction: %.5f"
            % (sum(val_loss) / len(val_loss), sum(val_loss_rec) / len(val_loss_rec))
        )

        with seeded_evaluation(eval_seed + 1, device):
            (
                best_fid,
                best_div,
                best_top1,
                best_top2,
                best_top3,
                best_matching,
                best_mpjpe,
                writer,
            ) = evaluation_ae_snap(
                model_dir,
                eval_loader,
                ae,
                wandb,
                epoch,
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
                metric_prefix="Val",
            )
        wandb.log(
            {
                "Val/FID_best": best_fid,
                "Val/Diversity_best": best_div,
                "Val/top1_best": best_top1,
                "Val/top2_best": best_top2,
                "Val/top3_best": best_top3,
                "Val/matching_best": best_matching,
                "Val/MPJPE": best_mpjpe,
            },
            step=it,
        )
        print(f"best fid {best_fid}")

        save(
            pjoin(model_dir, "latest.tar"),
            epoch,
            ae,
            optimizer,
            scheduler,
            it,
            "ae",
            extra_state={
                "best_metrics": {
                    "fid": best_fid,
                    "diversity": best_div,
                    "top1": best_top1,
                    "top2": best_top2,
                    "top3": best_top3,
                    "matching": best_matching,
                    "mpjpe": best_mpjpe,
                },
                "train_split_file": train_split_file,
                "val_split_file": val_split_file,
                "eval_seed": eval_seed,
            },
        )

        epoch += 1


if __name__ == "__main__":
    main()
