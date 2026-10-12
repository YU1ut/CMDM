import copy
import csv
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
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

import wandb
from models.Causal_DiT_snap import dit_mlarge
from models.VAE_snap import vae
from utils.config_utils import save_args_yaml
from utils.datasets_snap import SnapText2MotionDataset
from utils.eval_utils_snap import evaluation_dit_snap
from utils.snap_config.load_config import load_config
from utils.snap_evaluator.evaluator_wrapper import EvaluatorWrapper
from utils.snap_training import (
    latent_sample_quality,
    load_vae_for_inference,
    resolve_snap_split,
    seeded_evaluation,
    should_evaluate_epoch,
    take_samples,
)
from utils.train_utils import (
    def_value,
    print_current_loss,
    save,
    update_ema,
    update_lr_warm_up,
)
from utils.wandb_utils import namespace_to_dict

SNAP_FULL_DIM = 296
SNAP_EVAL_FEATURE_DIM = 148
DIT_BUILDERS = {"dit_mlarge": dit_mlarge}


def record_rejected_sample(path, iteration, sample_id, reason, latent_max_abs, raw_loss=""):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(
                ["iteration", "sample_id", "reason", "latent_max_abs", "raw_loss"]
            )
        writer.writerow([iteration, sample_id, reason, latent_max_abs, raw_loss])


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


@hydra.main(version_base=None, config_path="../conf", config_name="train_ae_dit_snap")
def main(cfg: DictConfig) -> None:
    retired = {"motion_recon_weight", "evaluator_cosine_weight"} & set(cfg)
    if retired:
        raise ValueError(f"Retired DiT option(s): {', '.join(sorted(retired))}")
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    wandb_mode = "disabled" if cfg.entity == "your_wandb_entity" else "online"
    wandb.init(
        name=cfg.name,
        project=cfg.project_name,
        entity=cfg.entity,
        config=namespace_to_dict(cfg),
        mode=wandb_mode,
    )
    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    if cfg.dataset_name == "snap":
        data_root = f"{cfg.dataset_dir}/SnapMoGen/"
        dim_pose = resolve_snap_feature_dim(cfg, require_eval_dim=True)
    else:
        raise NotImplementedError(
            "train_ae_dit_snap.py only supports dataset_name=snap"
        )

    motion_dir = pjoin(data_root, "renamed_feats")
    text_file = pjoin(data_root, "all_caption_clean.json")
    mean = np.load(pjoin(data_root, "meta_data/mean.npy"))
    std = np.load(pjoin(data_root, "meta_data/std.npy"))

    max_motion_length = int(cfg.get("snap_max_motion_length", 300))
    unit_length = int(cfg.get("snap_unit_length", 4))
    train_split_file = resolve_snap_split(
        data_root, cfg.get("snap_train_split", "train_ids_clean.txt")
    )
    val_split_file = resolve_snap_split(
        data_root, cfg.get("snap_val_split", "val_ids.txt")
    )
    print(f"Training split: {train_split_file}")
    print(f"Validation split: {val_split_file}")

    use_m_length = cfg.get("use_m_length", False)

    train_dataset = SnapText2MotionDataset(
        mean,
        std,
        train_split_file,
        motion_dir,
        text_file,
        unit_length,
        max_motion_length,
        max_text_length=cfg.get("snap_text_max_length", 120),
        min_motion_length=cfg.get("snap_min_motion_length", 128),
        return_id=True,
    )
    val_dataset = SnapText2MotionDataset(
        mean,
        std,
        val_split_file,
        motion_dir,
        text_file,
        unit_length,
        max_motion_length,
        max_text_length=cfg.get("snap_text_max_length", 120),
        min_motion_length=cfg.get("snap_min_motion_length", 128),
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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.snap_eval_batch_size,
        drop_last=False,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    run_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name)
    model_dir = pjoin(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    save_args_yaml(cfg, pjoin(run_dir, "config.yaml"))

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
        cfg.checkpoints_dir, cfg.dataset_name, cfg.ae_name, "model", cfg.ae_ckpt_name
    )
    if cfg.get("require_aux_vae", True):
        ae_config_path = pjoin(
            cfg.checkpoints_dir, cfg.dataset_name, cfg.ae_name, "config.yaml"
        )
        if not os.path.isfile(ae_config_path):
            raise FileNotFoundError(f"VAE config not found: {ae_config_path}")
        ae_config = OmegaConf.load(ae_config_path)
        if not ae_config.get("use_aux_model", False):
            raise ValueError(
                f"VAE {cfg.ae_name} was not trained with use_aux_model=true"
            )
        if not ae_config.get("reverse_proj", False):
            raise ValueError(f"VAE {cfg.ae_name} did not use reverse_proj=true")
    ckpt = torch.load(ae_ckpt_path, map_location="cpu")
    model_key = "ae"
    ignored_projection_keys = load_vae_for_inference(ae, ckpt[model_key])
    if ignored_projection_keys:
        print(f"Ignored VAE feature-loss weights: {ignored_projection_keys}")

    snap_text_max_length = int(cfg.get("snap_text_max_length", 120))
    snap_text_encoder_local_files_only = cfg.get(
        "snap_text_encoder_local_files_only", False
    )

    model = build_dit_model(cfg,
        input_dim=ae.output_emb_width,
        cond_mode="text",
        max_length=max_motion_length // ae.down_scale,
        num_frame_per_block=cfg.get("num_frame_per_block", 3),
        text_encoder_type=cfg.get("snap_text_encoder_type", "snap_t5"),
        text_encoder_max_length=snap_text_max_length,
        text_encoder_local_files_only=snap_text_encoder_local_files_only,
    )
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)

    all_params = 0
    pc_transformer = sum(
        param.numel()
        for param in [
            p
            for name, p in model.named_parameters()
            if not name.startswith("clip_model.")
        ]
    )
    all_params += pc_transformer
    print("Total parameters of all models: {:.2f}M".format(all_params / 1000_000))

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

    fid_eval_every = max(1, int(cfg.get("fid_eval_every", 1)))
    fid_eval_window_start = int(cfg.get("fid_eval_window_start", 200))
    fid_eval_window_end = int(cfg.get("fid_eval_window_end", 350))
    fid_eval_window_every = max(1, int(cfg.get("fid_eval_window_every", 25)))
    fid_eval_passes = max(1, int(cfg.get("fid_eval_passes", 1)))
    eval_seed = int(cfg.get("eval_seed", cfg.seed))
    early_stop_start_epoch = int(cfg.get("early_stop_start_epoch", 350))
    early_stop_patience_evals = int(cfg.get("early_stop_patience_evals", 0))
    fid_cond_scale = float(cfg.get("fid_cond_scale", 3.0))
    loss_skip_threshold = float(cfg.get("loss_skip_threshold", 100.0))
    latent_abs_max_threshold = float(cfg.get("latent_abs_max_threshold", 1000.0))
    if loss_skip_threshold <= 0:
        loss_skip_threshold = float("inf")
    if latent_abs_max_threshold <= 0:
        latent_abs_max_threshold = float("inf")
    clip_grad_norm = bool(cfg.get("clip_grad_norm", False))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    if clip_grad_norm and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive when clipping is enabled")

    if np.isfinite(loss_skip_threshold):
        print(f"Skip batch if raw loss > {loss_skip_threshold:.4f}")
    else:
        print("Batch skipping by loss threshold disabled")

    if np.isfinite(latent_abs_max_threshold):
        print(f"Skip batch if latent |z| max > {latent_abs_max_threshold:.4f}")
    else:
        print("Batch skipping by latent max threshold disabled")

    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    ae.eval()
    ae.to(device)
    ae.requires_grad_(False)
    model.to(device)
    ema_model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), betas=(0.9, 0.99), lr=cfg.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.milestones, gamma=cfg.lr_decay
    )

    epoch = 0
    it = 0
    best_fid = float("inf")
    best_epoch = None
    evals_since_best = 0
    if cfg.is_continue:
        checkpoint = torch.load(pjoin(model_dir, "latest.tar"), map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model"], strict=False
        )
        missing_keys2, unexpected_keys2 = ema_model.load_state_dict(
            checkpoint["ema_model"], strict=False
        )
        assert len(unexpected_keys) == 0
        assert len(unexpected_keys2) == 0
        assert all([k.startswith("clip_model.") for k in missing_keys])
        assert all([k.startswith("clip_model.") for k in missing_keys2])
        optimizer.load_state_dict(checkpoint["opt_model"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch, it = checkpoint["ep"] + 1, checkpoint["total_it"]
        if "best_fid" in checkpoint:
            best_fid = float(checkpoint["best_fid"])
            best_epoch = checkpoint.get("best_epoch")
            evals_since_best = int(checkpoint.get("evals_since_best", 0))
        elif cfg.get("resume_best_fid") is not None:
            best_fid = float(cfg.resume_best_fid)
        elif os.path.exists(pjoin(model_dir, "net_best_fid.tar")):
            raise ValueError(
                "The legacy latest.tar has no best_fid. Supply resume_best_fid "
                "from the run log before resuming, to protect net_best_fid.tar."
            )
        print("Load model epoch:%d iterations:%d" % (epoch, it))

    start_time = time.time()
    total_iters = cfg.epoch * len(train_loader)
    print(f"Total Epochs: {cfg.epoch}, Total Iters: {total_iters}")
    print(
        "Iters Per Epoch, Training: %04d, Validation: %03d"
        % (len(train_loader), len(val_loader))
    )

    logs = defaultdict(def_value, OrderedDict())
    rejected_samples_path = pjoin(run_dir, "rejected_samples.csv")

    while epoch < cfg.epoch:
        ae.eval()
        model.train()
        optimizer.zero_grad()

        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < cfg.warm_up_iter:
                update_lr_warm_up(it, cfg.warm_up_iter, optimizer, cfg.lr)

            conds, motion, m_lens, sample_ids = batch_data
            motion = motion.detach().float().to(device)
            m_lens = m_lens.detach().long().to(device)
            with torch.no_grad():
                latent = ae.encode(motion, m_lengths=m_lens if use_m_length else None)
                latent = rearrange(
                    latent, f"{ae.encode_dim} -> b l 1 d", d=cfg.output_dim
                )
            m_lens = torch.div(m_lens, ae.down_scale, rounding_mode="floor")

            conds = conds.to(device).float() if torch.is_tensor(conds) else conds

            keep, sample_max_abs, finite = latent_sample_quality(
                latent, latent_abs_max_threshold
            )
            sample_ids = list(sample_ids)
            rejected_indices = (~keep).nonzero(as_tuple=True)[0].tolist()
            for sample_index in rejected_indices:
                reason = (
                    "latent_non_finite"
                    if not bool(finite[sample_index])
                    else "latent_too_large"
                )
                logs[f"skip_samples_{reason}"] += 1.0
                record_rejected_sample(
                    rejected_samples_path,
                    it,
                    sample_ids[sample_index],
                    reason,
                    float(sample_max_abs[sample_index].item()),
                )
            if rejected_indices:
                logs["filtered_batches"] += 1.0
                print(
                    f"niter {it}: filtered {len(rejected_indices)}/{len(sample_ids)} "
                    f"samples; details in {rejected_samples_path}"
                )

            valid_indices = keep.nonzero(as_tuple=True)[0]
            if valid_indices.numel() == 0:
                logs["skip_batches"] += 1.0
                continue
            latent = latent[valid_indices]
            m_lens = m_lens[valid_indices]
            conds = take_samples(conds, valid_indices)
            sample_ids = take_samples(sample_ids, valid_indices)
            sample_max_abs = sample_max_abs[valid_indices]

            reduction = getattr(model, "padding_loss_reduction", "masked_mean")
            total_mass = (
                float(m_lens.sum().item())
                if reduction == "masked_mean"
                else float(len(sample_ids))
            )
            total_mass = max(total_mass, 1.0)
            accepted_mass = 0.0
            batch_loss = 0.0

            def train_chunk(indices):
                nonlocal accepted_mass, batch_loss
                raw_loss = model.forward_loss(
                    latent[indices], take_samples(conds, indices), m_lens[indices]
                )
                raw_loss_value = float(raw_loss.detach().item())
                if np.isfinite(raw_loss_value) and raw_loss_value <= loss_skip_threshold:
                    mass = (
                        float(m_lens[indices].sum().item())
                        if reduction == "masked_mean"
                        else float(indices.numel())
                    )
                    (raw_loss * (mass / total_mass)).backward()
                    batch_loss += raw_loss_value * mass / total_mass
                    accepted_mass += mass
                    return

                reason = (
                    "loss_non_finite"
                    if not np.isfinite(raw_loss_value)
                    else "loss_too_large"
                )
                del raw_loss
                if indices.numel() > 1:
                    for half in torch.tensor_split(indices, 2):
                        train_chunk(half)
                    return

                sample_index = int(indices[0].item())
                logs[f"skip_samples_{reason}"] += 1.0
                record_rejected_sample(
                    rejected_samples_path,
                    it,
                    sample_ids[sample_index],
                    reason,
                    float(sample_max_abs[sample_index].item()),
                    raw_loss_value,
                )
                print(
                    f"niter {it}: rejected {sample_ids[sample_index]} "
                    f"for {reason} (raw_loss={raw_loss_value})"
                )

            accum = min(max(1, int(cfg.accum)), latent.shape[0])
            batch_indices = torch.arange(latent.shape[0], device=device)
            for indices in torch.tensor_split(batch_indices, accum):
                train_chunk(indices)

            if accepted_mass == 0:
                optimizer.zero_grad()
                logs["skip_batches"] += 1.0
                continue

            if accepted_mass != total_mass:
                scale = total_mass / accepted_mass
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
                batch_loss *= scale

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm if clip_grad_norm else float("inf")
            )
            if not bool(torch.isfinite(grad_norm)):
                optimizer.zero_grad()
                logs["skip_batches_non_finite_grad"] += 1.0
                continue
            logs["grad_norm"] += float(grad_norm.item())
            if clip_grad_norm and grad_norm.item() > max_grad_norm:
                logs["grad_clip_applied"] += 1.0
            logs["loss"] += batch_loss
            logs["lr"] += optimizer.param_groups[0]["lr"]

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            update_ema(model, ema_model, 0.9999)

            if it % cfg.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    wandb.log({f"Train/{tag}": value / cfg.log_every}, step=it)
                    mean_loss[tag] = value / cfg.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(
                    start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i
                )

        stop_for_plateau = False
        if should_evaluate_epoch(
            epoch + 1,
            fid_eval_every,
            fid_eval_window_start,
            fid_eval_window_end,
            fid_eval_window_every,
        ):
            passes = []
            for pass_index in range(fid_eval_passes):
                with seeded_evaluation(eval_seed + pass_index, device):
                    passes.append(
                        evaluation_dit_snap(
                            val_loader,
                            ema_model,
                            ae,
                            eval_wrapper,
                            device,
                            cond_scale=fid_cond_scale,
                            down_scale=ae.down_scale,
                            use_m_length=use_m_length,
                        )
                    )
            pass_metrics = np.asarray(passes, dtype=float)
            fid, div, top1, top2, top3, matching = pass_metrics.mean(axis=0)
            fid_std = (
                float(pass_metrics[:, 0].std(ddof=1)) if fid_eval_passes > 1 else 0.0
            )
            wandb.log(
                {
                    "Val/FID": fid,
                    "Val/FID_std": fid_std,
                    "Val/Diversity": div,
                    "Val/top1": top1,
                    "Val/top2": top2,
                    "Val/top3": top3,
                    "Val/matching": matching,
                },
                step=it,
            )

            if fid < best_fid:
                prev_best = best_fid
                best_fid = float(fid)
                best_epoch = epoch + 1
                evals_since_best = 0
                save(
                    pjoin(model_dir, "net_best_fid.tar"),
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    it,
                    "model",
                    ema=ema_model,
                    extra_state={"best_fid": best_fid, "best_epoch": best_epoch},
                )
                if np.isfinite(prev_best):
                    print(
                        f"EMA FID improved from {prev_best:.4f} to {best_fid:.4f}; "
                        "saved net_best_fid.tar"
                    )
                else:
                    print(
                        f"Initialized EMA best FID at {best_fid:.4f}; "
                        "saved net_best_fid.tar"
                    )
            else:
                evals_since_best += 1
                print(f"EMA FID {fid:.4f}; current best {best_fid:.4f}")

            wandb.log({"Val/FID_best": best_fid}, step=it)
            stop_for_plateau = (
                early_stop_patience_evals > 0
                and epoch + 1 >= early_stop_start_epoch
                and evals_since_best >= early_stop_patience_evals
            )

        save(
            pjoin(model_dir, "latest.tar"),
            epoch,
            model,
            optimizer,
            scheduler,
            it,
            "model",
            ema=ema_model,
            extra_state={
                "best_fid": best_fid,
                "best_epoch": best_epoch,
                "evals_since_best": evals_since_best,
                "train_split_file": train_split_file,
                "val_split_file": val_split_file,
                "ae_checkpoint": ae_ckpt_path,
                "eval_seed": eval_seed,
                "fid_eval_passes": fid_eval_passes,
            },
        )
        if stop_for_plateau:
            print(
                f"Stopping after {evals_since_best} validation evaluations "
                f"without FID improvement; best epoch was {best_epoch}."
            )
            break

        epoch += 1


if __name__ == "__main__":
    main()
