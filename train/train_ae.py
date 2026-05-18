import os
import random
import time
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

import hydra
import numpy as np
import torch
import torch.optim as optim
import wandb
from einops import rearrange
from models.VAE import vae
from omegaconf import DictConfig
from Part_TMR.models.clip import ClipModel
from torch.utils.data import DataLoader
from utils.config_utils import save_args_yaml
from utils.datasets import AEDataset, Text2MotionDataset, collate_fn
from utils.eval_utils import evaluation_ae
from utils.evaluators import Evaluators
from utils.parts_utils import whole2parts
from utils.train_utils import def_value, print_current_loss, save, update_lr_warm_up
from utils.wandb_utils import namespace_to_dict


@hydra.main(version_base=None, config_path="../conf", config_name="train_ae")
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
    if cfg.dataset_name == "t2m":
        data_root = f"{cfg.dataset_dir}/HumanML3D/"
        dim_pose = 263
    else:
        raise NotImplementedError
    motion_dir = pjoin(data_root, "new_joint_vecs")
    text_dir = pjoin(data_root, "texts")
    mean = np.load(pjoin(data_root, "Mean.npy"))
    std = np.load(pjoin(data_root, "Std.npy"))

    train_split_file = pjoin(data_root, "train.txt")
    val_split_file = pjoin(data_root, "val.txt")

    train_dataset = AEDataset(mean, std, motion_dir, cfg.window_size, train_split_file)
    val_dataset = AEDataset(mean, std, motion_dir, cfg.window_size, val_split_file)

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
        drop_last=True,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    eval_mean = np.load(f"utils/eval_mean_std/{cfg.dataset_name}/eval_mean.npy")
    eval_std = np.load(f"utils/eval_mean_std/{cfg.dataset_name}/eval_std.npy")
    split_file = pjoin(data_root, "val.txt")
    max_motion_length = 192
    eval_dataset = Text2MotionDataset(
        mean,
        std,
        split_file,
        cfg.dataset_name,
        pjoin(data_root, "new_joint_vecs"),
        text_dir,
        4,
        max_motion_length,
        20,
        evaluation=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        shuffle=True,
    )
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    run_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name)
    model_dir = pjoin(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    save_args_yaml(cfg, pjoin(run_dir, "config.yaml"))

    kl_base_weight = cfg.kl_weight
    vf_base_weight = cfg.vf_weight
    ae = vae(
        input_width=dim_pose,
        output_emb_width=cfg.output_dim,
        vf_weight=cfg.vf_weight,
        adaptive_vf=cfg.use_adaptive_vf,
        root_loss_weight=cfg.root_loss_weight,
        kl_weight=kl_base_weight,
        reverse_proj=cfg.reverse_proj,
    )

    pc_vae = sum(param.numel() for param in ae.parameters())
    print("Total parameters of all models: {}M".format(pc_vae / 1000_000))

    device = torch.device(f"cuda:{cfg.gpu}" if cfg.gpu != "cpu" else "cpu")

    tmr_model = ClipModel(
        motion_embedding_dims=512,
        text_embedding_dims=768,
        projection_dims=256,
        mode="t2m",
    )

    print("Loading Part-TMR model from", cfg.aux_model_path)
    state_dict = torch.load(cfg.aux_model_path, map_location="cpu")

    tmr_model.load_state_dict(state_dict)

    for param in tmr_model.parameters():
        param.requires_grad = False

    tmr_model.to(device)
    tmr_model.eval()

    eval_wrapper = Evaluators(cfg.dataset_name, device=device)
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
    if cfg.recons_loss == "l1_smooth":
        criterion = torch.nn.SmoothL1Loss()
    else:
        criterion = torch.nn.MSELoss()

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
        optimizer.load_state_dict(checkpoint[f"opt_ae"])
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

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe = (
        1000,
        0,
        0,
        0,
        0,
        100,
        100,
    )

    while epoch < cfg.epoch:

        ae.train()
        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < cfg.warm_up_iter:
                current_lr = update_lr_warm_up(it, cfg.warm_up_iter, optimizer, cfg.lr)

            motions = batch_data.detach().to(device).float()

            motions_parts = whole2parts(motions, mode="t2m")
            motions_parts = [motion.to(device).float() for motion in motions_parts]

            with torch.no_grad():
                aux_feature = tmr_model.encode_motion(motions_parts, calc_mean=False)
            aux_feature = aux_feature.detach()

            motions = rearrange(motions, "b l d -> b l 1 d")

            # a naive way of writing gradient accumulation
            accum = cfg.accum
            accum_bs = cfg.batch_size // accum
            if cfg.kl_warmup_iter > 0:
                kl_scale = min(1.0, it / cfg.kl_warmup_iter)
            else:
                kl_scale = 1.0
            current_kl_weight = kl_base_weight * kl_scale
            vf_decay_start = max(0, cfg.warm_up_iter)
            vf_decay_iters = max(1, total_iters - vf_decay_start)
            vf_progress = min(
                1.0, max(0, it - vf_decay_start) / vf_decay_iters
            )
            current_vf_weight = vf_base_weight * (1.0 - 0.9 * vf_progress)
            ae.loss_with_vf.vf_weight = current_vf_weight

            for kk in range(accum):
                pred_motion, loss, loss_items = ae(
                    motions[kk * accum_bs : (kk + 1) * accum_bs],
                    aux_feature=aux_feature[:, kk * accum_bs : (kk + 1) * accum_bs],
                    need_loss=True,
                    kl_weight_override=current_kl_weight,
                    return_loss_components=True,
                )
                logs["lr"] += (optimizer.param_groups[0]["lr"]) / accum
                logs["loss"] += loss.item() / accum
                for tag, value in loss_items.items():
                    logs[tag] += value.item() / accum
                logs["vf_weight"] += current_vf_weight / accum
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

        save(pjoin(model_dir, "latest.tar"), epoch, ae, optimizer, scheduler, it, "ae")
        #################################################################################
        #                                      Eval Loop                                #
        #################################################################################
        print("Validation time:")
        ae.eval()
        val_loss_rec = []
        val_loss = []
        with torch.no_grad():
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

        (
            best_fid,
            best_div,
            best_top1,
            best_top2,
            best_top3,
            best_matching,
            mpjpe,
            writer,
        ) = evaluation_ae(
            model_dir,
            eval_loader,
            ae,
            wandb,
            epoch,
            log_step=it,
            device=device,
            best_fid=best_fid,
            best_div=best_div,
            best_top1=best_top1,
            best_top2=best_top2,
            best_top3=best_top3,
            eval_mean=eval_mean,
            eval_std=eval_std,
            best_matching=best_matching,
            eval_wrapper=eval_wrapper,
        )
        print(f"best fid {best_fid}")

        epoch += 1


if __name__ == "__main__":
    main()
