import copy
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
from models.Causal_DiT import dit
from models.VAE import vae
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from utils.config_utils import save_args_yaml
from utils.datasets import Text2MotionDataset
from utils.train_utils import (
    def_value,
    print_current_loss,
    save,
    update_ema,
    update_lr_warm_up,
)
from utils.wandb_utils import namespace_to_dict


@hydra.main(version_base=None, config_path="../conf", config_name="train_ae_dit")
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

    train_dataset = Text2MotionDataset(
        mean,
        std,
        train_split_file,
        cfg.dataset_name,
        motion_dir,
        text_dir,
        cfg.unit_length,
        cfg.max_motion_length,
        20,
        evaluation=False,
    )
    val_dataset = Text2MotionDataset(
        mean,
        std,
        val_split_file,
        cfg.dataset_name,
        motion_dir,
        text_dir,
        cfg.unit_length,
        cfg.max_motion_length,
        20,
        evaluation=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
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

    ae = vae(input_width=dim_pose, output_emb_width=cfg.output_dim)
    ckpt = torch.load(
        pjoin(
            cfg.checkpoints_dir, cfg.dataset_name, cfg.ae_name, "model", "net_best_fid.tar"
        ),
        map_location="cpu",
    )
    model_key = "ae"
    ae.load_state_dict(ckpt[model_key], strict=False)

    model = dit(input_dim=ae.output_emb_width, cond_mode="text")
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
    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    ae.eval()
    ae.to(device)
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
        print("Load model epoch:%d iterations:%d" % (epoch, it))

    start_time = time.time()
    total_iters = cfg.epoch * len(train_loader)
    print(f"Total Epochs: {cfg.epoch}, Total Iters: {total_iters}")
    print(
        "Iters Per Epoch, Training: %04d, Validation: %03d"
        % (len(train_loader), len(val_loader))
    )

    logs = defaultdict(def_value, OrderedDict())
    worst_loss = 100

    while epoch < cfg.epoch:
        ae.eval()
        model.train()
        optimizer.zero_grad()

        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < cfg.warm_up_iter:
                update_lr_warm_up(it, cfg.warm_up_iter, optimizer, cfg.lr)

            conds, motion, m_lens = batch_data
            motion = motion.detach().float().to(device)
            m_lens = m_lens.detach().long().to(device)

            with torch.no_grad():
                latent = ae.encode(motion)
                latent = rearrange(
                    latent, f"{ae.encode_dim} -> b l 1 d", d=cfg.output_dim
                )
            m_lens = m_lens // 4

            conds = conds.to(device).float() if torch.is_tensor(conds) else conds

            # a naive way of writing gradient accumulation
            accum = cfg.accum
            accum_bs = cfg.batch_size // accum

            skip_batch = False
            skipped_loss = 0.0
            batch_loss = 0.0
            batch_lr = 0.0
            for kk in range(accum):
                loss = model.forward_loss(
                    latent[kk * accum_bs : (kk + 1) * accum_bs],
                    conds[kk * accum_bs : (kk + 1) * accum_bs],
                    m_lens[kk * accum_bs : (kk + 1) * accum_bs],
                )
                raw_loss = loss.detach().item()
                if raw_loss > 10.0:
                    skip_batch = True
                    skipped_loss = raw_loss
                    break

                loss = loss / accum
                loss.backward()
                batch_loss += loss.item()
                batch_lr += (optimizer.param_groups[0]["lr"]) / accum

            if skip_batch:
                optimizer.zero_grad()
                logs["skipped_batches"] += 1
                print(
                    f"[Skip Batch] iter={it}, epoch={epoch}, "
                    f"inner_iter={i}, loss={skipped_loss:.4f} (> 10.0)"
                )
                continue

            logs["loss"] += batch_loss
            logs["lr"] += batch_lr

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

        save(
            pjoin(model_dir, "latest.tar"),
            epoch,
            model,
            optimizer,
            scheduler,
            it,
            "model",
            ema=ema_model,
        )
        #################################################################################
        #                                      Eval Loop                                #
        #################################################################################
        print("Validation time:")
        ae.eval()
        model.eval()
        val_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                conds, motion, m_lens = batch_data
                motion = motion.detach().float().to(device)
                m_lens = m_lens.detach().long().to(device)

                with torch.no_grad():
                    latent = ae.encode(motion)
                    latent = rearrange(
                        latent, f"{ae.encode_dim} -> b l 1 d", d=cfg.output_dim
                    )
                m_lens = m_lens // 4

                conds = conds.to(device).float() if torch.is_tensor(conds) else conds

                loss = model.forward_loss(latent, conds, m_lens)
                val_loss.append(loss.item())

        print(f"Validation loss:{np.mean(val_loss):.3f}")
        wandb.log({f"Val/loss": np.mean(val_loss)}, step=it)
        if np.mean(val_loss) < worst_loss:
            print(f"Improved loss from {worst_loss:.02f} to {np.mean(val_loss)}!!!")
            worst_loss = np.mean(val_loss)
        epoch += 1


if __name__ == "__main__":
    main()
