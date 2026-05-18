import os
import random
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from models.Causal_DiT import dit
from models.VAE import vae
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from utils.config_utils import apply_config_to_cfg, load_yaml_config
from utils.datasets import Text2MotionDataset, collate_fn
from utils.eval_utils import evaluation_dit
from utils.evaluators import Evaluators


@hydra.main(version_base=None, config_path="../conf", config_name="eval_ae_dit")
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
                "dataset_dir",
                "ae_name",
                "output_dim",
                "max_motion_length",
                "unit_length",
            ),
        )
        print(f"Loaded DiT config from {dit_config_path}")
    else:
        print(f"No saved DiT config found at {dit_config_path}. Using Hydra config.")

    ae_config_path = pjoin(
        cfg.checkpoints_dir, cfg.dataset_name, cfg.ae_name, "config.yaml"
    )
    ae_config = load_yaml_config(ae_config_path)
    if ae_config:
        cfg = apply_config_to_cfg(
            cfg,
            ae_config,
            ("output_dim",),
        )
        print(f"Loaded AE config from {ae_config_path}")
    else:
        print(f"No saved AE config found at {ae_config_path}. Using Hydra config.")

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
    #                                    Eval Data                                  #
    #################################################################################
    if cfg.dataset_name == "t2m":
        data_root = f"{cfg.dataset_dir}/HumanML3D/"
        dim_pose = 263
    else:
        raise NotImplementedError
    motion_dir = pjoin(data_root, "new_joint_vecs")
    text_dir = pjoin(data_root, "texts")
    max_motion_length = cfg.max_motion_length

    mean = np.load(pjoin(data_root, "Mean.npy"))
    std = np.load(pjoin(data_root, "Std.npy"))
    if cfg.feature_dim == 67:
        eval_mean = np.load(f"utils/eval_mean_std/{cfg.dataset_name}/eval_mean_67.npy")
        eval_std = np.load(f"utils/eval_mean_std/{cfg.dataset_name}/eval_std_67.npy")
    else:
        eval_mean = np.load(f"utils/eval_mean_std/{cfg.dataset_name}/eval_mean.npy")
        eval_std = np.load(f"utils/eval_mean_std/{cfg.dataset_name}/eval_std.npy")
    split_file = pjoin(data_root, "test.txt")
    eval_dataset = Text2MotionDataset(
        mean,
        std,
        split_file,
        cfg.dataset_name,
        motion_dir,
        text_dir,
        cfg.unit_length,
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
    model_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "model")

    ae = vae(input_width=dim_pose, output_emb_width=cfg.output_dim)
    ckpt = torch.load(
        pjoin(
            cfg.checkpoints_dir, cfg.dataset_name, cfg.ae_name, "model", "net_best_fid.tar"
        ),
        map_location="cpu",
    )
    model_key = "ae"
    ae.load_state_dict(ckpt[model_key], strict=False)

    ema_model = dit(input_dim=ae.output_emb_width, cond_mode="text")
    model_dir = os.path.join(model_dir, "latest.tar")
    checkpoint = torch.load(model_dir, map_location="cpu")
    missing_keys2, unexpected_keys2 = ema_model.load_state_dict(
        checkpoint["ema_model"], strict=False
    )
    assert len(unexpected_keys2) == 0
    assert all([k.startswith("clip_model.") for k in missing_keys2])

    device = torch.device(f"cuda:{cfg.gpu}" if cfg.gpu != "cpu" else "cpu")
    eval_wrapper = Evaluators(
        cfg.dataset_name, device=device, feature_dim=cfg.feature_dim
    )
    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    out_dir = pjoin(
        cfg.checkpoints_dir, cfg.dataset_name, cfg.name, f"eval_{cfg.feature_dim}"
    )
    os.makedirs(out_dir, exist_ok=True)
    f = open(pjoin(out_dir, f"eval_{cfg.cfg}.log"), "a")

    ema_model.eval()
    ema_model.to(device)

    ae.eval()
    ae.to(device)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mm = []
    clip_scores = []

    repeat_time = 10
    for i in range(repeat_time):
        with torch.no_grad():
            (
                best_fid,
                best_div,
                best_top1,
                best_top2,
                best_top3,
                best_matching,
                best_mm,
                clip_score,
            ) = (1000, 0, 0, 0, 0, 100, 0, -1)
            (
                best_fid,
                best_div,
                best_top1,
                best_top2,
                best_top3,
                best_matching,
                best_mm,
                clip_score,
            ) = evaluation_dit(
                model_dir,
                eval_loader,
                ema_model,
                ae,
                i,
                best_fid=best_fid,
                clip_score_old=clip_score,
                best_div=best_div,
                best_top1=best_top1,
                best_top2=best_top2,
                best_top3=best_top3,
                best_matching=best_matching,
                eval_wrapper=eval_wrapper,
                device=device,
                eval_mean=eval_mean,
                eval_std=eval_std,
                after_mean=None,
                after_std=None,
                cond_scale=cfg.cfg,
                cal_mm=cfg.cal_mm,
            )

        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        mm.append(best_mm)
        clip_scores.append(clip_score)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mm = np.array(mm)
    clip_scores = np.array(clip_scores)

    print(f"final result:")
    print(f"final result:", file=f, flush=True)

    msg_final = (
        f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n"
    )

    if cfg.feature_dim == 67:
        msg_final += f"\tCLIP-Score:{np.mean(clip_scores):.3f}, conf.{np.std(clip_scores) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"

    print(msg_final)
    print(msg_final, file=f, flush=True)
    f.close()


if __name__ == "__main__":
    main()
