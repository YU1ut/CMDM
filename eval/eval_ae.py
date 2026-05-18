import os
import random
import warnings
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from models.VAE import vae
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from utils.config_utils import apply_config_to_cfg, load_yaml_config
from utils.datasets import Text2MotionDataset, collate_fn
from utils.eval_utils import evaluation_ae
from utils.evaluators import Evaluators

warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="../conf", config_name="eval_ae")
def main(cfg: DictConfig) -> None:
    config_path = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "config.yaml")
    train_config = load_yaml_config(config_path)
    if train_config:
        cfg = apply_config_to_cfg(
            cfg,
            train_config,
            (
                "dataset_name",
                "dataset_dir",
                "output_dim",
            ),
        )
        print(f"Loaded config from {config_path}")
    else:
        print(f"No saved config found at {config_path}. Using Hydra config.")

    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    os.environ["OMP_NUM_THREADS"] = "1"
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
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
    max_motion_length = 192
    mean = np.load(pjoin(data_root, "Mean.npy"))
    std = np.load(pjoin(data_root, "Std.npy"))
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
    model_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "model")

    ae = vae(input_width=dim_pose, output_emb_width=cfg.output_dim, projection_dim=64)
    model_dir = os.path.join(model_dir, "net_best_fid.tar")
    checkpoint = torch.load(model_dir, map_location="cpu")
    ae.load_state_dict(checkpoint["ae"], strict=False)
    device = torch.device(f"cuda:{cfg.gpu}" if cfg.gpu != "cpu" else "cpu")
    eval_wrapper = Evaluators(cfg.dataset_name, device=device)
    #################################################################################
    #                                  Evaluation Loop                              #
    #################################################################################
    out_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "eval")
    os.makedirs(out_dir, exist_ok=True)
    f = open(pjoin(out_dir, "eval.log"), "w")

    ae.eval()
    ae.to(device)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mae = []
    repeat_time = 10
    for i in range(repeat_time):
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe = (
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
            mpjpe,
            writer,
        ) = evaluation_ae(
            model_dir,
            eval_loader,
            ae,
            None,
            i,
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
            save=False,
            draw=False,
        )
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        mae.append(mpjpe)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mae = np.array(mae)

    print(f"final result")
    print(f"final result", file=f, flush=True)

    msg_final = (
        f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        f"\tMAE:{np.mean(mae):.3f}, conf.{np.std(mae) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
    )

    print(msg_final)
    print(msg_final, file=f, flush=True)
    f.close()


if __name__ == "__main__":
    main()
