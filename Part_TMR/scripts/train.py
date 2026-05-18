import itertools
import logging
import os
import random
import sys
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from Part_TMR.datasets import TextMotionDataset
from Part_TMR.models.clip import ClipModel
from Part_TMR.scripts.test import eval, prepare_test_dataset
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    # set_seed(cfg.train.seed)
    train_dataloader, test_dataloader = prepare_dataset(cfg)
    eval_dataloader = prepare_test_dataset(cfg)
    model, optimizer, scheduler, tokenizer = prepare_model(cfg, train_dataloader)
    train(
        cfg,
        train_dataloader,
        test_dataloader,
        eval_dataloader,
        model,
        tokenizer,
        optimizer,
        scheduler,
    )


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def prepare_dataset(cfg):
    mean = np.load(pjoin(cfg.dataset.data_root, "Mean.npy"))
    std = np.load(pjoin(cfg.dataset.data_root, "Std.npy"))

    train_split_file = pjoin(cfg.dataset.data_root, "train.txt")
    train_dataset = TextMotionDataset(
        cfg,
        mean,
        std,
        train_split_file,
        fps=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=16,
    )

    val_split_file = pjoin(cfg.dataset.data_root, "val.txt")
    val_dataset = TextMotionDataset(
        cfg,
        mean,
        std,
        val_split_file,
        fps=True,
    )
    test_dataloader = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=16
    )

    return train_dataloader, test_dataloader


def prepare_model(cfg, train_dataloader):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_encoder_alias = cfg.model.text_encoder
    text_encoder_trainable: bool = cfg.train.train_text_encoder
    motion_embedding_dims: int = 512
    text_embedding_dims: int = 768
    projection_dims: int = 256

    tokenizer = AutoTokenizer.from_pretrained(
        text_encoder_alias, TOKENIZERS_PARALLELISM=True
    )

    model = ClipModel(
        text_encoder_alias,
        text_encoder_trainable,
        motion_embedding_dims,
        text_embedding_dims,
        projection_dims,
        dropout=0.5 if cfg.dataset.dataset_name == "HumanML3D" else 0.0,
        mode="t2m" if cfg.dataset.dataset_name == "HumanML3D" else "kit",
    )

    model.to(device)
    parameters = [
        {
            "params": model.motion_encoder.parameters(),
            "lr": cfg.train.optimizer.motion_lr * cfg.dataset.motion_lr_factor,
        },
        {
            "params": model.text_encoder.parameters(),
            "lr": cfg.train.optimizer.text_lr * cfg.dataset.text_lr_factor,
        },
        {
            "params": itertools.chain(
                model.motion_projection.parameters(),
                model.text_projection.parameters(),
            ),
            "lr": cfg.train.optimizer.head_lr * cfg.dataset.head_lr_factor,
        },
    ]
    optimizer = optim.Adam(parameters)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_dataloader) * cfg.train.epoch * 2
    )

    return model, optimizer, scheduler, tokenizer


def train(
    cfg,
    train_dataloader,
    test_dataloader,
    eval_dataloader,
    model,
    tokenizer,
    optimizer,
    scheduler,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    best_te_loss = 1e5
    best_t2m_r1 = 0
    best_m2t_r1 = 0
    best_h_r1 = 0
    best_ep = -1
    for epoch in range(cfg.train.epoch):
        print(
            f"running epoch {epoch}, best test loss {best_te_loss} best_t2m_r1 {best_t2m_r1} best_m2t_r1 {best_m2t_r1} after epoch {best_ep}"
        )
        step = 0
        tr_loss = 0
        model.train()
        pbar = tqdm(train_dataloader, leave=False)
        for batch in pbar:
            step += 1
            optimizer.zero_grad()

            Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, texts, _, _ = batch

            Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = (
                Root.to(device),
                R_Leg.to(device),
                L_Leg.to(device),
                Backbone.to(device),
                R_Arm.to(device),
                L_Arm.to(device),
            )

            motions = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

            texts = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)

            total_loss = model(motions, texts, return_loss=True)
            total_loss.backward()
            tr_loss += total_loss.item()
            optimizer.step()
            scheduler.step()
            pbar.set_description(f"train batchCE: {total_loss.item()}", refresh=True)
        tr_loss /= step

        step = 0
        te_loss = 0
        with torch.no_grad():
            model.eval()
            test_pbar = tqdm(test_dataloader, leave=False)
            for batch in test_pbar:
                step += 1
                Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, texts, _, _ = batch

                Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = (
                    Root.to(device),
                    R_Leg.to(device),
                    L_Leg.to(device),
                    Backbone.to(device),
                    R_Arm.to(device),
                    L_Arm.to(device),
                )
                motions = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

                texts = tokenizer(
                    texts, padding=True, truncation=True, return_tensors="pt"
                ).to(device)

                total_loss = model(motions, texts, return_loss=True)

                te_loss += total_loss.item()
                test_pbar.set_description(
                    f"test batchCE: {total_loss.item()}", refresh=True
                )
            te_loss /= step

        if te_loss < best_te_loss:
            best_te_loss = te_loss

        torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, "last_model.pt"))

        t2m_r1, m2t_r1 = eval(
            cfg, eval_dataloader, model, tokenizer=tokenizer, verbose=False
        )

        log.info(
            f"epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}, t2m_r1 {t2m_r1}, m2t_r1 {m2t_r1} "
        )

        best_t2m_r1 = max(best_t2m_r1, t2m_r1)
        best_m2t_r1 = max(best_m2t_r1, m2t_r1)

        if best_m2t_r1 == m2t_r1:
            best_ep = epoch
            torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, "best_model.pt"))


if __name__ == "__main__":
    main()
