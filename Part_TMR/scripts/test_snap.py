import logging
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from Part_TMR.datasets.dataset_snap import SnapMoGenTextMotionDataset
from Part_TMR.models.clip_snap import ClipModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="test_config_snap", config_path="../conf")
def main(cfg: DictConfig) -> None:
    require_snapmogen(cfg)
    saved_cfg = OmegaConf.load(pjoin(cfg.checkpoints_dir, ".hydra/config.yaml"))
    require_snapmogen(saved_cfg)
    saved_cfg.checkpoints_dir = str(cfg.checkpoints_dir)
    saved_cfg.dataset.data_root = str(cfg.dataset.data_root)
    saved_cfg.train.batch_size = cfg.train.batch_size
    saved_cfg.train.num_workers = cfg.train.num_workers
    saved_cfg.eval = cfg.eval
    print(OmegaConf.to_yaml(saved_cfg))
    test_dataloader = prepare_test_dataset(saved_cfg)
    model, tokenizer = prepare_test_model(saved_cfg)
    eval(saved_cfg, test_dataloader, model, tokenizer)


def require_snapmogen(cfg) -> None:
    dataset_name = cfg.dataset.dataset_name
    if dataset_name != "SnapMoGen":
        raise ValueError(
            f"Part-TMR SNAP scripts only support SnapMoGen, got {dataset_name!r}"
        )


def prepare_test_dataset(cfg):
    require_snapmogen(cfg)
    mean_path = cfg.dataset.get("mean_path", pjoin(cfg.dataset.data_root, "Mean.npy"))
    std_path = cfg.dataset.get("std_path", pjoin(cfg.dataset.data_root, "Std.npy"))
    train_split_file = cfg.dataset.get(
        "train_split", pjoin(cfg.dataset.data_root, "train.txt")
    )
    default_test_split = pjoin(cfg.dataset.data_root, "test.txt")
    test_split_file = cfg.dataset.get("test_split", default_test_split)
    mean = np.load(mean_path)
    std = np.load(std_path)

    if cfg.eval.eval_train:
        split_file = train_split_file
    else:
        split_file = test_split_file

    test_dataset = SnapMoGenTextMotionDataset(
        cfg,
        mean,
        std,
        split_file,
        eval_mode=True,
        fps=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=int(cfg.train.get("num_workers", 16)),
    )
    return test_dataloader


def prepare_test_model(cfg):
    require_snapmogen(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_encoder_alias = cfg.model.text_encoder
    motion_embedding_dims: int = 512
    text_embedding_dims: int = 768
    projection_dims: int = 256

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_alias)

    model = ClipModel(
        text_encoder_alias=text_encoder_alias,
        text_encoder_trainable=False,
        motion_embedding_dims=motion_embedding_dims,
        text_embedding_dims=text_embedding_dims,
        projection_dims=projection_dims,
        mode="snapmogen",
    )

    print(
        "    Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    )

    if cfg.eval.use_best_model:
        model_path = pjoin(cfg.checkpoints_dir, "best_model.pt")
    else:
        model_path = pjoin(cfg.checkpoints_dir, "last_model.pt")

    print(model_path)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    model.load_state_dict(state_dict)

    model.to(device)

    return model, tokenizer


def prepare_text_input(texts, tokenizer, device):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        device
    )


def eval(cfg, test_dataloader, model, tokenizer=None, verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_pair = dict()

    all_imgs_feat = []
    all_captions_feat = []

    all_img_idxs = []
    all_captions = []

    step = 0
    with torch.no_grad():
        model.eval()
        test_pbar = tqdm(test_dataloader, leave=False)
        for batch in test_pbar:
            step += 1
            (
                Root,
                R_Leg,
                L_Leg,
                Backbone,
                R_Arm,
                L_Arm,
                texts,
                m_length,
                img_indexs,
            ) = batch

            Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = (
                Root.to(device),
                R_Leg.to(device),
                L_Leg.to(device),
                Backbone.to(device),
                R_Arm.to(device),
                L_Arm.to(device),
            )

            motions = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

            texts_token = prepare_text_input(texts, tokenizer, device)

            motion_features = model.encode_motion(motions)
            text_features = model.encode_text(texts_token)

            # normalized features
            motion_features = motion_features / motion_features.norm(
                dim=1, keepdim=True
            )
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            for i in range(motion_features.size(0)):
                all_imgs_feat.append(motion_features[i].cpu().numpy())
                all_captions_feat.append(text_features[i].cpu().numpy())

                all_captions.append(texts[i])
                all_img_idxs.append(img_indexs[i].item())

    all_captions = np.array(all_captions)
    for img_idx, caption in zip(all_img_idxs, all_captions):
        dataset_pair[img_idx] = np.where(all_captions == caption)[0]

    all_imgs_feat = np.vstack(all_imgs_feat)
    all_captions_feat = np.vstack(all_captions_feat)

    # match test queries to target motions, get nearest neighbors
    sims_t2m = 100 * all_captions_feat.dot(all_imgs_feat.T)

    t2m_r1 = 0
    # Text->Motion
    ranks = np.zeros(sims_t2m.shape[0])
    for index, score in enumerate(tqdm(sims_t2m)):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    for k in [1, 2, 3, 5, 10]:
        # Compute metrics
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        if k == 1:
            t2m_r1 = r
        if verbose:
            log.info(f"t2m_recall_top{k}_correct_composition: {r:.2f}")
    if verbose:
        log.info(f"t2m_recall_median_correct_composition: {np.median(ranks) + 1:.2f}")

    # match motions queries to target texts, get nearest neighbors
    sims_m2t = sims_t2m.T

    m2t_r1 = 0
    # Motion->Text
    ranks = np.zeros(sims_m2t.shape[0])
    for index, score in enumerate(tqdm(sims_m2t)):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    for k in [1, 2, 3, 5, 10]:
        # Compute metrics
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        if k == 1:
            m2t_r1 = r
        if verbose:
            log.info(f"m2t_recall_top{k}_correct_composition: {r:.2f}")
    if verbose:
        log.info(f"m2t_recall_median_correct_composition: {np.median(ranks) + 1:.2f}")

    return t2m_r1, m2t_r1


if __name__ == "__main__":
    main()
