import csv
import os
import random
from os.path import join as pjoin
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import rearrange
from models.Causal_DiT import dit
from models.VAE import vae
from omegaconf import DictConfig
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion, plot_3d_motion_long


def load_prompts_and_lengths(cfg: DictConfig) -> tuple[list[str], list[int]]:
    prompt_csv = cfg.get("prompt_csv")
    if prompt_csv:
        csv_path = Path(prompt_csv).expanduser()
        if not csv_path.exists():
            raise FileNotFoundError(f"Prompt CSV not found: {csv_path}")
        prompts: list[str] = []
        lengths: list[int] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row_idx, row in enumerate(reader, start=2):
                prompt = (row.get("prompt") or "").strip()
                length_str = (row.get("length") or "").strip()
                if not prompt:
                    raise ValueError(
                        f"Missing prompt at {csv_path}:{row_idx}. "
                        "Expected columns: prompt,length"
                    )
                if not length_str:
                    raise ValueError(
                        f"Missing length at {csv_path}:{row_idx}. "
                        "Expected columns: prompt,length"
                    )

                try:
                    length = int(length_str)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid length at {csv_path}:{row_idx}: {length_str}"
                    ) from exc

                if length <= 0:
                    raise ValueError(
                        f"Length must be > 0 at {csv_path}:{row_idx}, got {length}"
                    )
                prompts.append(prompt)
                lengths.append(length)

        if not prompts:
            raise ValueError(f"No prompts found in CSV: {csv_path}")

        print(f"Loaded {len(prompts)} prompts from: {csv_path}")
        return prompts, lengths

    return [cfg.input_text] * cfg.num_samples, [cfg.length] * cfg.num_samples


@hydra.main(version_base=None, config_path="../conf", config_name="demo")
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
    #                                    Eval Data                                  #
    #################################################################################
    if cfg.dataset_name == "t2m":
        data_root = f"{cfg.dataset_dir}/HumanML3D/"
        dim_pose = 263
    else:
        raise NotImplementedError

    mean = np.load(pjoin(data_root, "Mean.npy"))
    std = np.load(pjoin(data_root, "Std.npy"))

    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "model")

    ae = vae(input_width=dim_pose, output_emb_width=cfg.output_dim)
    ckpt = torch.load(
        pjoin(
            cfg.checkpoints_dir,
            cfg.dataset_name,
            cfg.ae_name,
            "model",
            "net_best_fid.tar",
        ),
        map_location="cpu",
    )
    model_key = "ae"
    ae.load_state_dict(ckpt[model_key], strict=False)

    ema_model = dit(
        input_dim=ae.output_emb_width,
        cond_mode="text",
        chunk_size=cfg.chunk_size,
        n_tokens=cfg.n_tokens,
        max_length=cfg.n_tokens,
    )
    model_dir = os.path.join(model_dir, "latest.tar")
    checkpoint = torch.load(model_dir, map_location="cpu")
    missing_keys2, unexpected_keys2 = ema_model.load_state_dict(
        checkpoint["ema_model"], strict=False
    )
    assert len(unexpected_keys2) == 0
    assert all([k.startswith("clip_model.") for k in missing_keys2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    out_dir = pjoin(cfg.out_dir_root, cfg.exp)
    os.makedirs(out_dir, exist_ok=True)

    ae.eval()
    ae.to(device)
    ema_model.eval()
    ema_model.to(device)
    ema_model.num_past_latents = int(
        cfg.get("num_past_latents", ema_model.num_past_latents)
    )

    clip_text, frame_lengths = load_prompts_and_lengths(cfg)
    cond_scale = cfg.cfg
    use_autoregressive = bool(cfg.get("autoregressive", False)) and bool(
        cfg.get("prompt_csv")
    )
    if not use_autoregressive and cfg.get("prompt_csv"):
        expanded_texts: list[str] = []
        expanded_lengths: list[int] = []
        for text, length in zip(clip_text, frame_lengths):
            expanded_texts.extend([text] * cfg.num_samples)
            expanded_lengths.extend([length] * cfg.num_samples)
        clip_text = expanded_texts
        frame_lengths = expanded_lengths
    m_length = torch.tensor([max(1, length // 4) for length in frame_lengths]).to(
        device
    )
    with torch.no_grad():
        if any(length % 4 != 0 for length in frame_lengths):
            print("Warning: Some lengths are not divisible by 4 and will be floored.")
        if use_autoregressive:
            pred_latents = ema_model.generate_long(
                clip_text,
                frame_lengths,
                cond_scale=cond_scale,
                num_samples=cfg.num_samples,
            )
            render_lengths = [max(1, length // 4) * 4 for length in frame_lengths]
        else:
            pred_latents = ema_model.generate(clip_text, m_length, cond_scale)
            render_texts = clip_text
        pred_latents = rearrange(pred_latents, f"b l 1 d -> {ae.encode_dim}")
        samples = ae.decode(pred_latents)
        samples = rearrange(samples, "b l 1 d -> b l d")
    samples = (samples.detach().cpu().numpy() * std) + mean
    samples = recover_from_ric(torch.from_numpy(samples), joints_num=22).numpy()
    print("samples.shape", samples.shape)

    if use_autoregressive:
        for i in range(cfg.num_samples):
            sample = samples[i]
            output_path = os.path.join(out_dir, f"long_sample_rep{i:02d}.mp4")
            np.save(os.path.join(out_dir, f"long_sample_rep{i:02d}.npy"), sample)
            plot_3d_motion_long(
                output_path, sample, clip_text, render_lengths, fps=20, radius=4
            )
            print(f"Saved sample {i} to: {output_path}")
    else:
        for i in range(len(render_texts)):
            sample = samples[i]
            pair_idx = i // cfg.num_samples
            rep_idx = i % cfg.num_samples
            sample_name = f"sample_{pair_idx:02d}_rep{rep_idx:02d}"
            output_path = os.path.join(out_dir, f"{sample_name}.mp4")
            np.save(os.path.join(out_dir, f"{sample_name}.npy"), sample)
            plot_3d_motion(output_path, sample, render_texts[i], fps=20, radius=4)
            print(f"Saved sample {i} to: {output_path}")

    print(f"All samples saved to: {out_dir}")


if __name__ == "__main__":
    main()
