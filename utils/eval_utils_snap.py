"""Snap AE and DiT evaluation, extracted from CVPR-sample-code."""

import os

import torch
from einops import rearrange
from tqdm import tqdm

from utils.eval_utils import (
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    cosine_similarity_matrix,
)


@torch.no_grad()
def evaluation_dit_snap(
    val_loader,
    ema_model,
    ae,
    eval_wrapper,
    device,
    cond_scale,
    down_scale=None,
    use_m_length=False,
    sampler="euler",
):
    ema_model.eval()
    ae.eval()

    if down_scale is None:
        down_scale = int(getattr(ae, "down_scale", 4))
    if down_scale <= 0:
        raise ValueError(f"down_scale must be > 0, got {down_scale}")

    motion_annotation_list = []
    motion_pred_list = []

    r_precision_real = 0
    r_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    nb_sample = 0

    for batch in tqdm(val_loader):
        clip_text, motions, m_length = batch
        motions = motions.to(device).float()
        m_length = m_length.to(device).long()
        bs = motions.shape[0]

        et, _ = eval_wrapper.encode_text(clip_text, sample_mean=True)
        if len(motions.shape) == 3:
            motions_eval = motions
            motions_input = rearrange(motions, "b l d -> b l 1 d")
        elif len(motions.shape) == 4:
            motions_eval = rearrange(motions, "b l 1 d -> b l d")
            motions_input = motions
        else:
            raise ValueError(f"Invalid motions shape: {motions.shape}")
        m_length_input = m_length if use_m_length else None
        fid_em, em, _ = eval_wrapper.encode_motion(
            motions_eval[..., :148], m_length, sample_mean=True
        )

        pred_latents = ema_model.generate(
            clip_text,
            torch.div(m_length, down_scale, rounding_mode="floor"),
            cond_scale=cond_scale,
            sampler=sampler,
        )
        pred_latents = rearrange(pred_latents, "b l 1 d -> b l d")
        pred_latents = rearrange(pred_latents, f"b l d -> {ae.encode_dim}")
        pred_pose_eval = ae.decode(pred_latents, m_lengths=m_length_input)

        pred_motions = rearrange(pred_pose_eval, "b l 1 d -> b l d")
        fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(
            pred_motions[..., :148], m_length, sample_mean=True
        )

        motion_annotation_list.append(fid_em)
        motion_pred_list.append(fid_em_pred)

        et_np = et.cpu().numpy()
        em_np = em.cpu().numpy()
        em_pred_np = em_pred.cpu().numpy()

        temp_r = calculate_R_precision(
            et_np, em_np, top_k=3, sum_all=True, is_cosine_sim=True
        )
        temp_match = (cosine_similarity_matrix(et_np, em_np)).trace()
        r_precision_real += temp_r
        matching_score_real += temp_match

        temp_r = calculate_R_precision(
            et_np, em_pred_np, top_k=3, sum_all=True, is_cosine_sim=True
        )
        temp_match = (cosine_similarity_matrix(et_np, em_pred_np)).trace()
        r_precision += temp_r
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    r_precision_real = r_precision_real / nb_sample
    r_precision = r_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    msg = (
        "--> \t FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, "
        "R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), "
        "matching_real. %.4f, matching_pred. %.4f"
        % (
            fid,
            diversity_real,
            diversity,
            r_precision_real[0],
            r_precision_real[1],
            r_precision_real[2],
            r_precision[0],
            r_precision[1],
            r_precision[2],
            matching_score_real,
            matching_score_pred,
        )
    )
    print(msg)

    return (
        fid,
        diversity,
        r_precision[0],
        r_precision[1],
        r_precision[2],
        matching_score_pred,
    )


@torch.no_grad()
def evaluation_ae_snap(
    out_dir,
    val_loader,
    net,
    writer,
    ep,
    eval_wrapper,
    device,
    best_fid=1000,
    best_div=0,
    best_top1=0,
    best_top2=0,
    best_top3=0,
    best_matching=100,
    save=True,
    draw=True,
    use_m_length=False,
    best_mpjpe=1000,
    skeleton=None,
    forward_kinematic_func=None,
    metric_prefix="Test",
):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe_error_sum = 0
    frame_count_sum = 0
    nb_sample = 0

    for batch in tqdm(val_loader):
        clip_text, motions, m_length = batch
        motions = motions.to(device).float()
        m_length = m_length.to(device)
        bs = motions.shape[0]

        et, _ = eval_wrapper.encode_text(clip_text, sample_mean=True)

        if len(motions.shape) == 3:
            motions_eval = motions
            motions_input = rearrange(motions, "b l d -> b l 1 d")
        elif len(motions.shape) == 4:
            motions_eval = rearrange(motions, "b l 1 d -> b l d")
            motions_input = motions
        else:
            raise ValueError(f"Invalid motions shape: {motions.shape}")
        m_length_input = m_length if use_m_length else None
        fid_em, em, _ = eval_wrapper.encode_motion(
            motions_eval[..., :148], m_length, sample_mean=True
        )

        pred_pose_eval = net(motions_input, m_lengths=m_length_input)
        pred_motions = rearrange(pred_pose_eval, "b l 1 d -> b l d")
        fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(
            pred_motions[..., :148], m_length, sample_mean=True
        )

        if forward_kinematic_func is not None:
            batch_mpjpe_error, batch_frame_count = calculate_mpjpe_snap(
                forward_kinematic_func(val_loader.dataset, motions_eval, skeleton),
                forward_kinematic_func(val_loader.dataset, pred_motions, skeleton),
                mask=length_to_mask(m_length, motions.shape[1], device=motions.device),
                only_local=False,
            )

            mpjpe_error_sum += batch_mpjpe_error
            frame_count_sum += batch_frame_count

        motion_annotation_list.append(fid_em)
        motion_pred_list.append(fid_em_pred)

        et_np = et.cpu().numpy()
        em_np = em.cpu().numpy()
        em_pred_np = em_pred.cpu().numpy()

        temp_R = calculate_R_precision(
            et_np, em_np, top_k=3, sum_all=True, is_cosine_sim=True
        )
        temp_match = (cosine_similarity_matrix(et_np, em_np)).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match

        temp_R = calculate_R_precision(
            et_np, em_pred_np, top_k=3, sum_all=True, is_cosine_sim=True
        )
        temp_match = (cosine_similarity_matrix(et_np, em_pred_np)).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    msg = (
        "--> \t Eva. Ep/Re %d :, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, "
        "R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), "
        "matching_real. %.4f, matching_pred. %.4f"
        % (
            ep,
            fid,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
        )
    )
    mpjpe = None
    if forward_kinematic_func is not None:
        mpjpe = mpjpe_error_sum / frame_count_sum
        msg += f", MPJPE. {mpjpe:.4f}"
        if draw and writer is not None:
            writer.log({f"{metric_prefix}/MPJPE": mpjpe}, step=ep)

    print(msg)

    if draw and writer is not None:
        writer.log({f"{metric_prefix}/FID": fid}, step=ep)
        writer.log({f"{metric_prefix}/Diversity": diversity}, step=ep)
        writer.log({f"{metric_prefix}/top1": R_precision[0]}, step=ep)
        writer.log({f"{metric_prefix}/top2": R_precision[1]}, step=ep)
        writer.log({f"{metric_prefix}/top3": R_precision[2]}, step=ep)
        writer.log({f"{metric_prefix}/matching_score": matching_score_pred}, step=ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw:
            print(msg)
        best_fid = fid
        if save:
            torch.save(
                {"ae": net.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_fid.tar"),
            )

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!" % (
            best_div,
            diversity,
        )
        if draw:
            print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (
            best_top1,
            R_precision[0],
        )
        if draw:
            print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f !!!" % (
            best_top2,
            R_precision[1],
        )
        if draw:
            print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (
            best_top3,
            R_precision[2],
        )
        if draw:
            print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = "--> --> \t matching_score Improved from %.5f to %.5f !!!" % (
            best_matching,
            matching_score_pred,
        )
        if draw:
            print(msg)
        best_matching = matching_score_pred

    if mpjpe is not None and mpjpe < best_mpjpe:
        msg = "--> --> \t MPJPE Improved from %.5f to %.5f !!!" % (
            best_mpjpe,
            mpjpe,
        )
        if draw:
            print(msg)
        best_mpjpe = mpjpe
        if save:
            torch.save(
                {"ae": net.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_mpjpe.tar"),
            )

    net.train()
    return (
        best_fid,
        best_div,
        best_top1,
        best_top2,
        best_top3,
        best_matching,
        best_mpjpe,
        writer,
    )


def length_to_mask(
    length: torch.Tensor | list[int], max_len: int, device: torch.device | None = None
) -> torch.Tensor:
    if not isinstance(length, torch.Tensor):
        length = torch.as_tensor(length)
    if device is None:
        device = length.device
    length = length.to(device)
    return torch.arange(max_len, device=device).unsqueeze(0) < length.unsqueeze(1)


def calculate_mpjpe_snap(
    preds: torch.Tensor,
    gts: torch.Tensor,
    mask: torch.Tensor,
    only_local: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return total masked joint error and number of valid joints."""
    if preds.shape != gts.shape:
        raise ValueError(f"Joint shape mismatch: {preds.shape} vs {gts.shape}")
    if only_local:
        preds = preds - preds[..., :1, :]
        gts = gts - gts[..., :1, :]
    error = torch.norm(preds - gts, dim=-1)
    joint_mask = mask.to(preds.device).unsqueeze(-1).expand_as(error)
    return error.masked_fill(~joint_mask, 0).sum(), joint_mask.sum()
