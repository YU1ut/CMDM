import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class VAE(nn.Module):
    """Conv1D causal VAE with optional AR prior over the latent timeline.
    - Encoder outputs stats for q(z|x) at length T' (downsampled by time_ds)
    - Decoder upsamples back to original T
    """

    def __init__(
        self,
        input_width=272,
        output_emb_width=16,
        hidden_size=1024,
        down_t=2,
        stride_t=2,
        width=1024,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
        clip_range=[-30, 20],
        root_dim=4,
        vel_dim=-1,
        root_loss_weight=0.0,
        vel_loss_weight=0.0,
        kl_weight=1.0,
        vf_weight=1.0,
        adaptive_vf=True,
        aux_feature_dim=256,
        down_scale=4,
        reverse_proj=True,
        nll_reduction="sum",
        vel_loss_mode="feature_prefix",
    ):
        super().__init__()
        self.input_width = input_width
        self.output_emb_width = output_emb_width

        self.decode_proj = nn.Linear(output_emb_width, hidden_size)

        self.encoder = CausalEncoder(
            input_width,
            hidden_size,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
            latent_dim=output_emb_width,
            clip_range=clip_range,
        )
        self.decoder = CausalDecoder(
            input_width,
            hidden_size,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )

        self.root_loss_weight = root_loss_weight
        self.vel_loss_weight = vel_loss_weight
        self.kl_weight = kl_weight
        self.down_scale = down_scale
        self.encode_dim = "b l d"

        self.loss_with_vf = ReConsLossWithVF(
            motion_dim=input_width,
            root_dim=root_dim,
            vel_dim=vel_dim,
            vf_weight=vf_weight,
            adaptive_vf=adaptive_vf,
            cos_margin=0.5,
            distmat_margin=0.25,
            distmat_weight=1.0,
            cos_weight=1.0,
            down_scale=down_scale,
            nll_reduction=nll_reduction,
            vel_loss_mode=vel_loss_mode,
        )

        self.reverse_proj = reverse_proj
        if reverse_proj:
            self.linear_proj_z = nn.Linear(output_emb_width, aux_feature_dim)
        else:
            self.linear_proj_aux = nn.Linear(aux_feature_dim, output_emb_width)
        self.latent_norm_eps = 1e-6
        self.register_buffer(
            "latent_mean", torch.zeros(1, 1, output_emb_width), persistent=False
        )
        self.register_buffer(
            "latent_std", torch.ones(1, 1, output_emb_width), persistent=False
        )

    def _format_latent_stat(self, stat, name: str) -> torch.Tensor:
        stat = torch.as_tensor(stat, dtype=torch.float32)
        if stat.ndim == 0:
            stat = stat.repeat(self.output_emb_width)
        elif stat.ndim == 1:
            pass
        elif stat.shape[-1] == self.output_emb_width:
            stat = stat.reshape(-1, self.output_emb_width)
            if stat.shape[0] != 1:
                raise ValueError(
                    f"Expected {name} to have one row, got shape {tuple(stat.shape)}"
                )
            stat = stat[0]
        else:
            raise ValueError(
                f"Unsupported {name} shape {tuple(stat.shape)} for latent dim {self.output_emb_width}"
            )

        if stat.shape[0] != self.output_emb_width:
            raise ValueError(
                f"{name} dim mismatch: expected {self.output_emb_width}, got {stat.shape[0]}"
            )

        return stat.view(1, 1, -1)

    def set_latent_stats(self, mean, std, eps: float | None = None) -> None:
        if eps is None:
            eps = self.latent_norm_eps
        mean_tensor = self._format_latent_stat(mean, "latent mean")
        std_tensor = self._format_latent_stat(std, "latent std").clamp_min(eps)
        self.latent_mean.copy_(mean_tensor.to(self.latent_mean.device))
        self.latent_std.copy_(std_tensor.to(self.latent_std.device))

    def load_latent_stats(
        self, mean_path: str, std_path: str, eps: float | None = None
    ) -> None:
        if not os.path.exists(mean_path):
            raise FileNotFoundError(f"Latent mean file not found: {mean_path}")
        if not os.path.exists(std_path):
            raise FileNotFoundError(f"Latent std file not found: {std_path}")
        mean = np.load(mean_path)
        std = np.load(std_path)
        self.set_latent_stats(mean, std, eps=eps)

    def _latent_norm_stats(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.latent_mean.to(device=z.device, dtype=z.dtype)
        std = self.latent_std.to(device=z.device, dtype=z.dtype).clamp_min(
            self.latent_norm_eps
        )
        return mean, std

    def normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        mean, std = self._latent_norm_stats(z)
        return (z - mean) / std

    def denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        mean, std = self._latent_norm_stats(z)
        return z * std + mean

    def preprocess(self, x):
        x = rearrange(x, "b l 1 d -> b d l")
        return x

    def postprocess(self, x):
        x = rearrange(x, "b d l -> b l 1 d")
        return x

    def encode(self, x, m_lengths=None, normalize: bool = True):
        x_in = self.preprocess(x)
        z, _, _ = self.encoder(x_in)
        if normalize:
            z = self.normalize_latent(z)
        if m_lengths is not None:
            z = self._mask_latent(z, m_lengths)
        return z

    def _latent_lengths(self, m_lens: torch.Tensor) -> torch.Tensor:
        return torch.div(m_lens, self.down_scale, rounding_mode="floor")

    def _mask_latent(self, z: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        latent_lens = self._latent_lengths(m_lens)
        latent_mask = length_to_mask(latent_lens, z.shape[1], z.device).unsqueeze(-1)
        return z * latent_mask

    def _mask_motion(self, x: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        motion_mask = (
            length_to_mask(m_lens, x.shape[1], x.device).unsqueeze(-1).unsqueeze(-1)
        )
        return x * motion_mask

    def forward(
        self,
        x,
        aux_feature=None,
        need_loss=False,
        kl_weight_override=None,
        return_loss_components=False,
        m_lengths=None,
    ):
        x_in = self.preprocess(x)
        z, mu, logvar = self.encoder(x_in)

        if m_lengths is not None:
            z = self._mask_latent(z, m_lengths)

        x_rec = self.decode_proj(z)
        x_rec = self.decoder(x_rec)  # (B, Cin, T)
        x_out = self.postprocess(x_rec)

        if m_lengths is not None:
            x_out = self._mask_motion(x_out, m_lengths)

        if need_loss:
            if aux_feature is not None:
                if self.reverse_proj:
                    z = self.linear_proj_z(z)
                else:
                    aux_feature = self.linear_proj_aux(aux_feature)
            rec_loss, kl_loss, root_loss, vel_loss, vf_loss = self.loss_with_vf(
                x_rec,
                x_in,
                mu,
                logvar,
                z=z,
                aux_feature=aux_feature,
                enc_last_layer=self.encoder.model[-1].conv.weight,
                m_lengths=m_lengths,
            )
            effective_kl_weight = (
                self.kl_weight if kl_weight_override is None else kl_weight_override
            )
            vel_term = (
                vel_loss * self.vel_loss_weight
                if vel_loss is not None
                else torch.zeros((), device=rec_loss.device, dtype=rec_loss.dtype)
            )
            loss = (
                rec_loss
                + effective_kl_weight * kl_loss
                + root_loss * self.root_loss_weight
                + vel_term
                + vf_loss
            )
            if return_loss_components:
                zero_scalar = torch.zeros((), device=loss.device, dtype=loss.dtype)
                loss_items = {
                    "loss_total": loss.detach(),
                    "loss_rec": rec_loss.detach(),
                    "loss_kl": kl_loss.detach(),
                    "loss_root": root_loss.detach(),
                    "loss_vel": vel_loss.detach() if vel_loss is not None else zero_scalar,
                    "loss_vf": vf_loss.detach(),
                    "kl_weight": torch.as_tensor(
                        effective_kl_weight, device=loss.device, dtype=loss.dtype
                    ),
                }
                return x_out, loss, loss_items
            return x_out, loss

        return x_out

    def decode(self, z, m_lengths=None, denormalize: bool = True):
        if denormalize:
            z = self.denormalize_latent(z)
        z = self.decode_proj(z)
        x = self.decoder(z)  # (B, Cin, T)
        x = self.postprocess(x)
        if m_lengths is not None:
            x = self._mask_motion(x, m_lengths)
        return x


def vae(**kwargs):
    return VAE(**kwargs)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation + (1 - stride)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # no padding here
            dilation=dilation,
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))  # only pad on the left
        return self.conv(x)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        input_emb_width=272,
        hidden_size=1024,
        down_t=2,
        stride_t=2,
        width=1024,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
        latent_dim=16,
        clip_range=[],
    ):
        super().__init__()
        self.clip_range = clip_range
        self.proj = nn.Linear(hidden_size, latent_dim * 2)

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2

        blocks.append(CausalConv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                CausalConv1d(input_dim, width, filter_t, stride_t, 1),
                CausalResnet1D(
                    width, depth, dilation_growth_rate, activation=activation, norm=norm
                ),
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, hidden_size, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.model(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        mu, logvar = x.chunk(2, dim=2)
        logvar = torch.clamp(logvar, self.clip_range[0], self.clip_range[1])
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class CausalDecoder(nn.Module):
    def __init__(
        self,
        input_emb_width=272,
        hidden_size=1024,
        down_t=2,
        stride_t=2,
        width=1024,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(CausalConv1d(hidden_size, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                CausalResnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=True,
                    activation=activation,
                    norm=norm,
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
                CausalConv1d(width, out_dim, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(CausalConv1d(width, input_emb_width, 3, 1, 1))

        self.model = nn.Sequential(*blocks)

    def forward(self, z):
        z = z.transpose(1, 2)
        return self.model(z)


import torch
import torch.nn as nn


class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(
        self, n_in, n_state, dilation=1, activation="silu", norm=None, dropout=None
    ):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(
                num_groups=32, num_channels=n_in, eps=1e-6, affine=True
            )
            self.norm2 = nn.GroupNorm(
                num_groups=32, num_channels=n_in, eps=1e-6, affine=True
            )
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)

        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(
            n_state,
            n_in,
            1,
            1,
            0,
        )

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        dilation_growth_rate=1,
        reverse_dilation=True,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = [
            ResConv1DBlock(
                n_in,
                n_in,
                dilation=dilation_growth_rate**depth,
                activation=activation,
                norm=norm,
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class CausalResConv1DBlock(nn.Module):
    def __init__(
        self, n_in, n_state, dilation=1, activation="silu", norm=None, dropout=None
    ):
        super().__init__()
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(
                num_groups=32, num_channels=n_in, eps=1e-6, affine=True
            )
            self.norm2 = nn.GroupNorm(
                num_groups=32, num_channels=n_in, eps=1e-6, affine=True
            )
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.left_padding = (3 - 1) * dilation

        self.conv1 = nn.Conv1d(
            n_in, n_state, kernel_size=3, stride=1, padding=0, dilation=dilation
        )
        self.conv2 = nn.Conv1d(n_state, n_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
            x = self.activation1(x)
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = nn.functional.pad(x, (self.left_padding, 0))

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
            x = self.activation2(x)
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class CausalResnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        dilation_growth_rate=1,
        reverse_dilation=True,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = [
            CausalResConv1DBlock(
                n_in,
                n_in,
                dilation=dilation_growth_rate**depth,
                activation=activation,
                norm=norm,
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class ReConsLoss(nn.Module):
    def __init__(
        self,
        motion_dim=263,
        root_dim=4,
        vel_dim=-1,
        nll_reduction="sum",
        vel_loss_mode="feature_prefix",
    ):
        super(ReConsLoss, self).__init__()
        self.motion_dim = motion_dim
        self.root_dim = root_dim
        self.vel_dim = vel_dim
        if nll_reduction not in {"sum", "batch_mean"}:
            raise ValueError(f"Unsupported nll_reduction: {nll_reduction}")
        if vel_loss_mode not in {"feature_prefix", "temporal_delta"}:
            raise ValueError(f"Unsupported vel_loss_mode: {vel_loss_mode}")
        self.nll_reduction = nll_reduction
        self.vel_loss_mode = vel_loss_mode

    def softclip(self, tensor, min):
        result_tensor = min + F.softplus(tensor - min)
        return result_tensor

    def gaussian_nll(self, mu, log_sigma, x):
        return (
            0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
            + log_sigma
            + 0.5 * np.log(2 * np.pi)
        )

    def _length_mask(
        self,
        m_lengths: torch.Tensor | None,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if m_lengths is None:
            return None
        return length_to_mask(m_lengths, seq_len, device=device).unsqueeze(-1).to(dtype)

    def _gaussian_nll_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        m_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = self._length_mask(
            m_lengths,
            pred.shape[1],
            device=pred.device,
            dtype=pred.dtype,
        )
        sq_error = (target - pred) ** 2
        if mask is None:
            mse = sq_error.mean().clamp_min(1e-8)
            log_sigma = self.softclip(mse.sqrt().log(), -6)
            nll = self.gaussian_nll(pred, log_sigma, target)
            loss = nll.sum()
            return loss / pred.shape[0] if self.nll_reduction == "batch_mean" else loss

        denom = mask.expand_as(sq_error).sum().clamp_min(1.0)
        mse = (sq_error * mask).sum() / denom
        mse = mse.clamp_min(1e-8)
        log_sigma = self.softclip(mse.sqrt().log(), -6)
        nll = self.gaussian_nll(pred, log_sigma, target)
        loss = (nll * mask).sum()
        return loss / pred.shape[0] if self.nll_reduction == "batch_mean" else loss

    def forward(self, motion_pred, motion_gt, m_lengths: torch.Tensor | None = None):
        motion_pred = rearrange(motion_pred, "b d l -> b l d")
        motion_gt = rearrange(motion_gt, "b d l -> b l d")
        return self._gaussian_nll_loss(
            motion_pred[..., : self.motion_dim],
            motion_gt[..., : self.motion_dim],
            m_lengths,
        )

    def forward_KL(self, mu, logvar, m_lengths: torch.Tensor | None = None):
        loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if m_lengths is not None:
            latent_mask = length_to_mask(m_lengths, mu.shape[1], device=mu.device)
            latent_mask = latent_mask.unsqueeze(-1).to(loss.dtype)
            loss = loss * latent_mask
        loss = loss.sum(dim=(1, 2))
        return loss.mean()

    def forward_root(self, motion_pred, motion_gt, m_lengths: torch.Tensor | None = None):
        motion_pred = rearrange(motion_pred, "b d l -> b l d")
        motion_gt = rearrange(motion_gt, "b d l -> b l d")
        return self._gaussian_nll_loss(
            motion_pred[..., : self.root_dim],
            motion_gt[..., : self.root_dim],
            m_lengths,
        )

    def forward_vel(self, motion_pred, motion_gt, m_lengths: torch.Tensor | None = None):
        motion_pred = rearrange(motion_pred, "b d l -> b l d")
        motion_gt = rearrange(motion_gt, "b d l -> b l d")
        motion_pred = motion_pred[..., : self.vel_dim]
        motion_gt = motion_gt[..., : self.vel_dim]
        if self.vel_loss_mode == "temporal_delta":
            motion_pred = motion_pred[:, 1:] - motion_pred[:, :-1]
            motion_gt = motion_gt[:, 1:] - motion_gt[:, :-1]
            if m_lengths is not None:
                m_lengths = (m_lengths - 1).clamp_min(0)
        return self._gaussian_nll_loss(
            motion_pred,
            motion_gt,
            m_lengths,
        )


class ReConsLossWithVF(nn.Module):
    def __init__(
        self,
        motion_dim=263,
        root_dim=4,
        vel_dim=-1,
        vf_weight=1.0,
        adaptive_vf=False,
        cos_margin=0.0,
        distmat_margin=0.0,
        distmat_weight=1.0,
        cos_weight=1.0,
        down_scale=4,
        nll_reduction="sum",
        vel_loss_mode="feature_prefix",
    ):

        super().__init__()
        self.recons_loss = ReConsLoss(
            motion_dim=motion_dim,
            root_dim=root_dim,
            vel_dim=vel_dim,
            nll_reduction=nll_reduction,
            vel_loss_mode=vel_loss_mode,
        )
        self.root_dim = root_dim
        self.vel_dim = vel_dim
        self.down_scale = down_scale

        self.distmat_weight = distmat_weight
        self.cos_weight = cos_weight

        self.vf_weight = vf_weight
        self.adaptive_vf = adaptive_vf
        self.cos_margin = cos_margin
        self.distmat_margin = distmat_margin

    def calculate_adaptive_weight_vf(self, nll_loss, vf_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(
                nll_loss, last_layer, retain_graph=True, allow_unused=True
            )[0]
            vf_grads = torch.autograd.grad(
                vf_loss, last_layer, retain_graph=True, allow_unused=True
            )[0]
        else:
            raise ValueError("last_layer is not provided")

        if nll_grads is None or vf_grads is None:
            return torch.as_tensor(
                self.vf_weight, device=nll_loss.device, dtype=nll_loss.dtype
            )

        vf_weight = torch.norm(nll_grads) / torch.norm(vf_grads).clamp_min(1e-4)
        vf_weight = torch.clamp(vf_weight, 0.0, 1e4).detach()
        vf_weight = vf_weight * self.vf_weight
        return vf_weight

    def forward(
        self,
        x_rec,
        x_in,
        mu,
        logvar,
        last_layer=None,
        z=None,
        aux_feature=None,
        enc_last_layer=None,
        m_lengths=None,
    ):

        latent_lengths = None
        if m_lengths is not None:
            latent_lengths = torch.div(m_lengths, self.down_scale, rounding_mode="floor")

        rec_loss = self.recons_loss(x_rec, x_in, m_lengths=m_lengths)
        kl_loss = self.recons_loss.forward_KL(mu, logvar, m_lengths=latent_lengths)
        root_loss = self.recons_loss.forward_root(x_rec, x_in, m_lengths=m_lengths)
        vel_loss = (
            self.recons_loss.forward_vel(x_rec, x_in, m_lengths=m_lengths)
            if self.vel_dim != -1
            else None
        )
        if (
            z is not None
            and aux_feature is not None
            and aux_feature.shape[0] >= self.down_scale
            and aux_feature.shape[0] % self.down_scale == 0
        ):
            aux_feature = reduce(aux_feature, "(l h) b d -> b l d", "mean", h=self.down_scale)
            seq_len = min(aux_feature.shape[1], z.shape[1])
            z = z[:, :seq_len]
            aux_feature = aux_feature[:, :seq_len]

            latent_mask = None
            if latent_lengths is not None:
                latent_lengths = torch.clamp(latent_lengths, max=seq_len)
                latent_mask = length_to_mask(
                    latent_lengths, seq_len, device=z.device
                ).to(z.dtype)

            z_norm = torch.nn.functional.normalize(z, dim=2, eps=1e-6)
            aux_feature_norm = torch.nn.functional.normalize(
                aux_feature, dim=2, eps=1e-6
            )
            z_cos_sim = torch.einsum("bic,bjc->bij", z_norm, z_norm)
            aux_feature_cos_sim = torch.einsum(
                "bic,bjc->bij", aux_feature_norm, aux_feature_norm
            )
            diff = torch.abs(z_cos_sim - aux_feature_cos_sim)

            cos_distance = torch.nn.functional.relu(
                1
                - self.cos_margin
                - torch.nn.functional.cosine_similarity(aux_feature, z, dim=2, eps=1e-6)
            )
            if latent_mask is not None:
                pair_mask = latent_mask.unsqueeze(1) * latent_mask.unsqueeze(2)
                pair_denom = pair_mask.sum().clamp_min(1.0)
                vf_loss_1 = (
                    torch.nn.functional.relu(diff - self.distmat_margin) * pair_mask
                ).sum() / pair_denom
                cos_denom = latent_mask.sum().clamp_min(1.0)
                vf_loss_2 = (cos_distance * latent_mask).sum() / cos_denom
            else:
                vf_loss_1 = torch.nn.functional.relu(diff - self.distmat_margin).mean()
                vf_loss_2 = cos_distance.mean()
            vf_loss = vf_loss_1 * self.distmat_weight + vf_loss_2 * self.cos_weight

        else:
            vf_loss = None

        if vf_loss is not None:
            if self.adaptive_vf:
                vf_weight = self.calculate_adaptive_weight_vf(
                    rec_loss, vf_loss, last_layer=enc_last_layer
                )
            else:
                vf_weight = self.vf_weight
            vf_loss = vf_weight * vf_loss
        else:
            vf_loss = torch.tensor(0.0, device=x_rec.device, dtype=x_rec.dtype)

        return rec_loss, kl_loss, root_loss, vel_loss, vf_loss


def length_to_mask(length, max_len, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)

    length = length.to(device)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask
