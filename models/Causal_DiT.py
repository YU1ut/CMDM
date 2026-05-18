import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusions.transport import Sampler, create_transport
from einops import rearrange
from models.BERT.BERT_encoder import load_bert
from models.Transformer import AttentionBlock, rope_params


class DiT(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_mode,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0,
        clip_dim=512,
        diff_model="Flow",
        cond_mask_prob=0.1,
        num_frame_per_block=3,
        is_causal=True,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_frame_per_block = num_frame_per_block

        self.cond_mode = cond_mode
        self.cond_mask_prob = cond_mask_prob
        self.is_causal = is_causal
        if self.is_causal:
            print("Using Causal Transformer")
        else:
            print("Using Non-Causal Transformer")

        print("Loading CMDM...")
        self.t_embedder = TimestepEmbedder(self.latent_dim)

        # Patchification
        self.x_embedder = nn.Linear(self.input_dim, self.latent_dim)

        print("DiT causal init")
        print("latent_dim: ", self.latent_dim)
        print("ff_size: ", self.ff_size)
        print("num_heads: ", self.num_heads)
        print("num_layers: ", self.num_layers)
        print("num_frame_per_block: ", self.num_frame_per_block)

        max_length = kwargs.get("max_length", 49)
        self.freqs = rope_params(max_length, self.latent_dim // self.num_heads)
        print("max_length: ", max_length)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    self.latent_dim,
                    self.ff_size,
                    self.num_heads,
                    qk_norm=True,
                    cross_attn_norm=True,
                    eps=1e-6,
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.cond_mode == "text":
            self.y_embedder = nn.Linear(self.clip_dim, self.latent_dim)
        else:
            raise KeyError("Unsupported condition mode!!!")

        self.final_layer = nn.Linear(self.latent_dim, self.input_dim)
        self.initialize_weights()

        if self.cond_mode == "text":
            print("Loading CLIP...")
            self.clip_model = self.load_and_freeze_clip()

        self.chunk_size = kwargs.get("chunk_size", 0)
        self.n_tokens = kwargs.get("n_tokens", 197)
        self.num_past_latents = kwargs.get("num_past_latents", 1)
        self.clip_noise = kwargs.get("clip_noise", 20.0)
        self.uncertainty_scale = kwargs.get("uncertainty_scale", 2)
        self.sampling_timesteps = kwargs.get("sampling_timesteps", 50)
        self.scheduling_matrix_type = kwargs.get("scheduling_matrix_type", "pyramid")

        self.train_diffusion = create_transport(
            num_frame_per_block=self.num_frame_per_block
        )
        self.gen_diffusion = Sampler(self.train_diffusion)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

    def parameters_wo_clip(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

    def load_and_freeze_clip(self):
        bert_model_path = "distilbert/distilbert-base-uncased"
        clip_model = load_bert(bert_model_path)
        return clip_model

    def mask_cond(self, cond, force_mask=False):
        t, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(1, bs, 1)
            return cond * (1.0 - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        enc_text, mask = self.clip_model(
            raw_text
        )  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        return enc_text, mask

    def forward_step(self, x, t, conds, conds_mask, attention_mask, force_mask=False):
        t = self.t_embedder(t, dtype=x.dtype)
        conds = self.mask_cond(conds, force_mask=force_mask)
        x = self.x_embedder(x)
        x = x.flatten(2)
        conds = self.y_embedder(conds)
        conds = rearrange(conds, "len batch dim -> batch len dim", batch=x.shape[0])
        seq_lens = attention_mask.sum(dim=1)
        conds_lens = conds_mask.sum(dim=1)
        self.freqs = self.freqs.to(x.device)

        for block in self.blocks:
            x = block(
                x,
                t,
                seq_lens=seq_lens,
                freqs=self.freqs,
                context=conds,
                context_lens=conds_lens,
                causal=self.is_causal,
                num_frame_per_block=self.num_frame_per_block,
            )
        x = rearrange(x, "batch len dim -> batch len 1 dim")
        x = self.final_layer(x)
        return x

    def forward_loss(self, latents, y, m_lens):
        b, l, j, d = latents.shape
        device = latents.device

        non_pad_mask = lengths_to_mask(m_lens, l)
        latents = torch.where(
            non_pad_mask.unsqueeze(-1).unsqueeze(-1), latents, torch.zeros_like(latents)
        )

        target = latents.clone()

        force_mask = False
        if self.cond_mode == "text":
            with torch.no_grad():
                cond_vector, conds_mask = self.encode_text(y)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        attention_mask = non_pad_mask

        model_kwargs = dict(
            conds=cond_vector,
            force_mask=force_mask,
            attention_mask=attention_mask,
            conds_mask=conds_mask,
        )
        loss_dict = self.train_diffusion.training_losses(
            self.forward_step, target, model_kwargs, dim=(2, 3)
        )
        loss = loss_dict["loss"]
        loss = (loss * non_pad_mask).sum() / non_pad_mask.sum()

        return loss

    def forward_with_CFG(self, x, t, conds, conds_mask, attention_mask, cfg=1.0):
        if not cfg == 1.0:
            half = x[: len(x) // 2]
            x = torch.cat([half, half], dim=0)
        x = self.forward_step(x, t, conds, conds_mask, attention_mask)
        if not cfg == 1.0:
            cond_eps, uncond_eps = torch.split(x, len(x) // 2, dim=0)
            half_eps = uncond_eps + cfg * (cond_eps - uncond_eps)
            x = torch.cat([half_eps, half_eps], dim=0)
        return x

    def sample(
        self,
        x,
        cond_vector,
        conds_mask,
        padding_mask,
        from_noise_levels,
        to_noise_levels,
        cfg=3.0,
        prefix_latents=None,
    ):
        from_noise_levels = rearrange(from_noise_levels, "len batch -> batch len")
        to_noise_levels = rearrange(to_noise_levels, "len batch -> batch len")
        pred_v = self.forward_with_CFG(
            x, from_noise_levels, cond_vector, conds_mask, padding_mask, cfg=cfg
        )
        if prefix_latents is None:
            delta = to_noise_levels - from_noise_levels
            pred_x = x + delta.unsqueeze(-1).unsqueeze(-1) * pred_v
        else:
            pred_x0 = x + (1 - from_noise_levels).unsqueeze(-1).unsqueeze(-1) * pred_v
            if prefix_latents is not None:
                pred_x0[:, : prefix_latents.shape[1]] = prefix_latents
            new_noise = torch.randn_like(pred_x0)
            pred_x = pred_x0 * to_noise_levels.unsqueeze(-1).unsqueeze(
                -1
            ) + new_noise * (1 - to_noise_levels).unsqueeze(-1).unsqueeze(-1)
        return pred_x

    def generate(self, conds, m_lens, cond_scale=1.0, prefix_latents=None):
        device = next(self.parameters()).device

        if torch.is_tensor(m_lens):
            m_lens = m_lens.to(device=device, dtype=torch.long)
        else:
            m_lens = torch.tensor(m_lens, device=device, dtype=torch.long)

        b = len(m_lens)
        l = int(m_lens.max().item())

        non_pad_mask = lengths_to_mask(m_lens, l)
        num_prefix_frames = prefix_latents.shape[1] if prefix_latents is not None else 0
        if num_prefix_frames > 0:
            prefix_mask = torch.ones(
                (b, num_prefix_frames), device=device, dtype=torch.bool
            )
            non_pad_mask = torch.cat([prefix_mask, non_pad_mask], dim=1)

        if self.cond_mode == "text":
            with torch.no_grad():
                cond_vector, conds_mask = self.encode_text(conds)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if not cond_scale == 1.0:
            cond_vector = torch.cat([cond_vector, torch.zeros_like(cond_vector)], dim=1)
            conds_mask = torch.cat([conds_mask, conds_mask], dim=0)
            non_pad_mask = torch.cat([non_pad_mask, non_pad_mask], dim=0)
            if prefix_latents is not None:
                prefix_latents = torch.cat([prefix_latents, prefix_latents], dim=0)

        xs_pred = torch.zeros((b, 0, 1, self.input_dim), device=device)

        curr_frame = 0
        n_frames = l + num_prefix_frames
        batch_size = b
        self.device = device

        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            horizon_block = horizon // self.num_frame_per_block
            if horizon % self.num_frame_per_block != 0:
                horizon_block += 1
            scheduling_matrix = self._generate_scheduling_matrix(horizon_block)
            scheduling_matrix = scheduling_matrix.repeat(
                self.num_frame_per_block, axis=1
            )
            scheduling_matrix = scheduling_matrix[:, :horizon]
            scheduling_matrix = 1 - scheduling_matrix / self.sampling_timesteps

            chunk = torch.randn(
                (batch_size, horizon, 1, self.input_dim), device=self.device
            )
            xs_pred = torch.cat([xs_pred, chunk], dim=1)

            start_frame = max(0, curr_frame + horizon - self.n_tokens)

            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels = np.concatenate(
                    (np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m])
                )[:, None].repeat(batch_size, axis=1)
                to_noise_levels = np.concatenate(
                    (
                        np.zeros((curr_frame,), dtype=np.int64),
                        scheduling_matrix[m + 1],
                    )
                )[:, None].repeat(batch_size, axis=1)

                from_noise_levels = (
                    torch.from_numpy(from_noise_levels).to(self.device).float()
                )
                to_noise_levels = (
                    torch.from_numpy(to_noise_levels).to(self.device).float()
                )

                if not cond_scale == 1.0:
                    xs_pred_input = torch.cat([xs_pred, xs_pred], dim=0)
                    from_noise_levels = torch.cat(
                        [from_noise_levels, from_noise_levels], dim=1
                    )
                    to_noise_levels = torch.cat(
                        [to_noise_levels, to_noise_levels], dim=1
                    )

                generated_latents = self.sample(
                    xs_pred_input[:, start_frame:],
                    cond_vector=cond_vector,
                    conds_mask=conds_mask,
                    padding_mask=non_pad_mask[:, start_frame : curr_frame + horizon],
                    from_noise_levels=from_noise_levels[start_frame:],
                    to_noise_levels=to_noise_levels[start_frame:],
                    cfg=cond_scale,
                    prefix_latents=prefix_latents,
                )

                if not cond_scale == 1.0:
                    generated_latents = generated_latents.chunk(2, dim=0)[0]

                xs_pred[:, start_frame:] = generated_latents

            curr_frame += horizon

        return xs_pred[:, num_prefix_frames:]

    def generate_long(self, conds, m_lens, cond_scale=1.0, num_samples=1):
        device = next(self.parameters()).device
        self.device = device

        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        if len(conds) == 0:
            raise ValueError("conds must not be empty")
        if len(conds) != len(m_lens):
            raise ValueError(
                f"Expected len(conds) == len(m_lens), got {len(conds)} and {len(m_lens)}"
            )

        if torch.is_tensor(m_lens):
            frame_lens = m_lens.to(device=device, dtype=torch.long)
        else:
            frame_lens = torch.tensor(m_lens, device=device, dtype=torch.long)

        latent_lens = torch.clamp(frame_lens // 4, min=1)
        total_length = int(latent_lens.sum().item())

        long_latents = torch.zeros(
            num_samples, total_length, 1, self.input_dim, device=device
        )

        current_start_frame = 0
        max_past_latents = max(0, min(int(self.num_past_latents), int(self.n_tokens)))
        for i, m_len in enumerate(latent_lens.tolist()):
            print("Sampling", conds[i], m_len)

            past_latents = None
            if i > 0 and max_past_latents > 0:
                past_start_frame = max(0, current_start_frame - max_past_latents)
                past_latents = long_latents[:, past_start_frame:current_start_frame]

            segment_lens = torch.full(
                (num_samples,), m_len, device=device, dtype=torch.long
            )
            long_latents[
                :, current_start_frame : current_start_frame + m_len
            ] = self.generate(
                [conds[i]] * num_samples,
                segment_lens,
                cond_scale=cond_scale,
                prefix_latents=past_latents,
            )
            current_start_frame += m_len

        return long_latents

    def _generate_scheduling_matrix(self, horizon: int):
        match self.scheduling_matrix_type:
            case "pyramid":
                return self._generate_pyramid_scheduling_matrix(
                    horizon, self.uncertainty_scale
                )
            case "full_sequence":
                return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(
                    horizon.cpu().numpy(), axis=1
                )
            case "autoregressive":
                return self._generate_pyramid_scheduling_matrix(
                    horizon, self.sampling_timesteps
                )
            case "trapezoid":
                return self._generate_trapezoid_scheduling_matrix(
                    horizon, self.uncertainty_scale
                )

    def _generate_pyramid_scheduling_matrix(
        self, horizon: int, uncertainty_scale: float
    ):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = (
                    self.sampling_timesteps + int(t * uncertainty_scale) - m
                )

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def _generate_trapezoid_scheduling_matrix(
        self, horizon: int, uncertainty_scale: float
    ):
        height = self.sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = (
                    self.sampling_timesteps + int(t * uncertainty_scale) - m
                )
                scheduling_matrix[m, -t] = (
                    self.sampling_timesteps + int(t * uncertainty_scale) - m
                )

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)


#################################################################################
#                                     CMDM Zoos                                #
#################################################################################
def dit(**kwargs):
    layer = 8
    return DiT(
        latent_dim=layer * 64,
        ff_size=layer * 64 * 2,
        num_layers=layer,
        num_heads=layer // 2,
        dropout=0,
        clip_dim=768,
        diff_model="Flow",
        cond_mask_prob=0.1,
        **kwargs,
    )


def dit_large(**kwargs):
    layer = 20
    return DiT(
        latent_dim=512,
        ff_size=1024,
        num_layers=layer,
        num_heads=4,
        dropout=0,
        clip_dim=768,
        diff_model="Flow",
        cond_mask_prob=0.1,
        **kwargs,
    )


def dit_light(**kwargs):
    return DiT(
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0,
        clip_dim=768,
        diff_model="Flow",
        cond_mask_prob=0.1,
        **kwargs,
    )


#################################################################################
#                                 Inner Architectures                           #
#################################################################################


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, dtype=torch.float32):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        if t.dim() == 1:
            t = t.unsqueeze(1)

        B, L = t.shape
        t = t.reshape(B * L)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
        ).to(device=t.device, dtype=dtype)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        embedding = embedding.reshape(B, L, -1)
        return embedding

    def forward(self, t, dtype=torch.bfloat16):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, dtype=dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask  # (b, len)
