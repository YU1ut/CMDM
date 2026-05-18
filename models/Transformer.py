import torch
import torch.nn as nn


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    # compute in float32 for speed; returns complex64
    inv = torch.arange(0, dim, 2, dtype=torch.float32) / float(dim)
    base = torch.tensor(float(theta), dtype=torch.float32)
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float32), 1.0 / torch.pow(base, inv)
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, freqs):
    n, c = x.size(2), x.size(3) // 2

    # fast path: vectorized apply up to max length, then restore padding per sample
    # cast once to complex for whole batch (use float32/complex64 for speed)
    dtype_orig = x.dtype
    max_len = x.size(1)
    xb = x
    xc = torch.view_as_complex(
        xb.to(torch.float32).reshape(xb.size(0), max_len, n, -1, 2)
    )  # (B, Fmax, N, C/2)

    # slice freqs to max_len and broadcast over batch/heads
    freqs_i = freqs[:max_len].to(xc.dtype).view(max_len, 1, -1)  # (Fmax, 1, C)

    # apply rotary embedding (time-only) in one go
    yc = xc * freqs_i  # (B, Fmax, N, C/2)
    yr = torch.view_as_real(yc)
    yr = yr.flatten(3)  # (B, Fmax, N, C)
    return yr.to(dtype_orig)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    causal=False,
    num_frame_per_block=1,
    dtype=torch.float,
):
    # Build attention mask from q_lens / k_lens if provided, and combine with causal mask
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    device = q.device
    attn_mask = None
    if not (q_lens is None and k_lens is None):
        # Default to full lengths when lens are not provided
        if q_lens is None:
            q_lens = torch.full((b,), lq, dtype=torch.int32, device=device)
        else:
            q_lens = q_lens.to(device=device, dtype=torch.int32)
        if k_lens is None:
            k_lens = torch.full((b,), lk, dtype=torch.int32, device=device)
        else:
            k_lens = k_lens.to(device=device, dtype=torch.int32)

        # True indicates positions to be masked (not attended)
        q_positions = torch.arange(lq, device=device, dtype=torch.int32).unsqueeze(
            0
        )  # [1, Lq]
        k_positions = torch.arange(lk, device=device, dtype=torch.int32).unsqueeze(
            0
        )  # [1, Lk]
        q_pad = q_positions >= q_lens.unsqueeze(1)  # [B, Lq]
        k_pad = k_positions >= k_lens.unsqueeze(1)  # [B, Lk]

        # Broadcast over heads: [B, 1, Lq, Lk]
        attn_mask = q_pad.unsqueeze(1).unsqueeze(3) | k_pad.unsqueeze(1).unsqueeze(1)

    if causal:
        assert lq == lk
        total_length = lq
        ends = torch.zeros(total_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0, end=total_length, step=num_frame_per_block, device=device
        )

        for tmp in frame_indices:
            ends[tmp : tmp + num_frame_per_block] = tmp + num_frame_per_block

        def attention_mask(q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

        q_idx = torch.arange(lq, device=device).unsqueeze(1)
        k_idx = torch.arange(lk, device=device).unsqueeze(0)

        causal_mask = ~attention_mask(q_idx, k_idx).unsqueeze(0).unsqueeze(0)

        attn_mask = causal_mask if attn_mask is None else (attn_mask | causal_mask)

    if attn_mask is not None:
        attn_mask = ~attn_mask

    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
    )

    out = out.transpose(1, 2).contiguous()
    return out


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, freqs=None, causal=False, num_frame_per_block=1):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            freqs(Tensor): Rope freqs (time-only), shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = attention(
            q=rope_apply(q, freqs) if freqs is not None else q,
            k=rope_apply(k, freqs) if freqs is not None else k,
            v=v,
            k_lens=seq_lens,
            causal=causal,
            num_frame_per_block=num_frame_per_block,
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class T2VCrossAttention(SelfAttention):
    def forward(self, x, context, context_lens, freqs=None, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        q = rope_apply(q, freqs) if freqs is not None else q

        # compute attention
        x = attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self, dim, ffn_dim, num_heads, qk_norm=True, cross_attn_norm=True, eps=1e-6
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = LayerNorm(dim, eps)
        self.self_attn = SelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = (
            LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = T2VCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, 6 * self.dim, bias=True)
        )

    def forward(
        self,
        x,
        e,
        seq_lens,
        freqs,
        context,
        context_lens,
        causal=False,
        num_frame_per_block=1,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.adaLN_modulation(e)).chunk(6, dim=-1)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0],
            seq_lens,
            freqs,
            causal=causal,
            num_frame_per_block=num_frame_per_block,
        )
        # with amp.autocast(dtype=torch.float32):
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x
