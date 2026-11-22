# Rotary position embedding on only the first `rotary_dim` features of each
# head (rotate-half layout); the rest pass through unchanged. Qwen3.5 rotates
# a quarter of its 256-wide heads.
import numpy as np

import ie

name = "PartialRoPE"
category = "Attention parts"
description = "RoPE on the first rotary_dim features of each head of [B, H, S, D], offset by the cache position"

inputs = ["x"]
outputs = ["y"]

attrs = {
    "rotary_dim": {"default": "$rotary_dim", "help": "features rotated per head (even)"},
    "theta": {"default": 10000.0, "help": "RoPE base frequency"},
}


def shape(a, x):
    if len(x) != 4:
        raise ValueError(f"x must be [B, H, S, D], got {x}")
    if a.rotary_dim % 2 or (isinstance(x[3], int) and a.rotary_dim > x[3]):
        raise ValueError(f"rotary_dim {a.rotary_dim} must be even and at most the head size {x[3]}")
    return x


def forward(ctx, x):
    r = ctx.attrs.rotary_dim
    xs = x.numpy()
    seq = xs.shape[2]
    inv_freq = 1.0 / ctx.attrs.theta ** (np.arange(0, r, 2, dtype=np.float32) / r)
    angles = np.arange(ctx.start_pos, ctx.start_pos + seq, dtype=np.float32)[:, None] * inv_freq
    cos, sin = np.cos(angles), np.sin(angles)              # [S, r/2]
    x1, x2 = xs[..., : r // 2], xs[..., r // 2: r]
    out = xs.copy()
    out[..., : r // 2] = x1 * cos - x2 * sin
    out[..., r // 2: r] = x2 * cos + x1 * sin
    return ie.tensor(out)
