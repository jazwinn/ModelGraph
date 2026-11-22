# Depthwise causal 1-D convolution over the sequence, as used in front of
# linear-attention layers (Mamba, Gated DeltaNet):
#   y[t, c] = sum_k weight[c, k] * x[t - K + 1 + k, c]
# The last K - 1 inputs are kept between calls so decoding one token at a time
# sees the same left context as a full forward pass.
import numpy as np

import ie

name = "CausalConv1d"
category = "Linear attention"
description = "Depthwise causal conv over the sequence, with the last kernel-1 inputs cached between calls"

inputs = ["x"]
outputs = ["y"]

attrs = {
    "channels": "$lin_conv_dim",
    "kernel": {"default": "$lin_conv_kernel", "help": "taps per channel"},
}

# Rolling input history per layer. Scripts have no cache slot of their own, so
# a layer is told apart by its (trained, hence unique) weights; a call at
# position 0 starts a new sequence.
_history = {}


def params(a):
    return {"weight": {"shape": [a.channels, a.kernel], "std": 0.2}}


def shape(a, x):
    if x[-1] != a.channels:
        raise ValueError(f"last dimension must be {a.channels}, got {x}")
    return x


def forward(ctx, x):
    w = ctx.params.weight.numpy()                         # [C, K]
    xs = x.numpy()                                        # [B, S, C]
    k = w.shape[1]
    key = w[:2].tobytes()
    past = _history.get(key)
    if ctx.start_pos == 0 or past is None or past.shape[0] != xs.shape[0]:
        past = np.zeros((xs.shape[0], k - 1, xs.shape[2]), np.float32)
    full = np.concatenate([past, xs], axis=1)             # [B, K-1+S, C]
    _history[key] = full[:, -(k - 1):].copy()
    seq = xs.shape[1]
    y = sum(full[:, i:i + seq] * w[:, i] for i in range(k))
    return ie.tensor(y)
