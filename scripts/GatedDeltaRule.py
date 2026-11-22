# The recurrence of Gated DeltaNet (Qwen3-Next, Qwen3.5 linear attention).
# Each value head keeps a [dk, dv] memory S that decays and is corrected
# towards the new key/value pair every token:
#   S = exp(g) * S;  S += k (beta * (v - k^T S));  o = q^T S
# with g = -exp(A_log) * softplus(a + dt_bias) and beta = sigmoid(b).
# Queries and keys are L2-normalised and shared by groups of value heads.
import numpy as np

import ie

name = "GatedDeltaRule"
category = "Linear attention"
description = "Gated delta-rule recurrence: a decaying per-head memory updated by each key/value pair"

inputs = ["qkv", "b", "a"]
outputs = ["out"]

attrs = {
    "num_k_heads": "$lin_k_heads",
    "num_v_heads": "$lin_v_heads",
    "head_dim": {"default": "$lin_head_dim", "help": "size of each key and value head"},
}

# Recurrent state per layer, keyed by the layer's own A_log (see CausalConv1d).
_state = {}


def params(a):
    return {
        "A_log": {"shape": [a.num_v_heads], "init": "zeros"},
        "dt_bias": {"shape": [a.num_v_heads], "init": "ones"},
    }


def shape(a, qkv, b, gate):
    width = (2 * a.num_k_heads + a.num_v_heads) * a.head_dim
    if qkv[-1] != width:
        raise ValueError(f"qkv must end in {width} (q, k, v heads), got {qkv}")
    for n, s in (("b", b), ("a", gate)):
        if s[-1] != a.num_v_heads:
            raise ValueError(f"{n} must end in num_v_heads = {a.num_v_heads}, got {s}")
    return (*qkv[:-1], a.num_v_heads * a.head_dim)


def _l2norm(x):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)


def forward(ctx, qkv, b, a):
    at = ctx.attrs
    hk, hv, d = at.num_k_heads, at.num_v_heads, at.head_dim
    x = qkv.numpy()
    bs, seq = x.shape[:2]
    q = _l2norm(x[..., : hk * d].reshape(bs, seq, hk, d)) * d ** -0.5
    k = _l2norm(x[..., hk * d: 2 * hk * d].reshape(bs, seq, hk, d))
    v = x[..., 2 * hk * d:].reshape(bs, seq, hv, d)
    q = np.repeat(q, hv // hk, axis=2)
    k = np.repeat(k, hv // hk, axis=2)

    A_log = ctx.params.A_log.numpy()
    beta = 1 / (1 + np.exp(-b.numpy()))                                   # [B, S, Hv]
    decay = np.exp(-np.exp(A_log) * np.logaddexp(0, a.numpy() + ctx.params.dt_bias.numpy()))

    key = A_log.tobytes()
    S = _state.get(key)
    if ctx.start_pos == 0 or S is None or S.shape[0] != bs:
        S = np.zeros((bs, hv, d, d), np.float32)
    out = np.empty((bs, seq, hv, d), np.float32)
    for t in range(seq):
        S *= decay[:, t, :, None, None]
        kt = k[:, t]
        delta = (v[:, t] - np.einsum("bhkv,bhk->bhv", S, kt)) * beta[:, t, :, None]
        S += kt[..., :, None] * delta[..., None, :]
        out[:, t] = np.einsum("bhkv,bhk->bhv", S, q[:, t])
    _state[key] = S
    return ie.tensor(out.reshape(bs, seq, hv * d))
