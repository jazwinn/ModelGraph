# RMSNorm per head, then gated: rmsnorm(x) * weight * silu(z).
# Gated DeltaNet uses it on the recurrence output before the out projection.
import ie

name = "GatedRMSNorm"
category = "Linear attention"
description = "Per-head RMSNorm multiplied by silu(gate)"

inputs = ["x", "gate"]
outputs = ["y"]

attrs = {
    "head_dim": "$lin_head_dim",
    "eps": 1e-6,
}


def params(a):
    return {"weight": {"shape": [a.head_dim], "init": "ones"}}


def shape(a, x, gate):
    if x[-1] != gate[-1] or x[-1] % a.head_dim:
        raise ValueError(f"x and gate must match and split into heads of {a.head_dim}: {x} vs {gate}")
    return x


def forward(ctx, x, gate):
    d = ctx.attrs.head_dim
    bs, seq, width = x.shape
    h = x.reshape(bs, seq, width // d, d)
    h = h / ie.sqrt(ie.mean(h * h) + ctx.attrs.eps) * ctx.params.weight
    return h.reshape(bs, seq, width) * ie.silu(gate)
