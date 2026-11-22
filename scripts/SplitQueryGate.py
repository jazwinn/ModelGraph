# Gated attention (Qwen3-Next, Qwen3.5) projects each query head to twice its
# size: the first half is the query, the second an output gate applied as
# attn * sigmoid(gate). This splits [B, S, H*2*D] into the two [B, S, H*D].
name = "SplitQueryGate"
category = "Attention parts"
description = "[B, S, H*2*D] -> query [B, S, H*D] and gate [B, S, H*D], split per head"

inputs = ["x"]
outputs = ["query", "gate"]

attrs = {"num_heads": "$num_heads", "head_dim": "$head_dim"}


def shape(a, x):
    if x[-1] != a.num_heads * 2 * a.head_dim:
        raise ValueError(f"last dimension must be num_heads * 2 * head_dim = {a.num_heads * 2 * a.head_dim}, got {x}")
    half = (*x[:-1], a.num_heads * a.head_dim)
    return half, half


def forward(ctx, x):
    h, d = ctx.attrs.num_heads, ctx.attrs.head_dim
    bs, seq, _ = x.shape
    heads = x.reshape(bs, seq, h, 2 * d)
    return heads.slice(-1, 0, d).reshape(bs, seq, h * d), heads.slice(-1, d, d).reshape(bs, seq, h * d)
