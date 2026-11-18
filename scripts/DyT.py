# Dynamic Tanh: a drop-in replacement for LayerNorm / RMSNorm
#   y = weight * tanh(alpha * x) + bias
# Swap it in for a norm node to test a normalisation-free transformer.
import ie

name = "DyT"
category = "Custom"
description = "Dynamic Tanh normalisation: weight * tanh(alpha * x) + bias"

inputs = ["x"]
outputs = ["y"]

attrs = {
    "dim": "$d_model",
    "alpha_init": {"default": 0.5, "help": "starting value of the learned alpha"},
}


def params(a):
    return {
        "alpha": {"shape": [1], "init": "ones"},
        "weight": {"shape": [a.dim], "init": "ones"},
        "bias": {"shape": [a.dim], "init": "zeros"},
    }


def shape(a, x):
    if x[-1] != a.dim:
        raise ValueError(f"last dimension must be {a.dim}, got {x}")
    return x


def forward(ctx, x):
    p = ctx.params
    # alpha starts at 1 ("ones"); alpha_init rescales it without retraining.
    return ie.tanh(x * p.alpha * ctx.attrs.alpha_init) * p.weight + p.bias
