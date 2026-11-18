# LayerScale: multiply a residual branch by a small learned per-channel
# gain, which stabilises deep stacks. Insert it before a residual Add.
name = "LayerScale"
category = "Custom"
description = "x * gamma, with gamma a learned per-channel scale"

attrs = {"dim": "$d_model", "init": 0.1}


def params(a):
    return {"gamma": {"shape": [a.dim], "init": "ones"}}


def forward(ctx, x):
    return x * ctx.params.gamma * ctx.attrs.init
