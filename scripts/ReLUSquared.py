# Squared ReLU activation (from the Primer paper): relu(x) ** 2.
import ie

name = "ReLUSquared"
category = "Custom"
description = "relu(x)^2, an activation to try inside MLP blocks"


def forward(ctx, x):
    r = ie.relu(x)
    return r * r
