from typing import Optional
import torch
from liger_kernel.ops.linear import LigerLinearFunction


class LigerLinear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`"""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty((out_features,), device=device, dtype=dtype)
            )

    def forward(self, x: torch.Tensor):
        return LigerLinearFunction.apply(x, self.weight, self.bias)
