from typing import Optional
import torch
from liger_kernel.ops.linear import LigerLinearFunction


class LigerLinear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        return LigerLinearFunction.apply(x, weight, bias)
