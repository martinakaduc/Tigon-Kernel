from typing import Optional
import torch
from liger_kernel.ops.conv2d_transpose import LigerConv2dTransposeFunction


class LigerConv2dTranspose(torch.nn.Module):
    r"""Applies a conv2D transpose function to the incoming data
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        return LigerConv2dTransposeFunction.apply(x, weight, bias)
