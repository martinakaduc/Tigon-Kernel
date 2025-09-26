import torch
import torch.nn as nn

from liger_kernel.ops.mse import LigerMSEFunction


class LigerMSE(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        _input: torch.Tensor,
        target: torch.Tensor,
    ):
        return LigerMSEFunction.apply(_input, target, self.reduction)
