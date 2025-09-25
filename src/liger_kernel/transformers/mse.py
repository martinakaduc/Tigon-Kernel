import torch
import torch.nn as nn

from liger_kernel.ops.mse import LigerMSEFunction


class LigerMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return LigerMSEFunction.apply(x)
