import collections
from itertools import repeat
from typing import Optional
import torch
from liger_kernel.ops.conv2d_transpose import LigerConv2dTransposeFunction

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    parse.__name__ = name
    return parse

_pair = _ntuple(2, "_pair")

class LigerConv2dTranspose(torch.nn.ConvTranspose2d):
    r"""ConvTranspose2d using a custom Liger kernel in forward()."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor, output_size: Optional[list[int]] = None):
        return LigerConv2dTransposeFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.output_padding)
