from liger_kernel.ops.gelu import LigerGELUFunction
from liger_kernel.ops.linear import LigerLinearFunction
from liger_kernel.ops.mse import LigerMSEFunction
from liger_kernel.ops.relu import LigerReLUFunction
from liger_kernel.ops.silu import LigerSiLUFunction


def liger_relu(x):
    return LigerReLUFunction.apply(x)


def liger_gelu(x):
    return LigerGELUFunction.apply(x)


def liger_silu(x):
    return LigerSiLUFunction.apply(x)


def liger_linear(x, weight, bias=None):
    return LigerLinearFunction.apply(x, weight, bias)


def liger_mse(y, y_hat, reduction: str = "mean"):
    return LigerMSEFunction.apply(y, y_hat, reduction)
