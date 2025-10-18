import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.fused_linear_mse import LigerFusedLinearMSELoss
from liger_kernel.utils import infer_device

device = infer_device()


class TorchLMHeadMSE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based MSE loss.

    :param H: hidden size
    :param V: vocab size (output size)
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, reduction: str = "mean"):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.mse_loss = torch.nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        logits = self.lin(x)
        return self.mse_loss(logits, y)


class LigerLMHeadMSE(torch.nn.Module):
    def __init__(
        self, H: int, V: int, dtype: torch.dtype, reduction: str = "mean", accum_dtype=None
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.mse_loss = LigerFusedLinearMSELoss(reduction=reduction, accum_dtype=accum_dtype)

    def forward(self, x, y):
        return self.mse_loss(self.lin.weight, x, y)


#############################################################################
# Test the memory consumption of the linear fused MSE loss
#############################################################################


def bench_memory_fused_linear_mse(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    BT = input.x
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider

    torch_lm_head_mse = TorchLMHeadMSE(H=H, V=V, dtype=dtype).to(device)
    liger_lm_head_mse = LigerLMHeadMSE(H=H, V=V, dtype=dtype).to(device)
    liger_lm_head_mse_fp32_accum = LigerLMHeadMSE(
        H=H, V=V, dtype=dtype, accum_dtype=torch.float32
    ).to(device)

    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randn(BT, V, dtype=dtype, device=device)

    def fwd():
        if provider == "liger":
            return liger_lm_head_mse(_input, target)
        elif provider == "liger-fp32-accum":
            return liger_lm_head_mse_fp32_accum(_input, target)
        elif provider == "torch":
            return torch_lm_head_mse(_input, target)

    def full():
        y = fwd()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


#############################################################################
# Test the speed of the fused linear MSE loss
#############################################################################


def bench_speed_fused_linear_mse(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    BT = input.x
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    torch_lm_head_mse = TorchLMHeadMSE(H=H, V=V, dtype=dtype).to(device)
    liger_lm_head_mse = LigerLMHeadMSE(H=H, V=V, dtype=dtype).to(device)
    liger_lm_head_mse_fp32_accum = LigerLMHeadMSE(
        H=H, V=V, dtype=dtype, accum_dtype=torch.float32
    ).to(device)

    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randn(BT, V, dtype=dtype, device=device)

    def fwd():
        if provider == "liger":
            return liger_lm_head_mse(_input, target)
        elif provider == "liger-fp32-accum":
            return liger_lm_head_mse_fp32_accum(_input, target)
        elif provider == "torch":
            return torch_lm_head_mse(_input, target)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y = fwd()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            grad_to_none=[_input],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            rep=100,
            quantiles=QUANTILES,
        )
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "fused_linear_mse",
        "x_name": "BT",
        "x_label": "B x T",
        "x_values": [2**i for i in range(12, 16)],
        "kernel_providers": ["liger", "liger-fp32-accum", "torch"],
        "extra_benchmark_configs": [{"H": 4096, "V": 128256, "mode": "forward", "dtype": torch.bfloat16}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_fused_linear_mse,
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_fused_linear_mse,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
