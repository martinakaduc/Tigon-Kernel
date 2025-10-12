import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import set_seed

from liger_kernel.ops.linear import LigerLinearFunction
from liger_kernel.diffusers.functional import liger_linear
from liger_kernel.diffusers.linear import LigerLinear
from liger_kernel.utils import infer_device

device = infer_device()

set_seed(42)


class TorchLinearModel(torch.nn.Module):
    """Ground truth implementation of the standard linear layer.

    :param H: input hidden size
    :param V: output size
    :param bias: whether to use bias
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=H, out_features=V, bias=bias, dtype=dtype, device=device
        )

    def forward(self, input_tensor):
        return self.linear(input_tensor)


class LigerLinearModel(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(V, H, dtype=dtype, device=device))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(V, dtype=dtype, device=device))
        else:
            self.bias = None

        self.linear = LigerLinear()

    def forward(self, input_tensor):
        return self.linear(input_tensor, self.weight, self.bias)


#############################################################################
# Test the correctness of the fused linear layer
#############################################################################


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (4, 423, 167, 1423),  # random shape
        (2, 2, 8, 8),  # small shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-3, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize(
    "bias",
    [True, False],
)
def test_correctness(B, T, H, V, scalar, dtype, bias, atol, rtol):
    torch_linear = TorchLinearModel(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        bias=bias,
    ).to(device)
    liger_linear = LigerLinearModel(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        bias=bias,
    ).to(device)

    # init the linear layers with the same weights
    torch_linear.linear.weight.data = liger_linear.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )
    if bias:
        torch_linear.linear.bias.data = liger_linear.bias.data = torch.rand(
            V, device=device, dtype=dtype
        )

    _tensor = torch.rand(B * T, H, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    with torch.autograd.detect_anomaly():
        output1 = torch_linear(_input1)
        output2 = liger_linear(_input2)

        assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.sum().backward()
    output2.sum().backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(
        torch_linear.linear.weight.grad,
        liger_linear.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    if bias:
        assert_verbose_allclose(
            torch_linear.linear.bias.grad,
            liger_linear.bias.grad,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 2, 8, 8),
        (9, 7, 41, 41),  # weird shapes
        (8, 128, 1024, 4096),  # larger shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (0.5, torch.bfloat16, 5e-3, 5e-2),
        (0.5, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
def test_correctness_functional(B, T, H, V, scalar, dtype, bias, atol, rtol):
    # init the linear layers with the same weights
    _weight = torch.rand(V, H, device=device, dtype=dtype)
    _weight1 = _weight.detach().clone().requires_grad_(True)
    _weight2 = _weight.detach().clone().requires_grad_(True)

    _bias = None
    _bias1 = None
    _bias2 = None
    if bias:
        _bias = torch.rand(V, device=device, dtype=dtype)
        _bias1 = _bias.detach().clone().requires_grad_(True)
        _bias2 = _bias.detach().clone().requires_grad_(True)

    _tensor = torch.rand(B * T, H, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    output1 = liger_linear(
        input_tensor=_input1,
        weight=_weight1,
        bias=_bias1,
    )
    output2 = LigerLinearFunction.apply(
        _input2,
        _weight2,
        _bias2,
    )

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.sum().backward()
    output2.sum().backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(_weight1.grad, _weight2.grad, atol=atol, rtol=rtol)

    if bias:
        assert_verbose_allclose(_bias1.grad, _bias2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (4, 423, 167, 1423),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-3, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
def test_correctness_against_torch_linear(B, T, H, V, scalar, dtype, bias, atol, rtol):
    """Test that our fused linear matches torch.nn.Linear exactly"""

    # Create torch linear layer
    torch_linear = torch.nn.Linear(H, V, bias=bias, dtype=dtype, device=device)

    # Create input
    input_tensor = torch.rand(B * T, H, device=device, dtype=dtype) * scalar
    input1 = input_tensor.detach().clone().requires_grad_(True)
    input2 = input_tensor.detach().clone().requires_grad_(True)

    # Forward pass
    torch_output = torch_linear(input1)
    liger_output = LigerLinearFunction.apply(
        input2, torch_linear.weight, torch_linear.bias
    )

    assert_verbose_allclose(torch_output, liger_output, atol=atol, rtol=rtol)

    # Backward pass
    grad_output = torch.rand_like(torch_output)

    torch_output.backward(grad_output)
    liger_output.backward(grad_output)

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (1, 1, 32, 64),  # minimal case
        (4, 16, 256, 512),  # medium case
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 5e-3, 5e-2),
        (torch.float32, 1e-5, 5e-4),
    ],
)
def test_edge_cases(B, T, H, V, dtype, atol, rtol):
    """Test edge cases like very small tensors"""

    torch_linear = TorchLinearModel(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        bias=True,
    ).to(device)
    liger_linear = LigerLinearModel(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        bias=True,
    ).to(device)

    # init the linear layers with the same weights
    torch_linear.linear.weight.data = liger_linear.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )
    torch_linear.linear.bias.data = liger_linear.bias.data = torch.rand(
        V, device=device, dtype=dtype
    )

    input_tensor = torch.rand(B * T, H, device=device, dtype=dtype)
    input1 = input_tensor.detach().clone().requires_grad_(True)
    input2 = input_tensor.detach().clone().requires_grad_(True)

    output1 = torch_linear(input1)
    output2 = liger_linear(input2)

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.sum().backward()
    output2.sum().backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "autocast_dtype, atol, rtol",
    [
        (torch.bfloat16, 5e-3, 5e-2),
        (torch.float16, 5e-3, 5e-2),
    ],
)
def test_amp(autocast_dtype, atol, rtol):
    """Test automatic mixed precision compatibility"""
    B = 2
    T = 4
    H = 2048
    V = 3200
    scalar = 1.0
    bias = True
    dtype = torch.float32

    torch_linear = TorchLinearModel(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        bias=bias,
    ).to(device)
    liger_linear = LigerLinearModel(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        bias=bias,
    ).to(device)

    # init the linear layers with the same weights
    torch_linear.linear.weight.data = liger_linear.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )
    torch_linear.linear.bias.data = liger_linear.bias.data = torch.rand(
        V, device=device, dtype=dtype
    )

    _tensor = torch.rand(B * T, H, device=device, dtype=autocast_dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    with torch.autocast(device_type=device, dtype=autocast_dtype):
        output1 = torch_linear(_input1)
        output2 = liger_linear(_input2)

        assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

        output1.sum().backward()
        output2.sum().backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_linear.linear.weight.grad,
        liger_linear.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert_verbose_allclose(
        torch_linear.linear.bias.grad,
        liger_linear.bias.grad,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (16, 256, 2048, 8192),  # large case
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias", [True, False])
def test_memory_efficiency(B, T, H, V, dtype, bias):
    """Test that fused linear doesn't use significantly more memory than torch linear"""

    # This is more of a smoke test - in practice you'd measure actual memory usage
    torch_linear = torch.nn.Linear(H, V, bias=bias, dtype=dtype, device=device)

    input_tensor = torch.rand(B * T, H, device=device, dtype=dtype, requires_grad=True)

    # Test that it runs without OOM
    output = LigerLinearFunction.apply(
        input_tensor, torch_linear.weight, torch_linear.bias
    )

    # Test backward pass
    grad_output = torch.rand_like(output)
    output.backward(grad_output)

    assert output.shape == (B * T, V)
    assert input_tensor.grad is not None
    assert input_tensor.grad.shape == (B * T, H)


def test_no_bias():
    """Test linear layer without bias"""
    B, T, H, V = 4, 8, 16, 32
    dtype = torch.float32

    weight = torch.rand(V, H, device=device, dtype=dtype, requires_grad=True)
    input_tensor = torch.rand(B * T, H, device=device, dtype=dtype, requires_grad=True)

    # Test with None bias
    output = LigerLinearFunction.apply(input_tensor, weight, None)

    # Compare with torch linear without bias
    torch_linear = torch.nn.Linear(H, V, bias=False, dtype=dtype, device=device)
    torch_linear.weight.data = weight.data.clone()

    input_torch = input_tensor.detach().clone().requires_grad_(True)
    torch_output = torch_linear(input_torch)

    assert_verbose_allclose(output, torch_output, atol=1e-5, rtol=5e-4)

    # Test backward pass
    grad_output = torch.rand_like(output)
    output.backward(grad_output)
    torch_output.backward(grad_output)

    assert_verbose_allclose(input_tensor.grad, input_torch.grad, atol=1e-5, rtol=5e-4)


if __name__ == "__main__":
    pytest.main([__file__])
