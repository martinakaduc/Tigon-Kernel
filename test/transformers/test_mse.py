import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.transformers.functional import liger_mse
from liger_kernel.transformers.mse import LigerMSE
from liger_kernel.utils import infer_device

device = infer_device()
set_seed()


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8),
        (4, 16),
        (1, 1023),  # Large single row single-block dispatch
        (3, 7, 256),  # 3D input
        (1, 4096),  # test multi-block dispatch
        (1, 2, 4096),  # test multi-block dispatch on 3D input
        (32, 128),  # typical batch size
        (16, 512),  # medium size
    ],
)
@pytest.mark.parametrize(
    "reduction",
    ["mean", "sum", "none"],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-3, 1e-3),
        pytest.param(
            torch.bfloat16,
            5e-1,
            5e-1,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this device",
            ),
        ),
    ],
)
def test_liger_mse_loss(shape, reduction, dtype, atol, rtol):
    torch.manual_seed(0)
    input_tensor = torch.randn(*shape, dtype=dtype, device=device)
    target_tensor = torch.randn(*shape, dtype=dtype, device=device)

    input1 = input_tensor.clone().requires_grad_(True)
    target1 = target_tensor.clone()
    input2 = input_tensor.clone().requires_grad_(True)
    target2 = target_tensor.clone()

    # PyTorch reference
    torch_mse_loss = torch.nn.MSELoss(reduction=reduction)
    ref_out = torch_mse_loss(input1, target1)

    # Liger implementation
    liger_mse_loss = LigerMSE(reduction=reduction).to(device).to(dtype)
    liger_out = liger_mse_loss(input2, target2)

    assert_verbose_allclose(ref_out, liger_out, atol=atol, rtol=rtol)

    # Test gradients
    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output, retain_graph=True)
    liger_out.backward(grad_output, retain_graph=True)

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8),
        (4, 16),
        (1, 1023),
        (3, 7, 256),
        (1, 4096),
        (1, 2, 4096),
        (32, 128),
        (16, 512),
    ],
)
@pytest.mark.parametrize(
    "reduction",
    ["mean", "sum", "none"],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-3, 1e-3),
        pytest.param(
            torch.bfloat16,
            5e-1,
            5e-1,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this device",
            ),
        ),
    ],
)
def test_liger_mse_loss_functional(shape, reduction, dtype, atol, rtol):
    torch.manual_seed(0)
    input_tensor = torch.randn(*shape, dtype=dtype, device=device)
    target_tensor = torch.randn(*shape, dtype=dtype, device=device)

    input1 = input_tensor.clone().requires_grad_(True)
    target1 = target_tensor.clone()
    input2 = input_tensor.clone().requires_grad_(True)
    target2 = target_tensor.clone()

    # PyTorch reference
    ref_out = torch.nn.functional.mse_loss(input1, target1, reduction=reduction)

    # Liger functional implementation
    liger_out = liger_mse(input2, target2, reduction=reduction)

    assert_verbose_allclose(ref_out, liger_out, atol=atol, rtol=rtol)

    # Test gradients
    if reduction == "none":
        grad_output = torch.randn_like(ref_out)
    else:
        grad_output = torch.randn_like(ref_out)

    ref_out.backward(grad_output, retain_graph=True)
    liger_out.backward(grad_output, retain_graph=True)

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            5e-2,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this device",
            ),
        ),
    ],
)
def test_liger_mse_loss_edge_cases(dtype, atol, rtol):
    """Test edge cases for MSE loss."""
    torch.manual_seed(42)

    # Test identical input and target (loss should be 0)
    input_tensor = torch.randn(5, 10, dtype=dtype, device=device)
    target_tensor = input_tensor.clone()

    input1 = input_tensor.clone().requires_grad_(True)
    target1 = target_tensor.clone()
    input2 = input_tensor.clone().requires_grad_(True)
    target2 = target_tensor.clone()

    ref_out = torch.nn.functional.mse_loss(input1, target1, reduction="mean")
    liger_out = liger_mse(input2, target2, reduction="mean")

    # Loss should be very close to 0
    assert torch.abs(ref_out) < 1e-6
    assert torch.abs(liger_out) < 1e-6
    assert_verbose_allclose(ref_out, liger_out, atol=atol, rtol=rtol)

    # Test gradients (should also be 0)
    grad_output = torch.ones_like(ref_out)
    ref_out.backward(grad_output, retain_graph=True)
    liger_out.backward(grad_output, retain_graph=True)

    assert torch.allclose(input1.grad, torch.zeros_like(input1.grad), atol=1e-6)
    assert torch.allclose(input2.grad, torch.zeros_like(input2.grad), atol=1e-6)
    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shape",
    [
        (1,),  # 1D tensor
        (10,),  # 1D tensor
        (2, 3, 4, 5),  # 4D tensor
        (8, 16, 32),  # 3D tensor
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_liger_mse_loss_various_shapes(shape, reduction):
    """Test MSE loss with various tensor shapes."""
    torch.manual_seed(123)
    dtype = torch.float32
    atol, rtol = 1e-5, 1e-5

    input_tensor = torch.randn(*shape, dtype=dtype, device=device)
    target_tensor = torch.randn(*shape, dtype=dtype, device=device)

    input1 = input_tensor.clone().requires_grad_(True)
    target1 = target_tensor.clone()
    input2 = input_tensor.clone().requires_grad_(True)
    target2 = target_tensor.clone()

    ref_out = torch.nn.functional.mse_loss(input1, target1, reduction=reduction)
    liger_out = liger_mse(input2, target2, reduction=reduction)

    assert_verbose_allclose(ref_out, liger_out, atol=atol, rtol=rtol)

    # Test gradients
    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output, retain_graph=True)
    liger_out.backward(grad_output, retain_graph=True)

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)


def test_liger_mse_loss_numerical_stability():
    """Test numerical stability with large values."""
    torch.manual_seed(456)
    dtype = torch.float32
    atol, rtol = 1e-4, 1e-4

    # Test with large values
    input_tensor = torch.randn(4, 8, dtype=dtype, device=device) * 1000
    target_tensor = torch.randn(4, 8, dtype=dtype, device=device) * 1000

    input1 = input_tensor.clone().requires_grad_(True)
    target1 = target_tensor.clone()
    input2 = input_tensor.clone().requires_grad_(True)
    target2 = target_tensor.clone()

    ref_out = torch.nn.functional.mse_loss(input1, target1, reduction="mean")
    liger_out = liger_mse(input2, target2, reduction="mean")

    assert_verbose_allclose(ref_out, liger_out, atol=atol, rtol=rtol)

    # Test gradients
    grad_output = torch.ones_like(ref_out)
    ref_out.backward(grad_output, retain_graph=True)
    liger_out.backward(grad_output, retain_graph=True)

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
