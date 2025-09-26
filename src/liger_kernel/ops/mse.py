import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import element_mul_kernel, elementwise_mul_kernel
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device


@triton.jit
def liger_mse_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_elements,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    MSE Loss kernel.

    If reduction == "none": store elementwise squared diffs (same shape as input).
    Else: store a per-sample reduced loss value.
    """

    program_id = tl.program_id(0).to(tl.int64)

    X_ptr += program_id * X_stride
    Y_ptr += program_id * Y_stride
    loss_ptr += program_id * loss_stride

    # Accumulate scalar loss per sample if reduction != none
    loss_sum = 0.0

    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)

        x_vals = tl.load(X_ptr + offsets, mask=offsets < n_elements, other=0.0).to(
            tl.float32
        )
        y_vals = tl.load(Y_ptr + offsets, mask=offsets < n_elements, other=0.0).to(
            tl.float32
        )

        diff = x_vals - y_vals
        squared_diff = diff * diff

        if reduction == "none":
            # Write elementwise squared diffs back
            tl.store(loss_ptr + offsets, squared_diff, mask=offsets < n_elements)
        else:
            # Accumulate for later reduction
            loss_sum += tl.sum(tl.where(offsets < n_elements, squared_diff, 0.0))

    tl.debug_barrier()

    if reduction != "none":
        if reduction == "mean":
            loss = loss_sum / n_elements
        elif reduction == "sum":
            loss = loss_sum
        else:
            loss = 0.0  # should not happen
        tl.store(loss_ptr, loss)


@triton.jit
def liger_mse_grad_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    grad_ptr,
    grad_stride,
    n_elements,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    MSE gradient kernel.
    Computes gradient: 2 * (x - y) / n_elements (for mean reduction) or 2 * (x - y) (for sum/none).
    """

    program_id = tl.program_id(0).to(tl.int64)

    X_ptr += program_id * X_stride
    Y_ptr += program_id * Y_stride
    grad_ptr += program_id * grad_stride

    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)

        x_vals = tl.load(X_ptr + offsets, mask=offsets < n_elements, other=0.0).to(
            tl.float32
        )
        y_vals = tl.load(Y_ptr + offsets, mask=offsets < n_elements, other=0.0).to(
            tl.float32
        )

        grad = 2.0 * (x_vals - y_vals)
        if reduction == "mean":
            grad = grad / n_elements  # average over elements

        tl.store(grad_ptr + offsets, grad, mask=offsets < n_elements)


MAX_FUSED_SIZE = 4096 if infer_device() == "xpu" else 65536 // 2


def mse_forward(_input, target, reduction: str):
    """
    Forward pass for MSE loss.
    Matches PyTorch semantics:
      - "none": elementwise squared error, same shape as input
      - "mean": scalar mean
      - "sum": scalar sum
    """
    assert (
        _input.shape == target.shape
    ), f"Shape mismatch: {_input.shape} vs {target.shape}"

    original_shape = _input.shape
    _input_flat = (
        _input.contiguous().view(-1, _input.shape[-1])
        if _input.ndim > 1
        else _input.contiguous().view(1, -1)
    )
    target_flat = (
        target.contiguous().view(-1, target.shape[-1])
        if target.ndim > 1
        else target.contiguous().view(1, -1)
    )

    n_rows, n_cols = _input_flat.shape
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))

    if reduction == "none":
        # allocate full elementwise loss
        loss_tensor = torch.zeros_like(
            _input_flat, dtype=_input.dtype, device=_input.device
        )
    else:
        # allocate per-sample scalar loss
        loss_tensor = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)

    liger_mse_kernel[(n_rows,)](
        X_ptr=_input_flat,
        X_stride=_input_flat.stride(-2),
        Y_ptr=target_flat,
        Y_stride=target_flat.stride(-2),
        loss_ptr=loss_tensor,
        loss_stride=(
            loss_tensor.stride(-2) if loss_tensor.ndim > 1 else loss_tensor.stride(-1)
        ),
        n_elements=n_cols,
        reduction=reduction,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32 if not is_hip() else 16,
    )

    if reduction == "none":
        loss = loss_tensor.view(original_shape)
    elif reduction == "mean":
        loss = torch.mean(loss_tensor)
    elif reduction == "sum":
        loss = torch.sum(loss_tensor)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    return loss


def mse_backward(_input, target, grad_output, reduction: str):
    """
    Backward pass for MSE loss.

    Parameters:
    _input (tensor): Input tensor
    target (tensor): Target tensor
    grad_output (tensor): Gradient of loss with respect to output
    reduction (str): Reduction type

    Returns:
    tensor: Gradient with respect to input
    """
    original_shape = _input.shape
    _input_flat = (
        _input.contiguous().view(-1, _input.shape[-1])
        if _input.ndim > 1
        else _input.contiguous().view(1, -1)
    )
    target_flat = (
        target.contiguous().view(-1, target.shape[-1])
        if target.ndim > 1
        else target.contiguous().view(1, -1)
    )

    n_rows, n_cols = _input_flat.shape
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))

    # Allocate gradient tensor
    grad_input = torch.zeros_like(_input_flat, dtype=_input.dtype, device=_input.device)

    liger_mse_grad_kernel[(n_rows,)](
        X_ptr=_input_flat,
        X_stride=_input_flat.stride(-2),
        Y_ptr=target_flat,
        Y_stride=target_flat.stride(-2),
        grad_ptr=grad_input,
        grad_stride=grad_input.stride(-2),
        n_elements=n_cols,
        reduction=reduction,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32 if not is_hip() else 16,
    )

    if reduction == "mean":
        grad_input = grad_input / n_rows

    # Apply chain rule with grad_output
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # If MSE is the last layer, grad_output is 1.0. Skip the mul to save time
        pass
    elif grad_output.ndim == 0:
        # If reduction is ['mean', 'sum'], grad_output is just a scalar
        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )
    else:
        # If reduction is 'none', grad_output has same shape as grad_input
        elementwise_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            grad_output.stride(-2) if grad_output.ndim > 1 else grad_output.stride(-1),
            grad_input,
            grad_input.stride(-2),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

    grad_input = grad_input.view(original_shape)
    return grad_input


class LigerMSEFunction(torch.autograd.Function):
    """
    This class implements a custom autograd function for the Liger MSE loss.
    It overrides the forward and backward methods of the torch.autograd.Function class.
    """

    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ):
        """
        The forward pass of the Liger MSE loss.

        Parameters:
        ctx : The context object.
        _input (tensor): The input tensor.
        target (tensor): The target tensor of same shape as input.
        reduction (str): The reduction to apply to the output: "none" | "mean" | "sum".

        Returns:
        tensor: The computed MSE loss.
        """
        loss = mse_forward(_input, target, reduction)

        # Detach the input tensor to avoid memory issues
        ctx.save_for_backward(_input.detach(), target.detach())
        ctx.reduction = reduction

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass of the Liger MSE loss.

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The tensor containing the gradient of the loss with respect to the output.

        Returns:
        tuple: A tuple with the gradients with respect to the inputs. The elements are tensors or None.
        """
        (_input, target) = ctx.saved_tensors
        _input = mse_backward(_input, target, grad_output, ctx.reduction)

        return (
            _input,
            None,  # target gradient (not needed)
            None,  # reduction gradient (not needed)
        )


def liger_mse_loss(
    _input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Applies the Liger MSE loss function.

    Parameters:
    _input (tensor): Input tensor.
    target (tensor): Target tensor of same shape as input.
    reduction (str): Specifies the reduction to apply to the output:
                    'none' | 'mean' | 'sum'. Default: 'mean'

    Returns:
    tensor: MSE loss.
    """
    return LigerMSEFunction.apply(_input, target, reduction)
