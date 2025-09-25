import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import element_mul_kernel
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
    # set it as constexpr since reduction is always known at compile time
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes both MSE loss and the gradient of the input.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    loss_ptr: Pointer to tensor to store the loss.
    loss_stride (int): The stride of the loss tensor.
    n_elements (int): The number of elements in each sample.
    reduction (str): The string for the reduction to apply
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*D is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    # locate the start index
    X_ptr += program_id * X_stride
    Y_ptr += program_id * Y_stride
    loss_ptr += program_id * loss_stride

    # Compute MSE loss and gradients in a single pass
    loss_sum = 0.0

    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)

        # Load input and target values
        x_vals = tl.load(
            X_ptr + offsets,
            mask=offsets < n_elements,
            other=0.0,
        ).cast(tl.float32)

        y_vals = tl.load(
            Y_ptr + offsets,
            mask=offsets < n_elements,
            other=0.0,
        ).cast(tl.float32)

        # Compute difference
        diff = x_vals - y_vals

        # Compute squared difference for loss
        squared_diff = diff * diff
        loss_sum += tl.sum(tl.where(offsets < n_elements, squared_diff, 0.0))

        # Compute gradients: d/dx[(x-y)^2] = 2(x-y)
        grad = 2.0 * diff

        # Apply reduction scaling to gradients
        if reduction == "mean":
            grad = grad / n_elements
        # For "sum" reduction, no scaling needed
        # For "none" reduction, no scaling needed (handled per-sample)

        # Store gradients back to input tensor (in-place gradient storage)
        tl.store(X_ptr + offsets, grad, mask=offsets < n_elements)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written
    tl.debug_barrier()

    # Calculate final loss value
    if reduction == "mean":
        loss = loss_sum / n_elements
    else:  # "sum" or "none"
        loss = loss_sum

    tl.store(loss_ptr, loss)


# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
MAX_FUSED_SIZE = 4096 if infer_device() == "xpu" else 65536 // 2


def mse_forward(
    _input,
    target,
    reduction,
):
    """
    Forward pass for MSE loss computation.

    Parameters:
    _input (tensor): Input tensor of shape (B, D) or any shape
    target (tensor): Target tensor of same shape as input
    reduction (str): The reduction to apply: "none" | "mean" | "sum"

    Returns:
    tuple: (loss, _input_with_gradients)
    """
    assert _input.shape == target.shape, f"Input and target must have the same shape. Got input: {_input.shape}, target: {target.shape}"

    # Flatten tensors for processing
    original_shape = _input.shape
    _input_flat = _input.view(-1, _input.shape[-1]
                              ) if _input.ndim > 1 else _input.view(1, -1)
    target_flat = target.view(-1, target.shape[-1]
                              ) if target.ndim > 1 else target.view(1, -1)

    n_rows, n_cols = _input_flat.shape
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))

    # unreduced loss per sample
    loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)

    # ensure _input and target are contiguous
    if _input_flat.stride(-1) != 1:
        _input_flat = _input_flat.contiguous()
    if target_flat.stride(-1) != 1:
        target_flat = target_flat.contiguous()

    # Launch kernel
    liger_mse_kernel[(n_rows,)](
        X_ptr=_input_flat,
        X_stride=_input_flat.stride(
            -2) if _input_flat.ndim > 1 else _input_flat.stride(-1),
        Y_ptr=target_flat,
        Y_stride=target_flat.stride(
            -2) if target_flat.ndim > 1 else target_flat.stride(-1),
        loss_ptr=loss_1d,
        loss_stride=loss_1d.stride(-1),
        n_elements=n_cols,
        reduction=reduction,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32 if not is_hip() else 16,
    )

    # Reshape input back to original shape
    _input_flat = _input_flat.view(original_shape)

    if reduction == "none":
        loss = loss_1d.view(
            original_shape[:-1]) if len(original_shape) > 1 else loss_1d
    else:
        loss = torch.sum(loss_1d)

    return loss, _input_flat


def mse_backward(_input, grad_output):
    """
    Backward pass for MSE loss.

    Parameters:
    _input (tensor): Input tensor with computed gradients stored in-place
    grad_output (tensor): Gradient of loss with respect to output

    Returns:
    tensor: Gradient with respect to input
    """
    # If MSE is the last layer, grad_output is 1.0. Skip the mul to save time
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        pass
    # If reduction is 'none', grad_output has same shape as loss
    elif grad_output.ndim > 0:
        if grad_output.shape != _input.shape:
            # Broadcast grad_output to match input shape
            grad_shape = grad_output.shape + \
                (1,) * (_input.ndim - grad_output.ndim)
            grad_output = grad_output.view(grad_shape).expand(_input.shape)
        _input = _input * grad_output
    # If reduction is ['mean', 'sum'], grad_output is just a scalar
    else:
        # Flatten for element_mul_kernel
        _input_flat = _input.view(-1, _input.shape[-1]
                                  ) if _input.ndim > 1 else _input.view(1, -1)
        n_rows, n_cols = _input_flat.shape
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))

        element_mul_kernel[(n_rows,)](
            _input_flat,
            _input_flat.stride(-2) if _input_flat.ndim > 1 else _input_flat.stride(-1),
            grad_output,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        _input = _input_flat.view(_input.shape)

    return _input


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
        loss, _input = mse_forward(_input, target, reduction)

        # Detach the input tensor to avoid memory issues
        ctx.save_for_backward(_input.detach())

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
        (_input,) = ctx.saved_tensors
        _input = mse_backward(_input, grad_output)

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
