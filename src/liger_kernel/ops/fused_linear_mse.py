import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


@triton.jit
def liger_fused_linear_mse_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing MSE loss and gradients in a fused manner.

    This kernel computes both the MSE loss and the gradient of the input logits.
    The gradient computation is: d_loss/d_logits = 2 * (logits - target) / N
    where N is the normalization factor based on reduction type.

    Parameters:
    X_ptr: Pointer to input logits tensor (after linear layer)
    X_stride: Stride of the input tensor
    Y_ptr: Pointer to target tensor
    Y_stride: Stride of the target tensor
    loss_ptr: Pointer to loss output tensor
    loss_stride: Stride of the loss tensor
    n_cols: Number of columns (features) in the input
    reduction: Type of reduction ('mean', 'sum', 'none')
    BLOCK_SIZE: Block size for Triton operations
    """

    program_id = tl.program_id(0).to(tl.int64)

    # Locate the start index for this program
    X_ptr += program_id * X_stride
    Y_ptr += program_id * Y_stride
    loss_ptr += program_id * loss_stride

    loss_sum = 0.0

    # Process the tensors in blocks
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        # Load input logits and targets
        x_vals = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y_vals = tl.load(Y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Compute difference and squared difference
        diff = x_vals - y_vals
        squared_diff = diff * diff

        # Accumulate loss for reduction
        loss_sum += tl.sum(tl.where(mask, squared_diff, 0.0))

        # Compute gradient: d_loss/d_logits = 2 * (logits - target)
        # We'll apply the reduction factor later in the calling function
        grad = 2.0 * diff

        # Store gradient back to X_ptr (in-place gradient computation)
        tl.store(X_ptr + offsets, grad, mask=mask)

    # Store the loss
    if reduction == "mean":
        loss = loss_sum / n_cols
    elif reduction == "sum":
        loss = loss_sum
    else:  # reduction == "none"
        loss = loss_sum / n_cols  # Per-sample loss

    tl.store(loss_ptr, loss)


def fused_linear_mse_forward(
    _input,
    weight,
    target,
    bias=None,
    reduction="mean",
    accum_dtype=None,
):
    """
    Forward pass for fused linear + MSE loss.

    Args:
        _input: Input tensor of shape (BT, H) where BT is batch*seq_len, H is hidden size
        weight: Weight tensor of shape (V, H) where V is output size
        target: Target tensor of shape (BT, V)
        bias: Optional bias tensor of shape (V,)
        reduction: Reduction type ('mean', 'sum', 'none')
        accum_dtype: Optional dtype for gradient accumulation

    Returns:
        loss: Computed MSE loss
        grad_input: Gradient with respect to input
        grad_weight: Gradient with respect to weight (if weight.requires_grad)
        grad_bias: Gradient with respect to bias (if bias is not None)
    """
    device = _input.device

    # Get dimensions
    BT, H = _input.shape
    V = weight.shape[0]

    # Ensure target has correct shape
    if target.ndim == 1:
        # If target is 1D, assume it's indices and convert to one-hot
        target_indices = target
        target = torch.zeros(BT, V, dtype=_input.dtype, device=device)
        target.scatter_(1, target_indices.unsqueeze(1), 1.0)
    elif target.shape != (BT, V):
        raise ValueError(
            f"Target shape {target.shape} doesn't match expected shape ({BT}, {V})"
        )

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    # Compute memory chunking strategy similar to fused cross entropy
    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(
        triton.cdiv(BT, inc_factor)
    )  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_input = torch.zeros_like(_input, device=device)

    # Initialize gradient accumulators
    if accum_dtype is None:
        grad_weight = (
            torch.zeros_like(weight, device=device) if weight.requires_grad else None
        )
        grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    else:
        grad_weight = (
            torch.zeros_like(weight, dtype=accum_dtype, device=device)
            if weight.requires_grad
            else None
        )
        grad_bias = (
            torch.zeros_like(bias, dtype=accum_dtype, device=device)
            if bias is not None
            else None
        )

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

    # Process in chunks to manage memory
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        # Compute logits: matmul with original precision
        logits_chunk = _input_chunk @ weight.t()  # chunk_size x V
        if bias is not None:
            logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]  # chunk_size x V
        n_rows = logits_chunk.shape[0]

        # Prepare loss storage for this chunk
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,

        # Ensure tensors are contiguous
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # Run the fused kernel - computes both loss and gradients in-place
        liger_fused_linear_mse_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-2),
            loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            n_cols=V,
            reduction=reduction,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        # The kernel computed gradients in logits_chunk, now we need to:
        # 1. Apply reduction scaling to gradients
        # 2. Backpropagate through the linear layer

        grad_logits_chunk = logits_chunk  # chunk_size x V

        # Apply reduction scaling to gradients
        if reduction == "mean":
            # For mean reduction, divide by total number of elements
            total_elements = BT * V
            grad_logits_chunk = grad_logits_chunk / total_elements
        elif reduction == "sum":
            # For sum reduction, no additional scaling needed
            pass
        # For 'none' reduction, gradients are already per-sample

        # Backpropagate gradients through linear layer
        # grad_input = grad_logits @ weight
        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        # Accumulate weight gradients: grad_weight += input.T @ grad_logits
        if grad_weight is not None:
            grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk)

        # Accumulate bias gradients
        if bias is not None:
            torch.add(
                input=grad_bias,
                other=grad_logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    # Compute final loss
    if reduction == "none":
        loss = loss_1d
    elif reduction == "mean":
        loss = torch.mean(loss_1d)
    elif reduction == "sum":
        loss = torch.sum(loss_1d)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    # Cast gradients back to original dtype
    grad_weight = grad_weight.to(weight.dtype) if grad_weight is not None else None
    grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None

    return loss, grad_input, grad_weight, grad_bias


def fused_linear_mse_backward(grad_output, grad_input, grad_weight, grad_bias):
    """
    Backward pass for fused linear MSE loss.

    Args:
        grad_output: Gradient of loss with respect to output
        grad_input: Computed gradient with respect to input
        grad_weight: Computed gradient with respect to weight
        grad_bias: Computed gradient with respect to bias

    Returns:
        Scaled gradients
    """
    # If MSE is the last layer, grad_output is 1.0. Skip the mul to save time
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # Scale all gradients by grad_output
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        # Scale grad_input
        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        # Scale grad_weight
        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

        # Scale grad_bias
        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

    return grad_input, grad_weight, grad_bias


class LigerFusedLinearMSEFunction(torch.autograd.Function):
    """
    Autograd function for fused linear layer + MSE loss.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        reduction="mean",
        accum_dtype=None,
    ):
        """
        Forward pass of fused linear + MSE loss.

        Args:
            _input: Input tensor of shape (BT, H)
            weight: Weight tensor of shape (V, H)
            target: Target tensor of shape (BT, V) or (BT,) for indices
            bias: Optional bias tensor of shape (V,)
            reduction: Reduction type ('mean', 'sum', 'none')
            accum_dtype: Optional dtype for gradient accumulation

        Returns:
            loss: Computed MSE loss
        """
        loss, grad_input, grad_weight, grad_bias = fused_linear_mse_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            reduction=reduction,
            accum_dtype=accum_dtype,
        )

        # Store gradients for backward pass
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if bias is not None else None,
        )

        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        """
        Backward pass of fused linear + MSE loss.
        """
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_mse_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )

        return (
            grad_input,
            grad_weight,
            None,  # target
            grad_bias,
            None,  # reduction
            None,  # accum_dtype
        )
