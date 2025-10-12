from typing import Optional

import torch
import triton

from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd


def linear_forward(
    input_tensor,
    weight,
    bias=None,
):
    device = input_tensor.device
    dtype = input_tensor.dtype

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    BT, H = input_tensor.shape
    V = weight.shape[0]

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    # (BT + inc_factor - 1) // inc_factor
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    # (BT + chunk_size - 1) // chunk_size
    num_chunks = triton.cdiv(BT, chunk_size)

    # Initialize gradients if needed
    grad_weight = torch.zeros_like(
        weight, device=device) if weight.requires_grad else None
    grad_bias = torch.zeros_like(
        bias, device=device) if bias is not None and bias.requires_grad else None
    grad_input = torch.zeros_like(input_tensor)

    # Initialize output tensor
    output = torch.zeros((BT, V), dtype=dtype, device=device)

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)

        # chunk input, shape: chunk_size x H
        input_chunk = input_tensor[start_idx:end_idx]

        # shape: chunk_size x V
        output_chunk = input_chunk @ weight.t()
        if bias is not None:
            output_chunk = output_chunk + bias

        # Store output chunk
        output[start_idx:end_idx] = output_chunk

        # For backward pass, we need to compute gradients
        # This assumes we're in training mode and need gradients
        if weight.requires_grad:
            # grad_weight += input_chunk.t() @ grad_output_chunk
            # We'll store the input chunks for backward pass
            pass

    return output, grad_input, grad_weight, grad_bias, input_tensor


def linear_backward(grad_output, input_tensor, weight, bias, grad_input, grad_weight, grad_bias):
    device = input_tensor.device
    dtype = input_tensor.dtype

    BT, H = input_tensor.shape
    V = weight.shape[0]

    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)

    # Process chunks for backward pass
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)

        # Get chunks
        input_chunk = input_tensor[start_idx:end_idx]
        grad_output_chunk = grad_output[start_idx:end_idx]

        # Gradient w.r.t input: grad_input = grad_output @ weight
        grad_input_chunk = grad_output_chunk @ weight
        grad_input[start_idx:end_idx] = grad_input_chunk

        # Gradient w.r.t weight: grad_weight += input.t() @ grad_output
        if grad_weight is not None:
            grad_weight.add_(grad_output_chunk.t() @ input_chunk)

        # Gradient w.r.t bias: grad_bias += sum(grad_output, dim=0)
        if grad_bias is not None:
            grad_bias.add_(grad_output_chunk.sum(dim=0))

    return grad_input, grad_weight, grad_bias


class LigerLinearFunction(torch.autograd.Function):
    """
    Linear layer implementation

    Handle the forward and backward pass of a linear layer with memory-efficient chunking
    to avoid materializing large intermediate tensors.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_tensor (torch.Tensor): input tensor with shape (B*T, H), where B is batch size, 
                                       T is sequence length, H is hidden dimension.
            weight (torch.Tensor): weight matrix with shape (V, H), where V is output dimension
            bias (Optional[torch.Tensor]): bias vector with shape (V,). Default: None

        Returns:
            output (torch.Tensor): output tensor with shape (B*T, V)
        """

        output, grad_input, grad_weight, grad_bias, saved_input = linear_forward(
            input_tensor,
            weight,
            bias,
        )

        # Save tensors for backward pass
        ctx.save_for_backward(
            saved_input,
            weight,
            bias,
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if grad_bias is not None else None,
        )

        return output

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        (
            input_tensor,
            weight,
            bias,
            grad_input,
            grad_weight,
            grad_bias,
        ) = ctx.saved_tensors

        grad_input, grad_weight, grad_bias = linear_backward(
            grad_output, input_tensor, weight, bias, grad_input, grad_weight, grad_bias
        )

        return grad_input, grad_weight, grad_bias


def linear(input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
    """
    Memory-efficient linear layer

    Args:
        input_tensor (torch.Tensor): input tensor with shape (B*T, H)
        weight (torch.Tensor): weight matrix with shape (V, H) 
        bias (Optional[torch.Tensor]): bias vector with shape (V,)

    Returns:
        torch.Tensor: output tensor with shape (B*T, V)
    """
    return LigerLinearFunction.apply(input_tensor, weight, bias)
