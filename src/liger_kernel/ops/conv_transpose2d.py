from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd


# ------------------------------
# Triton forward kernel: Conv2D Transpose (NCHW)
# Weight layout: (C_in, C_out, K_h, K_w)
# ------------------------------

@triton.jit
def _conv2d_transpose_fwd_kernel(
    x_ptr,            # *const T,   [N, C_in, H_in, W_in]
    w_ptr,            # *const T,   [C_in, C_out, K_h, K_w]
    y_ptr,            # *mut   T,   [N, C_out, H_out, W_out]
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K_h, K_w,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    dilation_h: tl.constexpr, dilation_w: tl.constexpr,
    # strides
    sxn, sxc, sxh, sxw,
    swn, swc, swh, sww,
    syn, syc, syh, syw,
    # tiling
    BLOCK_CO: tl.constexpr, BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    pid_co = tl.program_id(0)  # tile over output channels
    pid_oh = tl.program_id(1)  # tile over H_out
    pid_n  = tl.program_id(2)  # tile over batch & W_out together via split grid later

    # Split pid_n into (n, ow_tile)
    tiles_w = tl.cdiv(W_out, BLOCK_OW)
    n = pid_n // tiles_w
    tile_ow = pid_n % tiles_w

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_oh = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)
    offs_ow = tile_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)

    # accumulators
    acc = tl.zeros((BLOCK_CO, BLOCK_OH, BLOCK_OW), dtype=tl.float32)

    for ci_base in range(0, C_in, BLOCK_CI):
        ci_idxs = ci_base + tl.arange(0, BLOCK_CI)
        ci_mask = ci_idxs < C_in

        for kh in tl.static_range(0, 8):
            if kh >= K_h:
                break
            for kw in tl.static_range(0, 8):
                if kw >= K_w:
                    break
                # Transpose relation
                ih_num = (offs_oh[:, None] + pad_h - kh * dilation_h)
                iw_num = (offs_ow[None, :] + pad_w - kw * dilation_w)
                div_h_ok = (ih_num % stride_h) == 0
                div_w_ok = (iw_num % stride_w) == 0
                ih = ih_num // stride_h
                iw = iw_num // stride_w
                in_bounds = (
                    (ih >= 0) & (ih < H_in) &
                    (iw >= 0) & (iw < W_in) &
                    div_h_ok & div_w_ok &
                    (offs_oh[:, None] < H_out) & (offs_ow[None, :] < W_out)
                )

                x_ptrs = (
                    x_ptr
                    + n * sxn
                    + (ci_idxs[:, None, None] * sxc)
                    + (ih[None, :, None] * sxh)
                    + (iw[None, None, :] * sxw)
                )
                x_vals = tl.load(x_ptrs, mask=ci_mask[:, None, None] & in_bounds[None, :, :], other=0.0)

                w_ptrs = (
                    w_ptr
                    + (ci_idxs[:, None] * swn)
                    + (offs_co[None, :] * swc)
                    + kh * swh + kw * sww
                )
                w_vals = tl.load(w_ptrs, mask=ci_mask[:, None] & (offs_co[None, :] < C_out), other=0.0)

                x_mat = tl.reshape(x_vals, (BLOCK_CI, BLOCK_OH * BLOCK_OW))
                prod = tl.dot(w_vals.T, x_mat)  # (BLOCK_CO, BLOCK_OH*BLOCK_OW)
                acc += tl.reshape(prod, (BLOCK_CO, BLOCK_OH, BLOCK_OW))

    y_ptrs = (
        y_ptr
        + n * syn
        + (offs_co[:, None, None] * syc)
        + (offs_oh[None, :, None] * syh)
        + (offs_ow[None, None, :] * syw)
    )
    mask = (
        (offs_co[:, None, None] < C_out)
        & (offs_oh[None, :, None] < H_out)
        & (offs_ow[None, None, :] < W_out)
    )
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask)


# ------------------------------
# Helper shapes
# ------------------------------

def _out_shape(N, C_out, H_in, W_in, kH, kW, sh, sw, ph, pw, dh=1, dw=1, ohp=0, owp=0):
    H_out = (H_in - 1) * sh - 2 * ph + dh * (kH - 1) + 1 + ohp
    W_out = (W_in - 1) * sw - 2 * pw + dw * (kW - 1) + 1 + owp
    return N, C_out, H_out, W_out


# ------------------------------
# Forward / Backward in the requested coding style
# ------------------------------

def conv2d_transpose_forward(
    input_tensor: torch.Tensor,           # (N, Cin, Hin, Win)
    weight: torch.Tensor,                 # (Cin, Cout, Kh, Kw)
    bias: Optional[torch.Tensor] = None,  # (Cout,)
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    output_padding: Tuple[int, int] = (0, 0),
):
    device = input_tensor.device
    dtype = input_tensor.dtype

    N, Cin, Hin, Win = input_tensor.shape
    Cin_w, Cout, Kh, Kw = weight.shape
    assert Cin_w == Cin, "Cin mismatch"

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    ohp, owp = output_padding

    if dh != 1 or dw != 1:
        raise NotImplementedError("dilation > 1 not supported in this Triton kernel yet")
    if ohp != 0 or owp != 0:
        raise NotImplementedError("output_padding != 0 not supported in this Triton kernel yet")

    N_, Cout_, Hout, Wout = _out_shape(N, Cout, Hin, Win, Kh, Kw, sh, sw, ph, pw, dh, dw, ohp, owp)
    assert N_ == N and Cout_ == Cout

    # Allocate grads (mirroring style; will be filled in backward)
    grad_input = torch.zeros_like(input_tensor)
    grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
    grad_bias = torch.zeros_like(bias) if (bias is not None and bias.requires_grad) else None

    # Output tensor
    output = torch.empty((N, Cout, Hout, Wout), dtype=dtype, device=device)

    # Launch Triton over tiles; keep it simple and stream over batch & width
    sxn, sxc, sxh, sxw = input_tensor.stride()
    swn, swc, swh, sww = weight.stride()
    syn, syc, syh, syw = output.stride()

    BLOCK_CO, BLOCK_OH, BLOCK_OW, BLOCK_CI = 64, 8, 16, 32
    grid = (
        triton.cdiv(Cout, BLOCK_CO),
        triton.cdiv(Hout, BLOCK_OH),
        N * triton.cdiv(Wout, BLOCK_OW),
    )

    _conv2d_transpose_fwd_kernel[grid](
        input_tensor, weight, output,
        N, Cin, Hin, Win,
        Cout, Hout, Wout,
        Kh, Kw,
        stride_h=sh, stride_w=sw,
        pad_h=ph, pad_w=pw,
        dilation_h=dh, dilation_w=dw,
        sxn=sxn, sxc=sxc, sxh=sxh, sxw=sxw,
        swn=swn, swc=swc, swh=swh, sww=sww,
        syn=syn, syc=syc, syh=syh, syw=syw,
        BLOCK_CO=BLOCK_CO, BLOCK_OH=BLOCK_OH, BLOCK_OW=BLOCK_OW,
        BLOCK_CI=BLOCK_CI,
        num_warps=4, num_stages=2,
    )

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    # Save input for backward in the same return style
    return output, grad_input, grad_weight, grad_bias, input_tensor, stride, padding, dilation, output_padding


def conv2d_transpose_backward(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    grad_input: torch.Tensor,
    grad_weight: Optional[torch.Tensor],
    grad_bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    output_padding: Tuple[int, int],
):
    # Use PyTorch's efficient gradient helpers for correctness and simplicity.
    # (You can replace with Triton kernels later if you need fully custom backward.)
    N, Cin, Hin, Win = input_tensor.shape

    # grad_input
    if grad_input is not None:
        grad_input.copy_(
            torch.nn.grad.conv_transpose2d_input(
                input_size=input_tensor.shape,
                weight=weight,
                grad_output=grad_output,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                output_padding=output_padding,
            )
        )

    # grad_weight
    if grad_weight is not None:
        grad_weight.copy_(
            torch.nn.grad.conv_transpose2d_weight(
                input=input_tensor,
                weight_size=weight.shape,
                grad_output=grad_output,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                output_padding=output_padding,
            )
        )

    # grad_bias
    if grad_bias is not None:
        grad_bias.add_(grad_output.sum(dim=(0, 2, 3)))

    return grad_input, grad_weight, grad_bias


class LigerConvTranspose2dFunction(torch.autograd.Function):
    """
    Conv2D Transpose implementation

    Mirrors the coding pattern of the provided Linear example:
    - forward(): Triton tiled kernel with bias add; returns (output, grad buffers, saved tensors)
    - backward(): uses torch gradient helpers, returns grads matching inputs
    """

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        output_padding: Tuple[int, int] = (0, 0),
    ):
        output, grad_input, grad_weight, grad_bias, saved_input, stride, padding, dilation, output_padding = conv2d_transpose_forward(
            input_tensor, weight, bias, stride, padding, dilation, output_padding
        )

        # Save for backward
        ctx.save_for_backward(
            saved_input,
            weight,
            bias if bias is not None else torch.tensor([], device=input_tensor.device),
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if grad_bias is not None else None,
        )
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.output_padding = output_padding

        return output

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        (
            input_tensor,
            weight,
            bias_tensor,
            grad_input,
            grad_weight,
            grad_bias,
        ) = ctx.saved_tensors

        bias = None if bias_tensor.numel() == 0 else bias_tensor

        grad_input, grad_weight, grad_bias = conv2d_transpose_backward(
            grad_output,
            input_tensor,
            weight,
            bias,
            grad_input,
            grad_weight,
            grad_bias,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.output_padding,
        )

        # Return grads in the order of forward's inputs
        return grad_input, grad_weight, grad_bias, None, None, None, None


def conv2d_transpose(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    output_padding: Tuple[int, int] = (0, 0),
):
    """
    Memory-efficient Conv2D Transpose layer (deconvolution)

    Args:
        input_tensor (torch.Tensor): (N, Cin, Hin, Win)
        weight (torch.Tensor): (Cin, Cout, Kh, Kw)
        bias (Optional[torch.Tensor]): (Cout,)
        stride, padding, dilation, output_padding: same as torch.nn.functional.conv_transpose2d

    Returns:
        torch.Tensor: output tensor with shape (N, Cout, Hout, Wout)
    """
    return LigerConvTranspose2dFunction.apply(
        input_tensor, weight, bias, stride, padding, dilation, output_padding
    )

