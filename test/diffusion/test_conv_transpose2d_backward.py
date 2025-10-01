import torch

# If your operator is exposed as liger_kernel.ops.conv2d_transpose.conv2d_transpose
from liger_kernel.ops.conv2d_transpose import conv2d_transpose  # noqa: F401
# Or, if you kept the filename conv2d_transpose_triton.py:
# from liger_kernel.ops.conv2d_transpose_triton import conv2d_transpose


def run_case(N, Cin, Cout, Hin, Win, Kh, Kw, stride, padding, dilation=(1,1), output_padding=(0,0), dtype=torch.float16):
    print(f"\n=== Backward Test Case ===")
    print(f"N={N}, Cin={Cin}, Cout={Cout}, H={Hin}, W={Win}, Kh={Kh}, Kw={Kw}, stride={stride}, padding={padding}, dilation={dilation}, output_padding={output_padding}")

    # Reference autograd (PyTorch)
    x_ref = torch.randn(N, Cin, Hin, Win, device="cuda", dtype=dtype, requires_grad=False)
    w_ref = torch.randn(Cin, Cout, Kh, Kw, device="cuda", dtype=dtype, requires_grad=True)
    b_ref = torch.randn(Cout, device="cuda", dtype=dtype, requires_grad=True)

    y_ref = torch.nn.functional.conv_transpose2d(x_ref, w_ref, b_ref, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding)
    y_ref.sum().backward()
    dw_ref = w_ref.grad.detach().clone()
    db_ref = b_ref.grad.detach().clone()
    dx_ref = torch.autograd.grad(outputs=[y_ref], inputs=[x_ref], grad_outputs=[torch.ones_like(y_ref)], retain_graph=False, allow_unused=True)[0]

    # Triton op under autograd (our Function provides backward via torch.grad helpers)
    x = x_ref.detach().clone().requires_grad_(False)
    w = w_ref.detach().clone().requires_grad_(True)
    b = b_ref.detach().clone().requires_grad_(True)

    y = conv2d_transpose(x, w, b, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding)
    loss = y.sum()
    loss.backward()

    dw = w.grad.detach().clone()
    db = b.grad.detach().clone()

    # Compare weights & bias grads
    max_diff_w = (dw - dw_ref).abs().max().item()
    max_diff_b = (db - db_ref).abs().max().item()
    ok_w = torch.allclose(dw, dw_ref, atol=1e-3 if dtype==torch.float16 else 1e-4, rtol=1e-3)
    ok_b = torch.allclose(db, db_ref, atol=1e-3 if dtype==torch.float16 else 1e-4, rtol=1e-3)

    print("dw.shape        =", tuple(dw.shape), "ref:", tuple(dw_ref.shape), "max diff:", max_diff_w, "ok:", ok_w)
    print("db.shape        =", tuple(db.shape), "ref:", tuple(db_ref.shape), "max diff:", max_diff_b, "ok:", ok_b)


if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA required for Triton tests"

    # 1) Small sanity
    run_case(N=1, Cin=1, Cout=1, Hin=4, Win=4, Kh=3, Kw=3, stride=(1,1), padding=(0,0), dtype=torch.float16)

    # 2) Common shapes
    run_case(N=2, Cin=3, Cout=2, Hin=8, Win=8, Kh=3, Kw=3, stride=(2,2), padding=(1,1), dtype=torch.float16)

    # 3) FP32 check
    run_case(N=1, Cin=3, Cout=3, Hin=6, Win=6, Kh=3, Kw=3, stride=(1,1), padding=(1,1), dtype=torch.float32)

