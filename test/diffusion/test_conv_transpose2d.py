import torch

# If your operator is exposed as liger_kernel.ops.conv2d_transpose.conv2d_transpose
from liger_kernel.ops.conv2d_transpose import conv2d_transpose  # noqa: F401
# If you kept the filename conv2d_transpose_triton.py and exported conv2d_transpose there, you can use:
# from liger_kernel.ops.conv2d_transpose_triton import conv2d_transpose

def run_case(N, Cin, Cout, Hin, Win, Kh, Kw, stride, padding, dilation=(1, 1), output_padding=(0, 0), dtype=torch.float16):
    print(f"\n=== Forward Test Case ===")
    print(f"N={N}, Cin={Cin}, Cout={Cout}, H={Hin}, W={Win}, Kh={Kh}, Kw={Kw}, stride={stride}, padding={padding}, dilation={dilation}, output_padding={output_padding}")

    x = torch.randn(N, Cin, Hin, Win, device="cuda", dtype=dtype)
    w = torch.randn(Cin, Cout, Kh, Kw, device="cuda", dtype=dtype)
    b = torch.randn(Cout, device="cuda", dtype=dtype)

    y_triton = conv2d_transpose(x, w, b, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding)
    y_ref = torch.nn.functional.conv_transpose2d(x, w, b, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding)

    max_diff = (y_triton - y_ref).abs().max().item()
    ok = torch.allclose(y_triton, y_ref, atol=1e-3 if dtype==torch.float16 else 1e-4, rtol=1e-3)
    print("y_triton.shape =", tuple(y_triton.shape))
    print("y_ref.shape    =", tuple(y_ref.shape))
    print("max abs diff   =", max_diff)
    print("allclose?      =", ok)
    return ok


if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA required for Triton tests"

    # 1) Small sanity
    run_case(N=1, Cin=1, Cout=1, Hin=4, Win=4, Kh=3, Kw=3, stride=(1,1), padding=(0,0), dtype=torch.float16)
    run_case(N=1, Cin=1, Cout=1, Hin=5, Win=5, Kh=3, Kw=3, stride=(2,2), padding=(1,1), dtype=torch.float16)

    # 2) Multi-channel
    run_case(N=2, Cin=3, Cout=2, Hin=8, Win=8, Kh=3, Kw=3, stride=(2,2), padding=(1,1), dtype=torch.float16)
    run_case(N=1, Cin=2, Cout=4, Hin=7, Win=9, Kh=5, Kw=5, stride=(1,1), padding=(2,2), dtype=torch.float16)

    # 3) FP32 check
    run_case(N=1, Cin=3, Cout=3, Hin=6, Win=6, Kh=3, Kw=3, stride=(1,1), padding=(1,1), dtype=torch.float32)

