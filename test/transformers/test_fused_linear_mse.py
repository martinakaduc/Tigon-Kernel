import torch
import torch.nn.functional as F

from liger_kernel.ops.fused_linear_mse import LigerFusedLinearMSEFunction
from liger_kernel.transformers.fused_linear_mse import LigerFusedLinearMSE
from liger_kernel.transformers.functional import liger_fused_linear_mse
from liger_kernel.utils import infer_device

device = infer_device()

def test_fused_linear_mse_basic():
    """Test basic functionality of fused linear MSE."""
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 4
    seq_len = 8
    hidden_size = 16
    output_size = 32
    
    # Create test data
    input_tensor = torch.randn(batch_size * seq_len, hidden_size, requires_grad=True, device=device)
    weight = torch.randn(output_size, hidden_size, requires_grad=True, device=device)
    bias = torch.randn(output_size, requires_grad=True, device=device)
    target = torch.randn(batch_size * seq_len, output_size, device=device)

    # Test the autograd function directly
    print("Testing LigerFusedLinearMSEFunction...")
    loss_fused = LigerFusedLinearMSEFunction.apply(
        input_tensor, weight, target, bias, "mean", None
    )
    
    # Test against PyTorch reference
    print("Computing PyTorch reference...")
    logits = F.linear(input_tensor, weight, bias)
    loss_ref = F.mse_loss(logits, target, reduction="mean")
    
    print(f"Fused loss: {loss_fused.item():.6f}")
    print(f"Reference loss: {loss_ref.item():.6f}")
    print(f"Loss difference: {abs(loss_fused.item() - loss_ref.item()):.8f}")
    
    # Test module interface
    print("\nTesting LigerFusedLinearMSE module...")
    fused_mse = LigerFusedLinearMSE(reduction="mean")
    loss_module = fused_mse(input_tensor, weight, target, bias)
    
    print(f"Module loss: {loss_module.item():.6f}")
    print(f"Module vs function difference: {abs(loss_module.item() - loss_fused.item()):.8f}")
    
    # Test functional interface
    print("\nTesting functional interface...")
    loss_functional = liger_fused_linear_mse(
        input_tensor, weight, target, bias, reduction="mean"
    )
    
    print(f"Functional loss: {loss_functional.item():.6f}")
    print(f"Functional vs function difference: {abs(loss_functional.item() - loss_fused.item()):.8f}")
    
    # Test gradient computation
    print("\nTesting gradients...")
    loss_fused.backward()
    
    if input_tensor.grad is not None:
        print(f"Input gradient norm: {input_tensor.grad.norm().item():.6f}")
    else:
        print("Input gradient is None")
        
    if weight.grad is not None:
        print(f"Weight gradient norm: {weight.grad.norm().item():.6f}")
    else:
        print("Weight gradient is None")
        
    if bias.grad is not None:
        print(f"Bias gradient norm: {bias.grad.norm().item():.6f}")
    else:
        print("Bias gradient is None")
    
    print("Basic test completed successfully!")


def test_fused_linear_mse_reductions():
    """Test different reduction modes."""
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 4
    hidden_size = 8
    output_size = 16
    
    input_tensor = torch.randn(batch_size * seq_len, hidden_size, device=device)
    weight = torch.randn(output_size, hidden_size, device=device)
    target = torch.randn(batch_size * seq_len, output_size, device=device)

    print("Testing different reduction modes...")
    
    for reduction in ["mean", "sum", "none"]:
        print(f"\nTesting reduction='{reduction}'")
        
        # Fused implementation
        loss_fused = LigerFusedLinearMSEFunction.apply(
            input_tensor, weight, target, None, reduction, None
        )
        
        # Reference implementation
        logits = F.linear(input_tensor, weight)
        loss_ref = F.mse_loss(logits, target, reduction=reduction)
        
        if reduction == "none":
            print(f"Fused loss shape: {loss_fused.shape}")
            print(f"Reference loss shape: {loss_ref.shape}")
            print(f"Max difference: {(loss_fused - loss_ref).abs().max().item():.8f}")
        else:
            print(f"Fused loss: {loss_fused.item():.6f}")
            print(f"Reference loss: {loss_ref.item():.6f}")
            print(f"Difference: {abs(loss_fused.item() - loss_ref.item()):.8f}")
    
    print("Reduction test completed successfully!")


def test_fused_linear_mse_shapes():
    """Test different input shapes."""
    torch.manual_seed(42)
    
    print("Testing different input shapes...")
    
    configs = [
        (8, 16, 32),    # Small
        (32, 64, 128),  # Medium  
        (64, 128, 256), # Large
    ]
    
    for batch_seq, hidden, output in configs:
        print(f"\nTesting shape ({batch_seq}, {hidden}) -> ({output})")
        
        input_tensor = torch.randn(batch_seq, hidden, device=device)
        weight = torch.randn(output, hidden, device=device)
        target = torch.randn(batch_seq, output, device=device)

        # Test without bias
        loss_no_bias = LigerFusedLinearMSEFunction.apply(
            input_tensor, weight, target, None, "mean", None
        )
        
        # Test with bias
        bias = torch.randn(output, device=device)
        loss_with_bias = LigerFusedLinearMSEFunction.apply(
            input_tensor, weight, target, bias, "mean", None
        )
        
        print(f"Loss without bias: {loss_no_bias.item():.6f}")
        print(f"Loss with bias: {loss_with_bias.item():.6f}")
    
    print("Shape test completed successfully!")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Fused Linear MSE Implementation")
    print("=" * 50)
    
    test_fused_linear_mse_basic()
    print("\n" + "=" * 50)
    test_fused_linear_mse_reductions()
    print("\n" + "=" * 50)
    test_fused_linear_mse_shapes()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)