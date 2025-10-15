from typing import Optional

import torch

from liger_kernel.ops.fused_linear_mse import LigerFusedLinearMSEFunction


class LigerFusedLinearMSE(torch.nn.Module):
    """
    Fused Linear layer with MSE loss.
    
    This module combines a linear transformation with MSE loss computation,
    avoiding the materialization of large intermediate logits tensors for memory efficiency.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
                        'none' | 'mean' | 'sum'. Default: 'mean'
        accum_dtype (torch.dtype): Optional dtype for gradient accumulation.
                                  Recommended for higher precision. Default: None
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        accum_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert reduction in ["mean", "sum", "none"], f"Unsupported reduction: {reduction}"
        self.reduction = reduction
        self.accum_dtype = accum_dtype
    
    def forward(
        self,
        _input: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of fused linear + MSE loss.
        
        Args:
            _input: Input tensor of shape (batch_size * seq_len, hidden_size)
            weight: Weight tensor of shape (output_size, hidden_size)
            target: Target tensor of shape (batch_size * seq_len, output_size) 
                   or (batch_size * seq_len,) for class indices
            bias: Optional bias tensor of shape (output_size,)
        
        Returns:
            torch.Tensor: MSE loss
        """
        return LigerFusedLinearMSEFunction.apply(
            _input,
            weight,
            target,
            bias,
            self.reduction,
            self.accum_dtype,
        )


class LigerFusedLinearMSELoss(torch.nn.Module):
    """
    Fused Linear layer with MSE loss (alias for LigerFusedLinearMSE).
    
    This is an alias for LigerFusedLinearMSE for consistency with other loss modules.
    """
    
    def __init__(
        self,
        reduction: str = "mean", 
        accum_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.fused_linear_mse = LigerFusedLinearMSE(
            reduction=reduction,
            accum_dtype=accum_dtype,
        )
    
    def forward(
        self,
        lin_weight: torch.Tensor,
        _input: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with weight as first argument for consistency.
        
        Args:
            lin_weight: Weight tensor of shape (output_size, hidden_size)
            _input: Input tensor of shape (batch_size * seq_len, hidden_size)
            target: Target tensor of shape (batch_size * seq_len, output_size)
            bias: Optional bias tensor of shape (output_size,)
        
        Returns:
            torch.Tensor: MSE loss
        """
        return self.fused_linear_mse(
            _input=_input,
            weight=lin_weight,
            target=target,
            bias=bias,
        )