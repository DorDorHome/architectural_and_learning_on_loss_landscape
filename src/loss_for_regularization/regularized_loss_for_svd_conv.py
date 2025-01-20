import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.svd_decomposed_single_conv import SVD_Conv2d
from src.models.svd_decomposed_FC import SVD_Linear






class RegularizedLoss_SVD_conv(nn.Module):
    def __init__(
        self, 
        main_loss_func: callable, 
        model: nn.Module, 
        lambda_orth: float = 1e-4, 
        allow_svd_values_negative: bool = False
    ):
        """
        Combines the main loss function with orthogonality regularization.

        Parameters:
        - main_loss_func (callable): The primary loss function (e.g., F.cross_entropy).
        - model (nn.Module): The model containing SVD_Conv2d layers.
        - lambda_orth (float): Weighting factor for the orthogonality regularization.
        - allow_svd_values_negative (bool): Whether to allow negative singular values.
        """
        super(RegularizedLoss_SVD_conv, self).__init__()
        self.main_loss_func = main_loss_func
        self.lambda_orth = lambda_orth
        self.allow_svd_values_negative = allow_svd_values_negative
        
        # Pre-identify and cache all SVD_Conv2d layers
        self.svd_layers = []
        for module in model.modules():
            if isinstance(module, SVD_Conv2d):
                self.svd_layers.append(module)
            if isinstance(module, SVD_Linear):
                self.svd_layers.append(module)
        
        if not self.svd_layers:
            print("Warning: No SVD_Conv2d layers found in the model.")
    
    def forward(self, output, target):
        # Compute the primary loss
        primary_loss = self.main_loss_func(output, target)
        
        # Initialize the orthogonality loss
        orthogonality_loss = 0.0
        
        # Iterate over all cached SVD_Conv2d layers to compute orthogonality losses
        for module in self.svd_layers:
            # Retrieve N and C matrices
            N = module.N  # Shape: (out_channels, r) or (in_channels * K, r)
            C = module.C  # Shape: (r, in_channels * K * K) or (r, out_channels * K)
            
            # Handle Sigma scaling
            if not self.allow_svd_values_negative:
                Sigma = module.Sigma.abs()
            else:
                Sigma = module.Sigma
            
            # Optionally, incorporate Sigma into orthogonality (if Sigma affects scaling)
            # For standard orthogonality (N^T N ~ I and C C^T ~ I), Sigma does not directly influence
            # However, if desired, you can modulate the matrices based on Sigma
            # For now, we'll proceed without modifying N and C based on Sigma
            
            # Compute orthogonality for N: ||N^T N - I||_F^2
            orth_N = torch.matmul(N.t(), N)  # Shape: (r, r)
            identity = torch.eye(module.r, device=N.device)
            loss_N = F.mse_loss(orth_N, identity)
            
            # Compute orthogonality for C: ||C C^T - I||_F^2
            orth_C = torch.matmul(C, C.t())  # Shape: (r, r)
            loss_C = F.mse_loss(orth_C, identity)
            
            # Aggregate the losses
            orthogonality_loss += loss_N + loss_C
        
        # Combine the primary loss with the orthogonality loss
        total_loss = primary_loss + self.lambda_orth * orthogonality_loss
        return total_loss