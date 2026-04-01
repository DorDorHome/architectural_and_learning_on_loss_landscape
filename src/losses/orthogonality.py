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

class KernelSORegularizer(nn.Module):
    def __init__(
        self,
        main_loss_func: callable,
        model: nn.Module,
        lambda_orth: float = 1e-4,
        normalization_mode: str = "naive mse sum correction"
    ):
        super(KernelSORegularizer, self).__init__()
        self.main_loss_func = main_loss_func
        self.lambda_orth = lambda_orth
        self.normalization_mode = normalization_mode
        
        self.conv_layers = []
        self.linear_layers = []
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.conv_layers.append(module)
            elif isinstance(module, nn.Linear):
                self.linear_layers.append(module)
                
        if not self.conv_layers and not self.linear_layers:
            print("Warning: No Conv2d or Linear layers found in the model for Kernel SO Regularization.")
            
    def forward(self, output, target):
        self.last_task_loss = self.main_loss_func(output, target)
        reg_loss = 0.0
        
        # Helper function
        def compute_P(G, I, D):
            if self.normalization_mode == "naive mse sum correction":
                return F.mse_loss(G, I, reduction='mean')
            elif self.normalization_mode == "correct by input size":
                return F.mse_loss(G, I, reduction='sum') / D
            elif self.normalization_mode == "no correction":
                return F.mse_loss(G, I, reduction='sum')
            else:
                raise ValueError(f"Unknown normalization mode: {self.normalization_mode}")
        
        # Conv2d Logic
        for module in self.conv_layers:
            W = module.weight
            M = W.shape[0]
            N = W.shape[1] * W.shape[2] * W.shape[3]
            W_reshaped = W.view(M, N)
            
            if M < N:
                G = torch.matmul(W_reshaped, W_reshaped.t())
                I = torch.eye(M, device=W.device)
                reg_loss += compute_P(G, I, M)
            else:
                G = torch.matmul(W_reshaped.t(), W_reshaped)
                I = torch.eye(N, device=W.device)
                reg_loss += compute_P(G, I, N)
                
        # Linear Logic
        for module in self.linear_layers:
            W = module.weight
            M, N = W.shape
            
            if M < N:
                G = torch.matmul(W, W.t())
                I = torch.eye(M, device=W.device)
                reg_loss += compute_P(G, I, M)
            else:
                G = torch.matmul(W.t(), W)
                I = torch.eye(N, device=W.device)
                reg_loss += compute_P(G, I, N)
                
        self.last_reg_loss = reg_loss * self.lambda_orth
        return self.last_task_loss + self.last_reg_loss
