import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from scipy.stats import norm
import numpy as np


class HLGaussLoss(nn.Module):
    """
    HL-Gauss loss for value function training based on:
    "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL"
    https://arxiv.org/abs/2403.03950
    
    Transforms scalar regression targets into categorical distributions using Gaussian smoothing.
    """
    
    def __init__(
        self,
        min_value: float = -10.0,
        max_value: float = 10.0,
        num_bins: int = 51,
        sigma_ratio: float = 0.75,
    ):
        """
        Args:
            min_value: Minimum value for the support of the categorical distribution
            max_value: Maximum value for the support of the categorical distribution
            num_bins: Number of bins for discretization
            sigma_ratio: Ratio of sigma to bin width (typically 0.75)
        """
        super().__init__()
        
        if num_bins < 3:
            raise ValueError(f"num_bins must be at least 3, got {num_bins}")
        if min_value >= max_value:
            raise ValueError(f"min_value ({min_value}) must be less than max_value ({max_value})")
        
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio
        
        # Calculate bin width
        self.bin_width = (max_value - min_value) / num_bins
        
        # Calculate sigma for Gaussian smoothing
        self.sigma = sigma_ratio * self.bin_width
        
        # Create bin centers
        self.register_buffer(
            'bin_centers',
            torch.linspace(min_value, max_value, num_bins)
        )
        
        # Precompute bin edges for integration
        bin_edges = torch.linspace(
            min_value - self.bin_width / 2,
            max_value + self.bin_width / 2,
            num_bins + 1
        )
        self.register_buffer('bin_edges', bin_edges)
    
    def _create_target_distribution(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Create smoothed categorical distribution from scalar targets.
        
        Args:
            targets: Scalar target values of shape (batch_size,) or (batch_size, seq_len)
            
        Returns:
            Categorical distributions of shape (batch_size, num_bins) or (batch_size, seq_len, num_bins)
        """
        # Ensure targets are within bounds
        targets = torch.clamp(targets, self.min_value, self.max_value)
        
        # Store original shape and flatten for processing
        original_shape = targets.shape
        targets_flat = targets.view(-1)
        batch_size = targets_flat.shape[0]
        
        # Expand dimensions for broadcasting
        targets_expanded = targets_flat.unsqueeze(1)  # (batch_size, 1)
        bin_edges = self.bin_edges.unsqueeze(0)  # (1, num_bins + 1)
        
        # Calculate CDF values at bin edges
        # Using the error function for numerical stability
        z_scores = (bin_edges - targets_expanded) / (self.sigma * np.sqrt(2))
        cdf_values = 0.5 * (1 + torch.erf(z_scores))
        
        # Calculate probabilities for each bin by taking differences
        probs = cdf_values[:, 1:] - cdf_values[:, :-1]  # (batch_size, num_bins)
        
        # Normalize to ensure probabilities sum to 1
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        
        # Reshape back to original shape with bins as last dimension
        if len(original_shape) == 2:
            probs = probs.view(original_shape[0], original_shape[1], self.num_bins)
        else:
            probs = probs.view(original_shape[0], self.num_bins)
        
        return probs
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute HL-Gauss loss.
        
        Args:
            predictions: Predicted logits of shape (batch_size, num_bins) or (batch_size, seq_len, num_bins)
            targets: Scalar target values of shape (batch_size,) or (batch_size, seq_len)
            mask: Optional mask of shape matching targets
            
        Returns:
            Loss value (scalar)
        """
        # Create target distributions
        target_distributions = self._create_target_distribution(targets)
        
        # Compute cross-entropy loss
        if len(predictions.shape) == 3:
            # Sequence data: (batch_size, seq_len, num_bins)
            log_probs = F.log_softmax(predictions, dim=-1)
            loss = -(target_distributions * log_probs).sum(dim=-1)
        else:
            # Single prediction: (batch_size, num_bins)
            log_probs = F.log_softmax(predictions, dim=-1)
            loss = -(target_distributions * log_probs).sum(dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            # Average over non-masked elements
            return loss.sum() / (mask.sum() + 1e-8)
        else:
            return loss.mean()
    
    def get_values_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert categorical logits back to scalar values using expectation.
        
        Args:
            logits: Logits of shape (batch_size, num_bins) or (batch_size, seq_len, num_bins)
            
        Returns:
            Scalar values of shape (batch_size,) or (batch_size, seq_len)
        """
        probs = F.softmax(logits, dim=-1)
        
        # Ensure bin_centers is on the same device as logits
        bin_centers = self.bin_centers.to(logits.device)
        
        if len(logits.shape) == 3:
            # Sequence data
            values = (probs * bin_centers.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        else:
            # Single prediction
            values = (probs * bin_centers.unsqueeze(0)).sum(dim=-1)
        
        return values


class ValueHeadWithHLGauss(nn.Module):
    """
    Value head using HL-Gauss categorical representation instead of scalar regression.
    """
    
    def __init__(
        self,
        hidden_size: int,
        min_value: float = -10.0,
        max_value: float = 10.0,
        num_bins: int = 51,
        sigma_ratio: float = 0.75,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_bins = num_bins
        
        # 3-layer neural network
        self.layer = nn.Linear(hidden_size, num_bins)
        
        # Initialize all layers
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)
        
        # HL-Gauss loss function
        self.hl_gauss_loss = HLGaussLoss(
            min_value=min_value,
            max_value=max_value,
            num_bins=num_bins,
            sigma_ratio=sigma_ratio,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both scalar values and logits.
        
        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            values: Scalar value predictions of shape (batch_size, seq_len)
            logits: Categorical logits of shape (batch_size, seq_len, num_bins)
        """
        
        logits = self.layer(hidden_states)  # (batch_size, seq_len, num_bins)
        
        values = self.hl_gauss_loss.get_values_from_logits(logits)
        return values, logits