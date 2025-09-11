# src/losses_spectral.py
"""
Specialized loss functions for seismic frequency recovery tasks.

This module provides loss functions specifically designed for seismic frequency recovery,
including spectral domain penalties and reconstruction-based losses.
"""
from __future__ import annotations

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.utils import PML_kw

__all__ = [
    "PML_SeismicRecoveryLoss",
    "PML_SpectralMSELoss", 
    "PML_ReconstructionPenaltyLoss",
    "create_spectral_loss",
]


class PML_SeismicRecoveryLoss(nn.Module):
    """
    Comprehensive loss function for seismic frequency recovery.
    
    This loss combines multiple components:
    1. Primary loss: MSE between predicted and target low-pass data
    2. Reconstruction penalty: Error when combining predicted LP with input HP
    3. FFT magnitude penalty: Spectral magnitude consistency
    4. FFT phase penalty: Spectral phase consistency
    
    The loss is designed to ensure both time-domain accuracy and spectral consistency
    in the frequency recovery task.
    
    Args:
        penalty_weight: Weight for reconstruction penalty term
        magnitude_weight: Weight for FFT magnitude penalty
        phase_weight: Weight for FFT phase penalty
        primary_loss: Type of primary loss ('mse', 'l1', 'huber')
    """
    
    def __init__(
        self,
        penalty_weight: float = 1.0,
        magnitude_weight: float = 0.5,
        phase_weight: float = 0.3,
        primary_loss: str = "mse"
    ):
        super().__init__()
        
        self.penalty_weight = penalty_weight
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight
        
        # Primary loss function
        if primary_loss == "mse":
            self.primary_loss_fn = nn.MSELoss()
        elif primary_loss == "l1":
            self.primary_loss_fn = nn.L1Loss()
        elif primary_loss == "huber":
            self.primary_loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported primary_loss: {primary_loss}")
    
    def forward(
        self,
        predicted_lp: torch.Tensor,
        target_lp: torch.Tensor,
        input_hp: torch.Tensor,
        target_full: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the complete loss.
        
        Args:
            predicted_lp: Predicted low-pass data (B, C, H, W)
            target_lp: Ground truth low-pass data (B, C, H, W)
            input_hp: Input high-pass data (B, C, H, W)
            target_full: Ground truth full spectrum data (B, C, H, W)
            
        Returns:
            Tuple of (total_loss, primary_loss, penalty_loss, magnitude_loss, phase_loss)
        """
        # Primary loss: accuracy of low-pass prediction
        primary_loss = self.primary_loss_fn(predicted_lp, target_lp)
        
        # Reconstruction penalty: how well HP + predicted_LP matches full spectrum
        reconstructed = predicted_lp + input_hp
        reconstruction_loss = torch.mean(torch.abs(reconstructed - target_full))
        
        # FFT-based spectral losses
        magnitude_loss, phase_loss = self._compute_spectral_losses(
            predicted_lp, target_lp, target_full
        )
        
        # Combine all loss components
        total_loss = (
            primary_loss +
            self.penalty_weight * reconstruction_loss +
            self.magnitude_weight * magnitude_loss +
            self.phase_weight * phase_loss
        )
        
        return total_loss, primary_loss, reconstruction_loss, magnitude_loss, phase_loss
    
    def _compute_spectral_losses(
        self,
        predicted_lp: torch.Tensor,
        target_lp: torch.Tensor,
        target_full: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute FFT-based magnitude and phase losses."""
        # Compute 2D FFT (remove channel dimension for FFT)
        pred_squeezed = predicted_lp.squeeze(1)  # (B, H, W)
        target_squeezed = target_lp.squeeze(1)   # (B, H, W)
        full_squeezed = target_full.squeeze(1)   # (B, H, W)
        
        # Real FFT for efficiency (assuming real-valued seismic data)
        predicted_fft = torch.fft.rfft2(pred_squeezed, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target_squeezed, dim=(-2, -1))
        full_fft = torch.fft.rfft2(full_squeezed, dim=(-2, -1))
        
        # Magnitude loss: compare predicted LP magnitude to target LP magnitude
        predicted_magnitude = torch.abs(predicted_fft)
        target_magnitude = torch.abs(target_fft)
        magnitude_loss = F.mse_loss(predicted_magnitude, target_magnitude)
        
        # Phase loss: compare predicted LP phase to full spectrum phase
        # Use full spectrum phase as reference for better guidance
        predicted_phase = torch.angle(predicted_fft)
        target_phase = torch.angle(full_fft)
        
        # Handle phase wrapping using cosine distance
        phase_diff = predicted_phase - target_phase
        phase_loss = 1.0 - torch.mean(torch.cos(phase_diff))
        
        return magnitude_loss, phase_loss


class PML_SpectralMSELoss(nn.Module):
    """
    Pure spectral domain MSE loss.
    
    Computes MSE between FFT magnitudes of predicted and target data.
    Useful as a component loss or for ablation studies.
    """
    
    def __init__(self, use_magnitude: bool = True, use_phase: bool = False):
        super().__init__()
        self.use_magnitude = use_magnitude
        self.use_phase = use_phase
        
        if not (use_magnitude or use_phase):
            raise ValueError("At least one of use_magnitude or use_phase must be True")
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral MSE loss.
        
        Args:
            predicted: Predicted data (B, C, H, W)
            target: Target data (B, C, H, W)
            
        Returns:
            Spectral MSE loss
        """
        # Remove channel dimension for FFT
        pred_squeezed = predicted.squeeze(1)  # (B, H, W)
        target_squeezed = target.squeeze(1)   # (B, H, W)
        
        # Compute FFT
        pred_fft = torch.fft.rfft2(pred_squeezed, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target_squeezed, dim=(-2, -1))
        
        loss = 0.0
        
        if self.use_magnitude:
            pred_mag = torch.abs(pred_fft)
            target_mag = torch.abs(target_fft)
            loss += F.mse_loss(pred_mag, target_mag)
        
        if self.use_phase:
            pred_phase = torch.angle(pred_fft)
            target_phase = torch.angle(target_fft)
            # Use cosine distance for phase
            phase_diff = pred_phase - target_phase
            loss += 1.0 - torch.mean(torch.cos(phase_diff))
        
        return loss


class PML_ReconstructionPenaltyLoss(nn.Module):
    """
    Reconstruction penalty loss.
    
    Measures how well the combination of high-pass input and predicted low-pass
    reconstructs the original full-spectrum data.
    
    Args:
        loss_type: Type of reconstruction loss ('l1', 'l2', 'huber')
    """
    
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    def forward(
        self,
        predicted_lp: torch.Tensor,
        input_hp: torch.Tensor,
        target_full: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction penalty.
        
        Args:
            predicted_lp: Predicted low-pass component (B, C, H, W)
            input_hp: High-pass input (B, C, H, W)
            target_full: Target full spectrum data (B, C, H, W)
            
        Returns:
            Reconstruction penalty loss
        """
        reconstructed = predicted_lp + input_hp
        return self.loss_fn(reconstructed, target_full)


# Factory function for creating common loss configurations
def create_spectral_loss(**kwargs):
    """
    Factory function for creating spectral recovery losses with common configurations.
    
    Optional **kwargs:
        loss_type (str) = "seismic_recovery": Type of loss to create
        penalty_weight (float) = 1.0: Weight for reconstruction penalty
        magnitude_weight (float) = 0.5: Weight for magnitude penalty  
        phase_weight (float) = 0.3: Weight for phase penalty
        primary_loss (str) = "mse": Primary loss type
        
    Returns:
        Configured loss function
    """
    loss_type = PML_kw("loss_type", kwargs, "seismic_recovery")
    
    if loss_type == "seismic_recovery":
        return PML_SeismicRecoveryLoss(
            penalty_weight=PML_kw("penalty_weight", kwargs, 1.0),
            magnitude_weight=PML_kw("magnitude_weight", kwargs, 0.5),
            phase_weight=PML_kw("phase_weight", kwargs, 0.3),
            primary_loss=PML_kw("primary_loss", kwargs, "mse")
        )
    elif loss_type == "spectral_mse":
        return PML_SpectralMSELoss(
            use_magnitude=PML_kw("use_magnitude", kwargs, True),
            use_phase=PML_kw("use_phase", kwargs, False)
        )
    elif loss_type == "reconstruction":
        return PML_ReconstructionPenaltyLoss(
            loss_type=PML_kw("reconstruction_loss_type", kwargs, "l1")
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
