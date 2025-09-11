# src/losses/__init__.py
"""
Loss Functions Module

This module contains all loss functions organized by their application domain:

- Base loss functions for general seismic processing
- Spectral loss functions for frequency recovery tasks
- Diffusion-specific loss functions
- Custom loss utilities and helpers
"""
from __future__ import annotations

from .losses import *
from .losses_spectral import *

__all__ = [
    # Base losses
    "PML_spectral_l2_per_band",
    "PML_lowband_weighted_loss",
    
    # Spectral recovery losses
    "PML_SeismicRecoveryLoss",
    "PML_SpectralMSELoss", 
    "PML_ReconstructionPenaltyLoss",
    "create_spectral_loss",
]
