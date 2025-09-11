# src/models/__init__.py
"""
Model Architectures Module

This module contains all model architectures organized by methodology:

diffusion/  - Diffusion models (DDPM, DDIM, noise schedulers, etc.)
spectral/   - Spectral U-Net models for frequency recovery tasks

Each submodule contains models specific to their domain while sharing
common architectural components and utilities.
"""
from __future__ import annotations

# Import diffusion models
from .diffusion import *

# Import spectral models  
from .spectral import *

__all__ = [
    # Diffusion models
    "PML_SpectralDDPM",
    "PML_DDPM",
    
    # Spectral models
    "PML_SpectralUNet", 
    "PML_create_spectral_unet",
    
    # Shared components
    "PML_UNet",
]
