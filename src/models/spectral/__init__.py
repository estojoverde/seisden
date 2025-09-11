# src/models/spectral/__init__.py
"""
Spectral Models Module

This module contains models specifically designed for seismic frequency recovery
and spectral processing tasks:

- SpectralUNet: U-Net architecture for frequency recovery
- Spectral-specific architectures and components
- Factory functions for model creation
"""
from __future__ import annotations

from .spectral_unet import *

__all__ = [
    # Main spectral model
    "PML_SpectralUNet",
    "PML_create_spectral_unet",
]
