# src/__init__.py
"""
Seismic Processing and Machine Learning Framework

This package provides a comprehensive framework for seismic data processing
using machine learning techniques, including:

- Diffusion models for seismic data generation and denoising
- Spectral U-Net models for frequency recovery
- Core training and evaluation infrastructure
- Visualization and analysis tools

The package is organized into the following modules:

core/           - Core utilities, logging, and base infrastructure
models/         - Model architectures separated by methodology
  diffusion/    - Diffusion model components (DDPM, schedulers, etc.)
  spectral/     - Spectral U-Net models for frequency recovery
data/           - Dataset classes and data loading utilities
training/       - Training loops, callbacks, and experiment management
losses/         - Loss functions for different model types
features/       - Feature processing and signal analysis tools
visualization/  - Plotting and visualization utilities
"""
from __future__ import annotations

# Core infrastructure
from .core import *

# Model architectures
from .models import *

# Data handling
from .data import *

# Training infrastructure
from .training import *

# Loss functions
from .losses import *

# Feature processing
from .features import *

# Visualization
from .visualization import *

__version__ = "1.0.0"
__author__ = "BaseTrainer/seisden Team"

__all__ = [
    # Core exports will be added by submodules
]
