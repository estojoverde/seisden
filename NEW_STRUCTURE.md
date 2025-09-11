# New Organized src/ Structure

## Overview

The `src/` directory has been reorganized into a cleaner, more modular structure that separates different functionalities and provides better organization for the growing codebase.

## New Directory Structure

```
src/
├── core/                      # Core Infrastructure
│   ├── __init__.py           # Core exports
│   ├── utils.py              # Utility functions (PML_kw, etc.)
│   ├── logging.py            # Logging infrastructure
│   ├── trainer.py            # Base model trainer
│   ├── metrics.py            # Evaluation metrics
│   └── callbacks.py          # Training callbacks
│
├── models/                    # Model Architectures
│   ├── __init__.py           # Model exports
│   ├── diffusion/            # Diffusion Models
│   │   ├── __init__.py       # Diffusion exports
│   │   ├── diffusion.py      # SpectralDDPM implementation
│   │   ├── unet_blocks.py    # UNet backbone for diffusion
│   │   ├── autoencoder.py    # Autoencoder components
│   │   ├── conditioning.py   # Conditioning mechanisms
│   │   └── utils.py          # Diffusion-specific utilities
│   └── spectral/             # Spectral Models
│       ├── __init__.py       # Spectral exports
│       └── spectral_unet.py  # SpectralUNet for frequency recovery
│
├── data/                      # Data Handling
│   ├── __init__.py           # Data exports
│   ├── dataset.py            # Base dataset classes
│   └── dataset_spectral.py   # Spectral frequency recovery datasets
│
├── training/                  # Training Infrastructure
│   ├── __init__.py           # Training exports
│   └── train_diffusion.py    # Diffusion training loops
│
├── losses/                    # Loss Functions
│   ├── __init__.py           # Loss exports
│   ├── losses.py             # Base loss functions
│   └── losses_spectral.py    # Spectral recovery losses
│
├── features/                  # Feature Processing
│   ├── __init__.py           # Feature exports
│   ├── fourier.py            # Fourier transforms and spectral utilities
│   └── layers.py             # Signal processing layers
│
└── visualization/             # Visualization Tools
    ├── __init__.py           # Visualization exports
    ├── plot.py               # Basic plotting utilities
    ├── visualization.py      # Advanced visualization tools
    └── templates.py          # Plot templates
```

## Key Improvements

### 1. **Clear Separation of Concerns**
- **Diffusion Models**: All diffusion-related components in `models/diffusion/`
- **Spectral Models**: Frequency recovery models in `models/spectral/`
- **Core Infrastructure**: Base utilities and training in `core/`

### 2. **Logical Grouping**
- **Data**: All dataset classes and data loading utilities
- **Training**: Training loops and experiment management
- **Losses**: Loss functions organized by application domain
- **Features**: Signal processing and feature extraction
- **Visualization**: Plotting and analysis tools

### 3. **Modular Import Structure**
```python
# Old structure (mixed functionality)
from src.models import PML_SpectralUNet, PML_SpectralDDPM

# New structure (clear separation)
from src.models.spectral import PML_SpectralUNet
from src.models.diffusion import PML_SpectralDDPM
```

### 4. **Better Maintainability**
- Each module has a focused responsibility
- Related functionality is co-located
- Easier to find and modify specific components
- Better code organization for team development

## Migration Guide

### Automatic Migration
Run the migration script to automatically update the structure:
```bash
python migrate_to_new_structure.py
```

### Manual Import Updates
If you have custom scripts, update imports as follows:

#### Core Infrastructure
```python
# Old
from src.utils import PML_kw
from src.logging import get_logger
from src.trainer import PML_ModelTrainer

# New
from src.core.utils import PML_kw
from src.core.logging import get_logger
from src.core.trainer import PML_ModelTrainer
```

#### Model Architectures
```python
# Old
from src.models import PML_SpectralUNet, PML_SpectralDDPM

# New
from src.models.spectral import PML_SpectralUNet
from src.models.diffusion import PML_SpectralDDPM
```

#### Datasets
```python
# Old
from src.dataset import PML_NpyPairedSeismic
from src.dataset_spectral import PML_SeismicFrequencyRecoveryDataset

# New
from src.data.dataset import PML_NpyPairedSeismic
from src.data.dataset_spectral import PML_SeismicFrequencyRecoveryDataset
```

#### Loss Functions
```python
# Old
from src.losses import PML_lowband_weighted_loss
from src.losses_spectral import create_spectral_loss

# New
from src.losses.losses import PML_lowband_weighted_loss
from src.losses.losses_spectral import create_spectral_loss
```

### Compatibility Layer
A compatibility layer is provided during transition:
```python
# Temporary compatibility (with deprecation warning)
import src.compat  # Provides old import paths

# Preferred new structure
from src.models.spectral import PML_SpectralUNet
```

## Module Descriptions

### `core/` - Core Infrastructure
Contains the fundamental building blocks used throughout the framework:
- **utils.py**: Helper functions like `PML_kw` for parameter handling
- **logging.py**: Centralized logging infrastructure
- **trainer.py**: Base `PML_ModelTrainer` class
- **metrics.py**: Evaluation metrics for model assessment
- **callbacks.py**: Training callbacks and hooks

### `models/` - Model Architectures
Organized by methodology to separate different approaches:

#### `models/diffusion/` - Diffusion Models
- **diffusion.py**: Main `PML_SpectralDDPM` implementation
- **unet_blocks.py**: UNet backbone architecture
- **autoencoder.py**: Autoencoder components
- **conditioning.py**: Conditioning mechanisms for guided generation
- **utils.py**: Diffusion-specific utilities and helpers

#### `models/spectral/` - Spectral Models
- **spectral_unet.py**: `PML_SpectralUNet` for frequency recovery tasks

### `data/` - Data Handling
Centralized data loading and preprocessing:
- **dataset.py**: Base dataset classes (`PML_NpyPairedSeismic`)
- **dataset_spectral.py**: Specialized datasets for frequency recovery

### `training/` - Training Infrastructure
Training loops and experiment management:
- **train_diffusion.py**: Diffusion model training loops and utilities

### `losses/` - Loss Functions
Organized by application domain:
- **losses.py**: Base loss functions for general seismic processing
- **losses_spectral.py**: Specialized losses for frequency recovery

### `features/` - Feature Processing
Signal processing and feature extraction tools:
- **fourier.py**: FFT utilities and spectral analysis
- **layers.py**: Neural network layers for signal processing

### `visualization/` - Visualization Tools
Plotting and analysis utilities:
- **plot.py**: Basic plotting functions
- **visualization.py**: Advanced visualization tools
- **templates.py**: Reusable plot templates

## Benefits of New Structure

### 1. **Scalability**
- Easy to add new model types without cluttering existing modules
- Clear places to add new functionality
- Better support for team development

### 2. **Maintainability**
- Related code is co-located
- Easier to understand module responsibilities
- Simpler to debug and modify specific components

### 3. **Reusability**
- Core components can be easily imported and reused
- Clear separation allows selective usage
- Better support for external projects using the framework

### 4. **Documentation**
- Each module has focused documentation
- Easier to understand the overall architecture
- Better API reference generation

## Testing the New Structure

After migration, verify everything works:

```bash
# Test core imports
python -c "from src.core import PML_kw, get_logger; print('✓ Core imports work')"

# Test model imports
python -c "from src.models.spectral import PML_SpectralUNet; print('✓ Spectral models work')"
python -c "from src.models.diffusion import PML_SpectralDDPM; print('✓ Diffusion models work')"

# Test data imports
python -c "from src.data import PML_SeismicFrequencyRecoveryDataset; print('✓ Data imports work')"

# Test loss imports
python -c "from src.losses import create_spectral_loss; print('✓ Loss imports work')"

# Run example usage
python example_usage.py
```

## Future Enhancements

The new structure makes it easy to add:

1. **New Model Types**: Add new subdirectories under `models/`
2. **Additional Training Methods**: Extend `training/` module
3. **Custom Loss Functions**: Add domain-specific losses under `losses/`
4. **New Features**: Extend `features/` with additional processing tools
5. **Advanced Visualization**: Add new plotting capabilities to `visualization/`

This organized structure provides a solid foundation for continued development and maintenance of the seismic processing framework.
