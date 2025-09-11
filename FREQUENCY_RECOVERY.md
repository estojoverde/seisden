# Seismic Frequency Recovery Experiment

This module implements a complete seismic frequency recovery experiment using U-Net architecture with specialized spectral loss functions. The experiment is integrated into the BaseTrainer/seisden repository structure with proper modular design and logging infrastructure.

## Overview

The frequency recovery task involves predicting low-pass (LP) seismic data from high-pass (HP) input data, with the goal of reconstructing the full-spectrum signal. This is achieved using a U-Net model with custom loss functions that operate in both time and frequency domains.

## Key Components

### Models (`src/spectral_unet.py`)
- **PML_SpectralUNet**: Wrapper around segmentation_models_pytorch U-Net
- **PML_create_spectral_unet()**: Factory function for creating configured models
- Support for various encoder backbones (ResNet, EfficientNet, etc.)

### Datasets (`src/dataset_spectral.py`)
- **PML_SeismicFrequencyRecoveryDataset**: Specialized dataset for HP/LP/full/mask data
- Automatic directory loading and mask-based filtering
- Integration with existing repository patterns

### Loss Functions (`src/losses_spectral.py`)
- **PML_SeismicRecoveryLoss**: Multi-component loss with time and frequency domain penalties
- **PML_SpectralMSELoss**: Pure frequency domain loss
- **PML_ReconstructionPenaltyLoss**: Reconstruction accuracy penalty
- FFT-based magnitude and phase consistency

### Training Infrastructure
- **train_spectralunet.py**: Main training script with detailed loss logging
- **run_experiment.py**: High-level experiment orchestration and parameter sweeps
- Integration with existing PML_ModelTrainer infrastructure

## Quick Start

### 1. Prepare Your Data

Organize your seismic data in the following directory structure:
```
data/
├── HP/          # High-pass seismic data (.npy files)
├── LP/          # Low-pass seismic data (.npy files)  
├── full/        # Full-spectrum seismic data (.npy files)
└── mask/        # Frequency domain masks (.npy files, optional)
```

Each subdirectory should contain `.npy` files with the same naming convention. The data should be 2D arrays representing seismic sections.

### 2. Basic Training

```bash
# Simple training run
python train_spectralunet.py \
    --data-dir /path/to/your/data \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-3

# Training with custom loss weights
python train_spectralunet.py \
    --data-dir /path/to/your/data \
    --epochs 100 \
    --penalty-weight 1.5 \
    --magnitude-weight 0.7 \
    --phase-weight 0.3
```

### 3. Configuration-Based Training

Create a configuration file (see `experiments/spectral_config.yaml` for template):

```bash
# Update data_dir in experiments/spectral_config.yaml
python run_experiment.py --config experiments/spectral_config.yaml
```

### 4. Parameter Sweeps

```bash
# Run hyperparameter sweep
python run_experiment.py \
    --experiment-name "hp_sweep_v1" \
    --sweep \
    --param-grid experiments/param_grid.yaml \
    --data-dir /path/to/your/data
```

## Architecture Details

### Model Architecture
- **Encoder**: Pretrained ResNet34/ResNet50/EfficientNet backbones
- **Decoder**: U-Net decoder with skip connections
- **Input**: Single-channel high-pass seismic data
- **Output**: Single-channel predicted low-pass data

### Loss Function Components

The `PML_SeismicRecoveryLoss` combines multiple objectives:

1. **Primary Loss** (λ=1.0): MSE/L1/Huber loss between predicted and target LP data
2. **Reconstruction Penalty** (λ=1.0): Error when combining predicted LP with input HP
3. **FFT Magnitude Penalty** (λ=0.5): Spectral magnitude consistency in frequency domain
4. **FFT Phase Penalty** (λ=0.3): Phase coherence using cosine distance

```python
total_loss = primary_loss + λ₁×reconstruction_loss + λ₂×magnitude_loss + λ₃×phase_loss
```

### Frequency Domain Processing

The model incorporates frequency domain analysis through:
- Real FFT (rfft2) for computational efficiency
- Magnitude spectrum comparison
- Phase coherence measurement with wrap-around handling
- Frequency domain mask filtering (when available)

## Configuration Options

### Model Configuration
```yaml
encoder: "resnet34"           # Encoder backbone
encoder_weights: "imagenet"   # Pretrained weights
in_channels: 1               # Input channels
classes: 1                   # Output channels
```

### Loss Configuration
```yaml
loss_type: "seismic_recovery"
penalty_weight: 1.0          # Reconstruction penalty weight
magnitude_weight: 0.5        # FFT magnitude penalty weight
phase_weight: 0.3            # FFT phase penalty weight
primary_loss: "mse"          # Primary loss type
```

### Training Configuration
```yaml
epochs: 100
batch_size: 8
learning_rate: 1e-3
weight_decay: 1e-4
optimizer: "adam"
val_frequency: 5             # Validate every N epochs
```

## Experiment Management

### Directory Structure
```
experiments/
├── spectral_recovery_baseline_20240101_120000/
│   ├── run_001/
│   │   ├── config.yaml      # Run configuration
│   │   ├── results.json     # Run results and metrics
│   │   └── best_model.pth   # Best model checkpoint
│   ├── experiment_summary.json  # Aggregate results
│   └── best_result.json     # Best run across all parameters
```

### Results Tracking

Each experiment automatically tracks:
- Loss components (primary, reconstruction, magnitude, phase)
- Training and validation metrics
- Model checkpoints (best validation loss)
- Configuration and hyperparameters
- Training duration and status

### Hyperparameter Sweeps

The parameter sweep functionality allows systematic exploration of:
- Learning rates and optimization parameters
- Loss component weights
- Model architecture variants
- Batch sizes and training configurations

## Integration with Existing Repository

### Repository Patterns
- **PML_ Prefixing**: All components follow the existing naming convention
- **Logging Infrastructure**: Uses `src/logging.py` for consistent logging
- **Trainer Integration**: Extends `PML_ModelTrainer` for compatibility
- **Utils Integration**: Uses `PML_kw` helper functions for parameter handling

### Modular Design
- **Models**: Separate module for U-Net architecture variants
- **Datasets**: Specialized dataset classes with repository conventions
- **Losses**: Dedicated loss functions for frequency recovery
- **Training**: Standalone training scripts with experiment orchestration

## Advanced Usage

### Custom Loss Functions

Create custom loss configurations:
```python
from src.losses_spectral import create_spectral_loss

# Custom loss with different weights
loss_fn = create_spectral_loss(
    loss_type="seismic_recovery",
    penalty_weight=2.0,
    magnitude_weight=1.0,
    phase_weight=0.1
)
```

### Model Variants

Use different encoder backbones:
```python
from src.models import PML_create_spectral_unet

# EfficientNet-based model
model = PML_create_spectral_unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)
```

### Custom Datasets

Extend the dataset for specific requirements:
```python
from src.dataset_spectral import PML_SeismicFrequencyRecoveryDataset

class CustomSeismicDataset(PML_SeismicFrequencyRecoveryDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom preprocessing
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # Add custom transformations
        return sample
```

## Troubleshooting

### Common Issues

1. **Data Loading Errors**
   - Ensure HP, LP, and full directories exist with matching file names
   - Check that .npy files have compatible shapes
   - Verify data path in configuration

2. **Memory Issues**
   - Reduce batch size in configuration
   - Use gradient accumulation for effective larger batches
   - Consider mixed precision training

3. **Convergence Problems**
   - Adjust learning rate and loss component weights
   - Verify data normalization and scaling
   - Check for NaN values in loss computation

4. **GPU Issues**
   - Set device to "cpu" for CPU-only training
   - Reduce batch size for GPU memory constraints
   - Check CUDA installation and compatibility

### Debugging

Enable detailed logging:
```bash
export PYTHONPATH=/path/to/BaseTrainer/seisden:$PYTHONPATH
python train_spectralunet.py --data-dir /path/to/data --epochs 1 --batch-size 1
```

## Performance Optimization

### Training Speed
- Use `num_workers > 0` for data loading
- Enable `pin_memory=True` for GPU training
- Consider mixed precision training for modern GPUs

### Memory Optimization
- Gradient accumulation for larger effective batch sizes
- Gradient checkpointing for reduced memory usage
- Dynamic loss scaling for mixed precision

### Model Selection
- Start with ResNet34 for baseline performance
- Try EfficientNet for better efficiency
- Use ResNet50 for higher capacity models

## Citation

If you use this frequency recovery implementation in your research, please cite the BaseTrainer/seisden repository and relevant papers on seismic processing and U-Net architectures.

## Support

For issues and questions:
1. Check this README and configuration examples
2. Review the existing repository documentation
3. Examine the modular code structure in `src/`
4. Test with small datasets first to validate setup
