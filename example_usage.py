#!/usr/bin/env python3
"""
example_usage.py

Example script demonstrating how to use the seismic frequency recovery experiment
components programmatically.

This script shows how to:
1. Create and configure models
2. Set up datasets
3. Define custom loss functions
4. Run training loops
5. Evaluate results

Run this script to verify that all components are working correctly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

# Import our modules
from src.models.spectral import PML_create_spectral_unet
from src.data.dataset_spectral import PML_SeismicFrequencyRecoveryDataset
from src.losses.losses_spectral import create_spectral_loss
from src.core.logging import get_logger

def create_dummy_data(data_dir: Path, num_samples: int = 10):
    """Create dummy seismic data for testing purposes."""
    data_dir = Path(data_dir)
    
    # Create subdirectories
    for subdir in ["HP", "LP", "full", "mask"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Generate dummy 2D seismic data
    height, width = 256, 256
    
    # Create arrays to hold all samples (N, H, W) format
    all_hp_data = np.zeros((num_samples, height, width), dtype=np.float32)
    all_lp_data = np.zeros((num_samples, height, width), dtype=np.float32)
    all_full_data = np.zeros((num_samples, height, width), dtype=np.float32)
    all_mask_data = np.ones((num_samples, height, width), dtype=np.float32)
    
    for i in range(num_samples):
        # Create correlated HP, LP, and full spectrum data
        full_data = np.random.randn(height, width).astype(np.float32)
        
        # Simulate frequency filtering (rough approximation)
        hp_data = full_data - np.mean(full_data, axis=0, keepdims=True)  # Remove low frequencies
        lp_data = np.mean(full_data, axis=0, keepdims=True) + 0.1 * np.random.randn(height, width).astype(np.float32)
        
        # Create frequency mask (example: emphasize certain frequencies)
        mask = np.ones((height, width), dtype=np.float32)
        mask[height//4:3*height//4, width//4:3*width//4] = 0.5  # Reduce middle frequencies
        
        # Store in arrays
        all_hp_data[i] = hp_data
        all_lp_data[i] = lp_data
        all_full_data[i] = full_data
        all_mask_data[i] = mask
    
    # Save all data as single files
    np.save(data_dir / "HP" / "all_samples.npy", all_hp_data)
    np.save(data_dir / "LP" / "all_samples.npy", all_lp_data)
    np.save(data_dir / "full" / "all_samples.npy", all_full_data)
    np.save(data_dir / "mask" / "all_samples.npy", all_mask_data)
    
    return data_dir


def test_model_creation():
    """Test model creation and basic forward pass."""
    print("Testing model creation...")
    
    # Create model
    model = PML_create_spectral_unet(
        encoder_name="resnet34",
        encoder_weights=None,  # No pretrained weights for testing
        in_channels=1,
        classes=1
    )
    
    print(f"Model created: {type(model)}")
    print(f"Model info: {model.get_model_info()}")
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 256, 256)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == dummy_input.shape, "Output shape should match input shape"
    print("✓ Model test passed!")
    
    return model


def test_dataset_creation(data_dir: Path):
    """Test dataset creation and data loading."""
    print("Testing dataset creation...")
    
    # Load the data files
    hp_data = str(data_dir / "HP" / "all_samples.npy")
    lp_data = str(data_dir / "LP" / "all_samples.npy") 
    full_data = str(data_dir / "full" / "all_samples.npy")
    mask_data = str(data_dir / "mask" / "all_samples.npy")
    
    # Create dataset with individual files
    dataset = PML_SeismicFrequencyRecoveryDataset(
        hp_data=hp_data,
        lp_data=lp_data,
        full_data=full_data,
        mask_data=mask_data,
        b_return_mask=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test data loading
    sample = dataset[0]
    
    # The dataset returns a tuple: (hp, lp, full) or (hp, lp, full, mask)
    if len(sample) == 4:
        hp, lp, full, mask = sample
        print(f"hp shape: {hp.shape}")
        print(f"lp shape: {lp.shape}")
        print(f"full shape: {full.shape}")
        print(f"mask shape: {mask.shape}")
    else:
        hp, lp, full = sample
        print(f"hp shape: {hp.shape}")
        print(f"lp shape: {lp.shape}")
        print(f"full shape: {full.shape}")
    
    # Test dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    
    if len(batch) == 4:
        batch_hp, batch_lp, batch_full, batch_mask = batch
        print("Batch items: hp, lp, full, mask")
        print(f"Batch hp shape: {batch_hp.shape}")
        print(f"Batch lp shape: {batch_lp.shape}")
        print(f"Batch full shape: {batch_full.shape}")
        print(f"Batch mask shape: {batch_mask.shape}")
    else:
        batch_hp, batch_lp, batch_full = batch
        print("Batch items: hp, lp, full")
        print(f"Batch hp shape: {batch_hp.shape}")
        print(f"Batch lp shape: {batch_lp.shape}")
        print(f"Batch full shape: {batch_full.shape}")
    
    print("✓ Dataset test passed!")
    
    return dataset


def test_loss_functions():
    """Test loss function creation and computation."""
    print("Testing loss functions...")
    
    # Create loss function
    loss_fn = create_spectral_loss(
        loss_type="seismic_recovery",
        penalty_weight=1.0,
        magnitude_weight=0.5,
        phase_weight=0.3
    )
    
    print(f"Loss function created: {type(loss_fn)}")
    
    # Create dummy data for loss computation (with gradients)
    batch_size = 2
    predicted_lp = torch.randn(batch_size, 1, 256, 256, requires_grad=True)
    target_lp = torch.randn(batch_size, 1, 256, 256)
    input_hp = torch.randn(batch_size, 1, 256, 256)
    target_full = torch.randn(batch_size, 1, 256, 256)
    
    # Compute loss
    total_loss, primary_loss, recon_loss, mag_loss, phase_loss = loss_fn(
        predicted_lp, target_lp, input_hp, target_full
    )
    
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Primary loss: {primary_loss.item():.6f}")
    print(f"Reconstruction loss: {recon_loss.item():.6f}")
    print(f"Magnitude loss: {mag_loss.item():.6f}")
    print(f"Phase loss: {phase_loss.item():.6f}")
    
    # Verify loss is differentiable
    total_loss.backward()
    
    print("✓ Loss function test passed!")
    
    return loss_fn


def test_training_step(model, dataset, loss_fn):
    """Test a complete training step."""
    print("Testing training step...")
    
    # Setup
    device = torch.device("cpu")  # Use CPU for testing
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    
    # Handle tuple format from dataset
    if len(batch) == 4:
        hp_data, lp_data, full_data, mask_data = batch
    else:
        hp_data, lp_data, full_data = batch
        mask_data = None
    
    # Move to device
    hp_data = hp_data.to(device)
    lp_data = lp_data.to(device)
    full_data = full_data.to(device)
    if mask_data is not None:
        mask_data = mask_data.to(device)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(hp_data)
    
    # Apply mask if available
    if mask_data is not None:
        predictions = predictions * mask_data
    
    # Compute loss
    total_loss, primary_loss, recon_loss, mag_loss, phase_loss = loss_fn(
        predicted_lp=predictions,
        target_lp=lp_data,
        input_hp=hp_data,
        target_full=full_data
    )
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    print(f"Training step completed - Loss: {total_loss.item():.6f}")
    print("✓ Training step test passed!")


def main():
    """Run all tests to verify the frequency recovery experiment components."""
    logger = get_logger("example_usage")
    logger.info("Starting frequency recovery experiment component tests...")
    
    try:
        # Create dummy data directory
        data_dir = Path("./test_data")
        print("Creating dummy data...")
        create_dummy_data(data_dir, num_samples=5)
        print(f"✓ Dummy data created in: {data_dir}")
        
        # Test model
        model = test_model_creation()
        
        # Test dataset
        dataset = test_dataset_creation(data_dir)
        
        # Test loss functions
        loss_fn = test_loss_functions()
        
        # Test training step
        test_training_step(model, dataset, loss_fn)
        
        print("\n" + "="*50)
        print("🎉 All tests passed successfully!")
        print("The frequency recovery experiment components are working correctly.")
        print("="*50)
        
        # Cleanup dummy data
        import shutil
        shutil.rmtree(data_dir)
        print(f"✓ Cleaned up test data directory: {data_dir}")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
