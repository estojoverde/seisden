#!/usr/bin/env python3
"""
train_spectralunet.py

Training script for seismic frequency recovery using U-Net with spectral loss.

This script integrates with the existing BaseTrainer/seisden repository structure,
using the PML_ModelTrainer and logging infrastructure while providing specialized
functionality for frequency recovery tasks.

Usage:
    python train_spectralunet.py --config-file config_spectral.yaml
    python train_spectralunet.py --data-dir /path/to/data --epochs 100
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import base trainer infrastructure
from src.core.trainer import PML_ModelTrainer
from src.models.spectral import PML_create_spectral_unet
from src.data.dataset_spectral import PML_SeismicFrequencyRecoveryDataset
from src.losses import create_spectral_loss
from src.core.logging import get_logger
from src.core.utils import PML_kw

__all__ = ["SpectralUNetTrainer", "main"]


class SpectralUNetTrainer(PML_ModelTrainer):
    """
    Specialized trainer for seismic frequency recovery using U-Net.
    
    Extends the base PML_ModelTrainer with specific functionality for:
    - Multi-component loss computation (primary + reconstruction + spectral)
    - Frequency recovery validation metrics
    - Custom logging of spectral loss components
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        **kwargs
    ):
        super().__init__(model, loss_fn, optimizer, device, **kwargs)
        
        self.logger = get_logger("SpectralUNetTrainer")
        
        # Track loss components for logging
        self.loss_components = {
            "primary": [],
            "reconstruction": [],
            "magnitude": [],
            "phase": []
        }
    
    def compute_loss(self, batch, predictions):
        """
        Compute multi-component loss for frequency recovery.
        
        Args:
            batch: Dictionary containing HP, LP, full, and mask data
            predictions: Model predictions (predicted LP)
            
        Returns:
            loss: Total loss for backpropagation
            loss_dict: Dictionary of loss components for logging
        """
        hp_data = batch["hp"]
        lp_target = batch["lp"]
        full_target = batch["full"]
        
        # Handle mask if available
        mask = batch.get("mask")
        if mask is not None:
            # Apply mask to predicted LP (similar to datagen.ipynb)
            predictions = predictions * mask
        
        # Compute multi-component loss
        total_loss, primary_loss, recon_loss, mag_loss, phase_loss = self.loss_fn(
            predicted_lp=predictions,
            target_lp=lp_target,
            input_hp=hp_data,
            target_full=full_target
        )
        
        # Store components for logging
        loss_dict = {
            "total": total_loss.item(),
            "primary": primary_loss.item(),
            "reconstruction": recon_loss.item(),
            "magnitude": mag_loss.item(),
            "phase": phase_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_step(self, batch):
        """Single training step with detailed loss logging."""
        self.model.train()
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Forward pass
        hp_data = batch["hp"]
        predictions = self.model(hp_data)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(batch, predictions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update loss component tracking
        for component, value in loss_dict.items():
            if component != "total":
                self.loss_components[component].append(value)
        
        return loss_dict
    
    def validate_step(self, batch):
        """Single validation step."""
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            hp_data = batch["hp"]
            predictions = self.model(hp_data)
            
            # Compute loss
            loss, loss_dict = self.compute_loss(batch, predictions)
        
        return loss_dict
    
    def log_epoch_summary(self, epoch, train_losses, val_losses):
        """Log detailed epoch summary with loss components."""
        # Average loss components
        avg_train_components = {}
        for component in self.loss_components:
            if self.loss_components[component]:
                avg_train_components[component] = sum(self.loss_components[component]) / len(self.loss_components[component])
                self.loss_components[component] = []  # Reset for next epoch
        
        # Log summary
        self.logger.info(f"Epoch {epoch + 1} Summary:")
        self.logger.info(f"  Train Total Loss: {train_losses['total']:.6f}")
        self.logger.info(f"  Valid Total Loss: {val_losses['total']:.6f}")
        
        if avg_train_components:
            self.logger.info("  Train Loss Components:")
            for component, value in avg_train_components.items():
                self.logger.info(f"    {component.title()}: {value:.6f}")
        
        self.logger.info("  Valid Loss Components:")
        for component, value in val_losses.items():
            if component != "total":
                self.logger.info(f"    {component.title()}: {value:.6f}")


def create_datasets(data_dir: str, **kwargs):
    """
    Create training and validation datasets.
    
    Args:
        data_dir: Path to directory containing HP/LP/full/mask subdirectories
        **kwargs: Additional dataset configuration
        
    Returns:
        train_dataset, val_dataset: Dataset instances
    """
    # Extract configuration with defaults
    train_split = PML_kw("train_split", kwargs, 0.8)
    val_split = PML_kw("val_split", kwargs, 0.2)
    img_size = PML_kw("img_size", kwargs, (256, 256))
    
    # Create full dataset
    full_dataset = PML_SeismicFrequencyRecoveryDataset(
        data_dir=data_dir,
        img_size=img_size,
        load_mask=True,  # Load masks for LP filtering
        **kwargs
    )
    
    # Split into train/validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    return train_dataset, val_dataset


def create_model(**kwargs):
    """Create spectral U-Net model."""
    return PML_create_spectral_unet(**kwargs)


def create_loss(**kwargs):
    """Create spectral recovery loss function."""
    return create_spectral_loss(**kwargs)


def create_optimizer(model, **kwargs):
    """Create optimizer."""
    lr = PML_kw("learning_rate", kwargs, 1e-3)
    weight_decay = PML_kw("weight_decay", kwargs, 1e-4)
    optimizer_type = PML_kw("optimizer", kwargs, "adam")
    
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        momentum = PML_kw("momentum", kwargs, 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Spectral U-Net for seismic frequency recovery")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, required=True,
                      help="Path to data directory with HP/LP/full/mask subdirectories")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                      help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                      help="Weight decay")
    
    # Model arguments
    parser.add_argument("--encoder", type=str, default="resnet34",
                      help="Encoder backbone for U-Net")
    parser.add_argument("--encoder-weights", type=str, default="imagenet",
                      help="Encoder pretrained weights")
    
    # Loss arguments
    parser.add_argument("--penalty-weight", type=float, default=1.0,
                      help="Weight for reconstruction penalty")
    parser.add_argument("--magnitude-weight", type=float, default=0.5,
                      help="Weight for FFT magnitude penalty")
    parser.add_argument("--phase-weight", type=float, default=0.3,
                      help="Weight for FFT phase penalty")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                      help="Device to use (cuda, cpu, or auto)")
    parser.add_argument("--num-workers", type=int, default=4,
                      help="Number of data loader workers")
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                      help="Directory to save model checkpoints")
    
    # Validation arguments
    parser.add_argument("--val-frequency", type=int, default=5,
                      help="Validation frequency (epochs)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Setup logging
    logger = get_logger("train_spectralunet")
    logger.info(f"Starting spectral U-Net training with args: {args}")
    
    try:
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset, val_dataset = create_datasets(
            data_dir=args.data_dir,
            img_size=(256, 256)
        )
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False
        )
        
        # Create model
        logger.info("Creating model...")
        model = create_model(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            in_channels=1,
            classes=1
        )
        model = model.to(device)
        
        # Create loss function
        logger.info("Creating loss function...")
        loss_fn = create_loss(
            penalty_weight=args.penalty_weight,
            magnitude_weight=args.magnitude_weight,
            phase_weight=args.phase_weight
        )
        
        # Create optimizer
        logger.info("Creating optimizer...")
        optimizer = create_optimizer(
            model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = SpectralUNetTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            save_dir=args.save_dir
        )
        
        # Training loop
        logger.info("Starting training...")
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            # Training
            train_losses = {"total": 0.0, "primary": 0.0, "reconstruction": 0.0, "magnitude": 0.0, "phase": 0.0}
            train_batches = 0
            
            for batch in train_loader:
                loss_dict = trainer.train_step(batch)
                for key, value in loss_dict.items():
                    train_losses[key] += value
                train_batches += 1
            
            # Average training losses
            for key in train_losses:
                train_losses[key] /= train_batches
            
            # Validation
            val_losses = {"total": 0.0, "primary": 0.0, "reconstruction": 0.0, "magnitude": 0.0, "phase": 0.0}
            val_batches = 0
            
            if (epoch + 1) % args.val_frequency == 0:
                for batch in val_loader:
                    loss_dict = trainer.validate_step(batch)
                    for key, value in loss_dict.items():
                        val_losses[key] += value
                    val_batches += 1
                
                # Average validation losses
                for key in val_losses:
                    val_losses[key] /= val_batches
                
                # Log epoch summary
                trainer.log_epoch_summary(epoch, train_losses, val_losses)
                
                # Save best model
                if val_losses["total"] < best_val_loss:
                    best_val_loss = val_losses["total"]
                    checkpoint_path = Path(args.save_dir) / "best_model.pth"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'args': args
                    }, checkpoint_path)
                    logger.info(f"Saved best model with validation loss: {best_val_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_losses['total']:.6f}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
