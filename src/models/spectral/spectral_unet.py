# src/models/spectral_unet.py
"""
Spectral U-Net models for seismic frequency recovery tasks.

This module provides U-Net architectures specifically designed for seismic frequency recovery,
where the goal is to predict low-pass components from high-pass filtered data.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError(
        "segmentation_models_pytorch is required for SpectralUNet. "
        "Install it with: pip install segmentation-models-pytorch"
    )

from ...core.utils import PML_kw

__all__ = [
    "PML_SpectralUNet",
    "PML_create_spectral_unet",
]


class PML_SpectralUNet(nn.Module):
    """
    U-Net wrapper specifically designed for seismic frequency recovery.
    
    This model takes high-pass filtered seismic data as input and predicts
    the corresponding low-pass components. The model is based on segmentation_models_pytorch
    U-Net with customizable encoder and activation functions.
    
    Architecture:
        - Encoder: ResNet-based (default: ResNet34) with ImageNet pretraining
        - Decoder: U-Net decoder with skip connections
        - Output: Single channel with sigmoid activation for [0,1] normalization
    
    Args:
        encoder_name: Name of the encoder backbone (e.g., 'resnet34', 'resnet50')
        encoder_weights: Pretrained weights source ('imagenet', 'ssl', 'swsl', None)
        in_channels: Number of input channels (default: 1 for seismic data)
        classes: Number of output channels (default: 1 for low-pass prediction)
        activation: Output activation function ('sigmoid', 'tanh', None)
        **kwargs: Additional arguments passed to smp.Unet
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet", 
        in_channels: int = 1,
        classes: int = 1,
        activation: str = "sigmoid",
        **kwargs: Any
    ):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.activation = activation
        
        # Create the U-Net model
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor (B, C, H, W) - high-pass filtered seismic data
            
        Returns:
            Predicted low-pass components (B, C, H, W)
        """
        return self.unet(x)
    
    def get_encoder(self) -> nn.Module:
        """Get the encoder part of the U-Net."""
        return self.unet.encoder
    
    def get_decoder(self) -> nn.Module:
        """Get the decoder part of the U-Net."""
        return self.unet.decoder
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        return {
            'encoder_name': self.encoder_name,
            'encoder_weights': self.encoder_weights,
            'in_channels': self.in_channels,
            'classes': self.classes,
            'activation': self.activation,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def PML_create_spectral_unet(**kwargs: Any) -> PML_SpectralUNet:
    """
    Factory function to create a SpectralUNet with common defaults.
    
    Optional **kwargs (defaults shown):
        encoder_name (str) = "resnet34"
            Encoder backbone architecture
        encoder_weights (str) = "imagenet" 
            Pretrained weights source
        in_channels (int) = 1
            Number of input channels
        classes (int) = 1
            Number of output classes/channels
        activation (str) = "sigmoid"
            Output activation function
        
    Returns:
        Configured PML_SpectralUNet model
        
    Example:
        >>> model = PML_create_spectral_unet(
        ...     encoder_name="resnet50",
        ...     encoder_weights="imagenet"
        ... )
    """
    encoder_name = PML_kw("encoder_name", kwargs, "resnet34")
    encoder_weights = PML_kw("encoder_weights", kwargs, "imagenet")
    in_channels = int(PML_kw("in_channels", kwargs, 1))
    classes = int(PML_kw("classes", kwargs, 1))
    activation = PML_kw("activation", kwargs, "sigmoid")
    
    # Remove known parameters to pass remaining kwargs to the model
    known_params = {'encoder_name', 'encoder_weights', 'in_channels', 'classes', 'activation'}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in known_params}
    
    return PML_SpectralUNet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
        **extra_kwargs
    )
