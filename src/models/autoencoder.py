"""
Autoencoder module providing:
  - Fully convolutional encoder/decoder & autoencoder
  - Dense (flattened latent) encoder/decoder & autoencoder
  - Dense VAE encoder/decoder & autoencoder
Common features:
  * Optional automatic latent shape inference (given input_shape)
  * Sampling utilities (prior / posterior where applicable)
  * Centralized parameter counting helpers
  * Optional logger (uses PML_Logger if none provided)
  * Backward compatibility aliases for old class names
"""

from __future__ import annotations
from typing import Sequence, Optional, Literal, Tuple, Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# from .utils import count_parameters, model_size_mb

try:
    # Local logger
    from ..logging import PML_Logger
except ImportError:
    PML_Logger = None  # Fallback if logger module not available



# ---------------------------------------------------------------------------
# Logging mixin
# ---------------------------------------------------------------------------
class _WithLogger:
    """Mixin to provide a self.logger attribute.

    If an external logger (logging.Logger) is not provided, creates one through
    PML_Logger (if available) or falls back to a silent stub.
    """
    def _init_logger(self, logger: Optional[Any], name: str):
        if logger is not None:
            self.logger = logger if hasattr(logger, "info") else logger.get_logger()
            return
        if PML_Logger is not None:
            self.logger = PML_Logger(name_prefix=name).get_logger()
        else:
            import logging
            self.logger = logging.getLogger(name)
            if not self.logger.handlers:
                self.logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Building Blocks (Shared)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Conv2d -> (optional Dropout2d) -> Activation."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: nn.Module = nn.PReLU,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        dropout = float(max(0.0, min(1.0, dropout)))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.dropout(self.conv(x)))


class EncoderBlock(nn.Module):
    """Two ConvBlocks + MaxPool downsampling."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: nn.Module = nn.PReLU,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.block1 = ConvBlock(in_channels, out_channels, activation=activation, dropout=dropout)
        self.block2 = ConvBlock(out_channels, out_channels, activation=activation, dropout=dropout * 2)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.pool(x)


class DecoderBlock(nn.Module):
    """Upsample (ConvTranspose2d) + ConvBlock refinement."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: nn.Module = nn.PReLU,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_act = activation()
        self.conv = ConvBlock(out_channels, out_channels, activation=activation, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_act(self.up(x))
        return self.conv(x)


# ---------------------------------------------------------------------------
# Fully Convolutional Encoder / Decoder
# ---------------------------------------------------------------------------
class FullyConvEncoder(nn.Module, _WithLogger):
    """
    Fully convolutional encoder producing a latent feature map (no flatten).

    If input_shape=(C,H,W) is provided, latent_shape is inferred immediately.
    Otherwise, latent_shape is available after first forward pass.
    """
    def __init__(self,
                 in_channels: int,
                 channels: Sequence[int],
                 activation: nn.Module = nn.PReLU,
                 dropout: float = 0.1,
                 input_shape: Optional[Sequence[int]] = None,
                 logger: Optional[Any] = None,
                 **kwargs):
        super().__init__()
        self._init_logger(logger, "FullyConvEncoder")
        self.in_channels = in_channels
        self.channels = list(channels)
        self.blocks = nn.ModuleList()
        prev = in_channels
        for ch in self.channels:
            self.blocks.append(EncoderBlock(prev, ch, activation=activation, dropout=dropout))
            prev = ch
        self._latent_shape = None
        if input_shape is not None:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                latent = self.forward(dummy)
                self._latent_shape = latent.shape[1:]
        self.logger.info("FullyConvEncoder initialized (channels=%s)", self.channels)

    @property
    def latent_shape(self) -> Optional[Tuple[int, int, int]]:
        return self._latent_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self._latent_shape is None:
            self._latent_shape = tuple(x.shape[1:])
            self.logger.debug("Inferred latent_shape=%s", self._latent_shape)
        return x


class FullyConvDecoder(nn.Module, _WithLogger):
    """
    Fully convolutional decoder expecting latent feature maps.

    channels argument should mirror the encoder's channels order (ascending).
    """
    def __init__(self,
                 out_channels: int,
                 channels: Sequence[int],
                 activation: nn.Module = nn.PReLU,
                 dropout: float = 0.1,
                 latent_shape: Optional[Sequence[int]] = None,
                 logger: Optional[Any] = None,
                 **kwargs):
        super().__init__()
        self._init_logger(logger, "FullyConvDecoder")
        self.out_channels = out_channels
        self.channels = list(channels)
        dec_channels = list(reversed(self.channels))
        blocks = []
        prev = dec_channels[0]
        for ch in dec_channels[1:]:
            blocks.append(DecoderBlock(prev, ch, activation=activation, dropout=dropout))
            prev = ch
        blocks.append(DecoderBlock(prev, out_channels, activation=activation, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.final_act = nn.Sigmoid()
        self.latent_shape = tuple(latent_shape) if latent_shape is not None else None
        self.logger.info("FullyConvDecoder initialized (channels=%s)", self.channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.final_act(self.blocks(z))

    def sample_prior(self,
                     n: int,
                     device: Optional[str] = None,
                     spatial_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Sample random latent feature maps ~ N(0,1) and decode.

        If latent_shape known: (C_lat,H_lat,W_lat); else spatial_shape must be given.
        """
        if device is None:
            device = next(self.parameters()).device
        if self.latent_shape is None:
            if spatial_shape is None:
                raise ValueError("latent_shape unknown. Provide spatial_shape=(H,W).")
            c_lat = self.channels[-1]
            h, w = spatial_shape
        else:
            c_lat, h, w = self.latent_shape
        z = torch.randn(n, c_lat, h, w, device=device)
        return self.forward(z)


class FullyConvAutoencoder(nn.Module, _WithLogger):
    """Wrapper combining FullyConvEncoder + FullyConvDecoder with convenience methods."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 channels: Sequence[int],
                 activation: nn.Module = nn.PReLU,
                 dropout: float = 0.1,
                 input_shape: Optional[Sequence[int]] = None,
                 logger: Optional[Any] = None,
                 **kwargs):
        super().__init__()
        self._init_logger(logger, "FullyConvAutoencoder")
        self.encoder = FullyConvEncoder(in_channels, channels,
                                        activation=activation,
                                        dropout=dropout,
                                        input_shape=input_shape,
                                        logger=self.logger)
        self.decoder = FullyConvDecoder(out_channels,
                                        channels,
                                        activation=activation,
                                        dropout=dropout,
                                        latent_shape=self.encoder.latent_shape,
                                        logger=self.logger)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sample_prior(self, n: int, device: Optional[str] = None) -> torch.Tensor:
        return self.decoder.sample_prior(n, device=device)


# ---------------------------------------------------------------------------
# Dense (vector latent) Encoder / Decoder / Autoencoder
# ---------------------------------------------------------------------------
class DenseEncoder(nn.Module, _WithLogger):
    """Convolutional feature extractor with flatten + Linear -> latent vector."""
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_shape: Sequence[int],
                 num_blocks: int = 3,
                 channels: Optional[Sequence[int]] = None,
                 activation: nn.Module = nn.ReLU,
                 logger: Optional[Any] = None,
                 **kwargs):
        super().__init__()
        self._init_logger(logger, "DenseEncoder")
        if channels is None:
            channels = [32 * (2 ** i) for i in range(num_blocks)]
        assert len(channels) == num_blocks, "channels length must equal num_blocks"
        layers = []
        prev = in_channels
        for ch in channels:
            layers += [nn.Conv2d(prev, ch, 3, padding=1), activation(), nn.MaxPool2d(2)]
            prev = ch
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feat = self.conv(dummy)
        self._enc_out_shape = feat.shape[1:]
        self._flatten_dim = feat.view(1, -1).shape[1]
        self.fc = nn.Linear(self._flatten_dim, latent_dim)
        self.logger.info("DenseEncoder latent_dim=%d flatten_dim=%d out_shape=%s",
                         latent_dim, self._flatten_dim, self._enc_out_shape)

    @property
    def enc_out_shape(self):
        return self._enc_out_shape

    @property
    def flatten_dim(self):
        return self._flatten_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DenseDecoder(nn.Module, _WithLogger):
    """Linear expand + stack of ConvTranspose2d layers to reconstruct."""
    def __init__(self,
                 out_channels: int,
                 latent_dim: int,
                 enc_out_shape: Sequence[int],
                 num_blocks: int = 3,
                 channels: Optional[Sequence[int]] = None,
                 activation: nn.Module = nn.ReLU,
                 logger: Optional[Any] = None,
                 **kwargs):
        super().__init__()
        self._init_logger(logger, "DenseDecoder")
        if channels is None:
            channels = [32 * (2 ** i) for i in range(num_blocks)]
        decoder_channels = list(reversed(channels))
        self.enc_out_shape = tuple(enc_out_shape)
        self.flatten_dim = int(torch.prod(torch.tensor(enc_out_shape)))
        self.fc = nn.Linear(latent_dim, self.flatten_dim)
        layers = [nn.Unflatten(1, self.enc_out_shape)]
        prev = decoder_channels[0]
        for ch in decoder_channels[1:]:
            layers += [nn.ConvTranspose2d(prev, ch, 2, stride=2), activation()]
            prev = ch
        layers += [nn.ConvTranspose2d(prev, out_channels, 2, stride=2), nn.Sigmoid()]
        self.deconv = nn.Sequential(*layers)
        self.logger.info("DenseDecoder initialized enc_out_shape=%s", self.enc_out_shape)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        return self.deconv(x)


class DenseAutoencoder(nn.Module, _WithLogger):
    """Deterministic dense (vector latent) autoencoder."""
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_shape: Sequence[int],
                 num_blocks: int = 3,
                 channels: Optional[Sequence[int]] = None,
                 activation: nn.Module = nn.ReLU,
                 logger: Optional[Any] = None,
                 **kwargs):
        super().__init__()
        self._init_logger(logger, "DenseAutoencoder")
        self.encoder = DenseEncoder(in_channels, latent_dim, input_shape,
                                    num_blocks=num_blocks, channels=channels,
                                    activation=activation, logger=self.logger)
        self.decoder = DenseDecoder(in_channels, latent_dim, self.encoder.enc_out_shape,
                                    num_blocks=num_blocks, channels=channels,
                                    activation=activation, logger=self.logger)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sample_prior(self, n: int, device: Optional[str] = None) -> torch.Tensor:
        """Sample z~N(0,I) and decode."""
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.encoder.fc.out_features, device=device)
        return self.decode(z)


# ---------------------------------------------------------------------------
# Dense VAE
# ---------------------------------------------------------------------------
class DenseEncoderVAE(nn.Module, _WithLogger):
    """Encoder producing (mu, logvar) for latent Gaussian."""
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_shape: Sequence[int],
                 num_blocks: int = 3,
                 channels: Optional[Sequence[int]] = None,
                 activation: nn.Module = nn.ReLU,
                 logger: Optional[Any] = None,
                 **kwargs):
        super().__init__()
        self._init_logger(logger, "DenseEncoderVAE")
        if channels is None:
            channels = [32 * (2 ** i) for i in range(num_blocks)]
        layers = []
        prev = in_channels
        for ch in channels:
            layers += [nn.Conv2d(prev, ch, 3, padding=1), activation(), nn.MaxPool2d(2)]
            prev = ch
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.conv(dummy)
        self._enc_out_shape = out.shape[1:]
        self._flatten_dim = out.view(1, -1).shape[1]
        self.fc_mu = nn.Linear(self._flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flatten_dim, latent_dim)
        self.logger.info("DenseEncoderVAE latent_dim=%d flatten_dim=%d", latent_dim, self._flatten_dim)

    @property
    def enc_out_shape(self):
        return self._enc_out_shape

    @property
    def flatten_dim(self):
        return self._flatten_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)


class DenseDecoderVAE(nn.Module, _WithLogger):
    """Decoder turning latent vector into reconstruction."""
    def __init__(self,
                 out_channels: int,
                 latent_dim: int,
                 enc_out_shape: Sequence[int],
                 num_blocks: int = 3,
                 channels: Optional[Sequence[int]] = None,
                 activation: nn.Module = nn.ReLU,
                 logger: Optional[Any] = None,
                 **kwargs):
        super().__init__()
        self._init_logger(logger, "DenseDecoderVAE")
        if channels is None:
            channels = [32 * (2 ** i) for i in range(num_blocks)]
        decoder_channels = list(reversed(channels))
        self.enc_out_shape = tuple(enc_out_shape)
        self.flatten_dim = int(torch.prod(torch.tensor(enc_out_shape)))
        self.fc = nn.Linear(latent_dim, self.flatten_dim)
        layers = [nn.Unflatten(1, self.enc_out_shape)]
        prev = decoder_channels[0]
        for ch in decoder_channels[1:]:
            layers += [nn.ConvTranspose2d(prev, ch, 2, stride=2), activation()]
            prev = ch
        layers += [nn.ConvTranspose2d(prev, out_channels, 2, stride=2), nn.Sigmoid()]
        self.deconv = nn.Sequential(*layers)
        self.logger.info("DenseDecoderVAE initialized enc_out_shape=%s", self.enc_out_shape)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        return self.deconv(x)


class DenseAutoencoderVAE(nn.Module, _WithLogger):
    """Variational dense autoencoder with reparameterization and sampling helpers."""
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_shape: Sequence[int],
                 num_blocks: int = 3,
                 channels: Optional[Sequence[int]] = None,
                 activation: nn.Module = nn.ReLU,
                 logger: Optional[Any] = None,
                 **kwargs):
        super().__init__()
        self._init_logger(logger, "DenseAutoencoderVAE")
        self.encoder = DenseEncoderVAE(in_channels, latent_dim, input_shape,
                                       num_blocks=num_blocks, channels=channels,
                                       activation=activation, logger=self.logger)
        self.decoder = DenseDecoderVAE(in_channels, latent_dim, self.encoder.enc_out_shape,
                                       num_blocks=num_blocks, channels=channels,
                                       activation=activation, logger=self.logger)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    # Sampling utilities
    def sample_prior(self, n: int, device: Optional[str] = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        latent_dim = self.encoder.fc_mu.out_features
        z = torch.randn(n, latent_dim, device=device)
        return self.decoder(z)

    def sample_posterior(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Return n_samples reconstructions by sampling posterior q(z|x)."""
        mu, logvar = self.encoder(x)
        outs = []
        for _ in range(n_samples):
            z = self.reparameterize(mu, logvar)
            outs.append(self.decoder(z))
        return torch.stack(outs, dim=1)  # (B, n_samples, C, H, W)


def dense_vae_loss(output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   target: torch.Tensor,
                   beta: float = 1.0,
                   reduction: str = "mean") -> torch.Tensor:
    """Standard VAE loss: reconstruction + beta * KL."""
    recon, mu, logvar = output
    if reduction == "sum":
        recon_loss = F.mse_loss(recon, target, reduction='sum')
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        recon_loss = F.mse_loss(recon, target, reduction='mean')
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_autoencoder(kind: Literal['fcn', 'dense', 'dense_vae'],
                       **kwargs) -> nn.Module:
    """Factory to create different autoencoder variants."""
    k = kind.lower()
    if k == 'fcn':
        return FullyConvAutoencoder(**kwargs)
    if k == 'dense':
        return DenseAutoencoder(**kwargs)
    if k == 'dense_vae':
        return DenseAutoencoderVAE(**kwargs)
    raise ValueError(f"Unknown kind '{kind}'. Expected one of: fcn, dense, dense_vae.")


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# (Old names kept so existing notebooks keep working)
# ---------------------------------------------------------------------------
DenoisingAutoencoder = DenseAutoencoder
VAE_Encoder = DenseEncoderVAE
VAE_Decoder = DenseDecoderVAE
VAE = DenseAutoencoderVAE
vae_loss = dense_vae_loss


__all__ = [
    # Utilities
    "count_parameters", "model_size_mb",
    # Fully conv parts
    "FullyConvEncoder", "FullyConvDecoder", "FullyConvAutoencoder",
    # Dense parts
    "DenseEncoder", "DenseDecoder", "DenseAutoencoder",
    # Dense VAE parts
    "DenseEncoderVAE", "DenseDecoderVAE", "DenseAutoencoderVAE",
    # Loss / factory
    "dense_vae_loss", "create_autoencoder",
    # Aliases
    "DenoisingAutoencoder", "VAE_Encoder", "VAE_Decoder", "VAE", "vae_loss"
]