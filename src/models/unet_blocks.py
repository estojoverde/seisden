# src/models/unet_blocks.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import PML_kw

__all__ = [
    "PML_safe_group_count",
    "PML_ResidualBlock",
    "PML_AttentionBlock",
    "PML_UNet",
]


# ---------------------------
# Helpers
# ---------------------------

def PML_safe_group_count(n_channels: int, n_groups_wanted: int) -> int:
    """
    Choose a GroupNorm group count that divides n_channels.
    """
    n_groups = max(1, min(int(n_groups_wanted), int(n_channels)))
    while n_channels % n_groups != 0 and n_groups > 1:
        n_groups -= 1
    return n_groups


def _match_size_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Crop or pad x (right/bottom) to match spatial size of ref. Avoids in-place ops.
    """
    _, _, h, w = x.shape
    _, _, hr, wr = ref.shape
    dh = hr - h
    dw = wr - w
    if dh == 0 and dw == 0:
        return x
    # Pad if needed (right, bottom)
    pad_right = max(0, dw)
    pad_bottom = max(0, dh)
    if pad_right or pad_bottom:
        x = F.pad(x, (0, pad_right, 0, pad_bottom), mode="replicate")
        _, _, h, w = x.shape
    # Crop if over
    if h > hr or w > wr:
        x = x[:, :, :hr, :wr]
    return x


def _broadcast_film(
    gamma: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    y: torch.Tensor
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Ensure gamma/beta are shaped (B,C,1,1) to modulate y (B,C,H,W).
    Accept (B,C) or (B,C,1,1) or (C,) (broadcasted to batch).
    """
    if gamma is None and beta is None:
        return None, None

    B, C, _, _ = y.shape

    def _fix(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape == (B, C):
            return t.unsqueeze(-1).unsqueeze(-1)
        if t.ndim == 4 and t.shape[:2] == (B, C):
            return t
        if t.ndim == 1 and t.shape[0] == C:
            return t.view(1, C, 1, 1).expand(B, -1, -1, -1)
        raise ValueError(f"FiLM gamma/beta has incompatible shape {tuple(t.shape)} for y {tuple(y.shape)}")

    g = _fix(gamma) if gamma is not None else None
    b = _fix(beta) if beta is not None else None
    return g, b


# ---------------------------
# Blocks
# ---------------------------

class PML_ResidualBlock(nn.Module):
    """
    Residual block: Conv2d -> GN -> (FiLM?) -> SiLU -> Dropout ->
                    Conv2d -> GN -> (FiLM?) -> SiLU, with skip projection if needed.

    Required args:
        n_in:  in-channels
        n_out: out-channels

    Optional **kwargs:
        n_groups_gn (int) = 8
        f_dropout (float) = 0.0
        ksize (int) = 3
        c_padding (str) = "same"   # "same" -> padding=ksize//2, else explicit via n_ksize_pad
        n_ksize_pad (int|None) = None

    FiLM (per forward via **kwargs**):
        # direct tensors (any of these work)
        gamma / beta              -> used for both layers
        gamma1 / beta1, gamma2 / beta2

        # or a dict container
        dic_film = { "gamma1":..., "beta1":..., "gamma2":..., "beta2":... }
                 or { "gamma":..., "beta":... }  # applied to both layers

        # or a module + conditioning vector
        film / film_module / conditioner  : a FiLM module
        h_cond / cond / c_cond / metadata : (B,D) conditioning vector

        We try these call signatures (in order) when invoking the module:
          film(h, n_channels=C, n_layer=L)
          film(h, n_channels=C)
          film(h, n_layer=L)
          film(h)
          film(h, C, L)                # positional fallbacks
          film(h, C)
          film(h, L)
    """

    def __init__(self, n_in: int, n_out: int, **kwargs: Any) -> None:
        super().__init__()
        n_groups_gn = int(PML_kw("n_groups_gn", kwargs, 8))
        f_dropout = float(PML_kw("f_dropout", kwargs, 0.0))
        ksize = int(PML_kw("ksize", kwargs, 3))
        c_padding = PML_kw("c_padding", kwargs, "same")
        n_ksize_pad = PML_kw("n_ksize_pad", kwargs, None)

        if c_padding == "same":
            pad = ksize // 2
        else:
            pad = int(n_ksize_pad) if n_ksize_pad is not None else 0

        n_g1 = PML_safe_group_count(n_out, n_groups_gn)
        n_g2 = PML_safe_group_count(n_out, n_groups_gn)

        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=ksize, padding=pad, bias=False)
        self.gn1 = nn.GroupNorm(n_g1, n_out)
        self.act1 = nn.SiLU(inplace=False)
        self.drop = nn.Dropout2d(p=f_dropout) if f_dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=ksize, padding=pad, bias=False)
        self.gn2 = nn.GroupNorm(n_g2, n_out)
        self.act2 = nn.SiLU(inplace=False)

        self.proj = nn.Conv2d(n_in, n_out, kernel_size=1, bias=False) if n_in != n_out else nn.Identity()

    @staticmethod
    def _resolve_conditioning(**kwargs: Any) -> Tuple[Optional[Any], Optional[torch.Tensor]]:
        """
        Resolve (film_module, h_cond) from flexible kwargs naming.
        """
        film = kwargs.get("film", None) or kwargs.get("film_module", None) or kwargs.get("conditioner", None)
        h = kwargs.get("h_cond", None) or kwargs.get("cond", None) or kwargs.get("c_cond", None) or kwargs.get("metadata", None)
        return film, h

    @staticmethod
    def _get_film_from_kwargs(n_layer: int, C: int, **kwargs: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Resolve (gamma, beta) for a given layer and channel count C from:
          - direct kwargs gamma/beta (and layer-specific gamma1/beta1, gamma2/beta2)
          - dic_film dict (layer-specific keys or shared)
          - film module + conditioning vector (robust call attempts)
        """
        # 0) direct per-layer overrides
        g_direct = kwargs.get(f"gamma{n_layer}", None)
        b_direct = kwargs.get(f"beta{n_layer}", None)
        if g_direct is not None or b_direct is not None:
            return g_direct, b_direct

        # 1) direct shared overrides
        g_shared = kwargs.get("gamma", None)
        b_shared = kwargs.get("beta", None)
        if g_shared is not None or b_shared is not None:
            return g_shared, b_shared

        # 2) explicit dict
        dic_film: Optional[Dict[str, torch.Tensor]] = kwargs.get("dic_film", None)
        if dic_film is not None:
            g = dic_film.get(f"gamma{n_layer}", dic_film.get("gamma", None))
            b = dic_film.get(f"beta{n_layer}", dic_film.get("beta", None))
            if g is not None or b is not None:
                return g, b

        # 3) film module + h_cond
        film, h = PML_ResidualBlock._resolve_conditioning(**kwargs)
        if film is not None and h is not None and hasattr(film, "forward"):
            # Try rich signatures & positional fallbacks
            try:
                return film(h, n_channels=C, n_layer=n_layer)
            except TypeError:
                pass
            try:
                return film(h, n_channels=C)
            except TypeError:
                pass
            try:
                return film(h, n_layer=n_layer)
            except TypeError:
                pass
            try:
                return film(h)
            except TypeError:
                pass
            # positional
            try:
                return film(h, C, n_layer)
            except TypeError:
                pass
            try:
                return film(h, C)
            except TypeError:
                pass
            try:
                return film(h, n_layer)
            except TypeError:
                pass

        return None, None

    def _apply_film_if_any(self, y: torch.Tensor, n_layer: int, **kwargs: Any) -> torch.Tensor:
        gamma, beta = self._get_film_from_kwargs(n_layer, y.shape[1], **kwargs)
        if gamma is None and beta is None:
            return y
        g, b = _broadcast_film(gamma, beta, y)
        if g is not None:
            y = y * g
        if b is not None:
            y = y + b
        return y

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        residual = self.proj(x)

        y = self.conv1(x)
        y = self.gn1(y)
        y = self._apply_film_if_any(y, n_layer=1, **kwargs)
        y = self.act1(y)

        y = self.drop(y)
        y = self.conv2(y)
        y = self.gn2(y)
        y = self._apply_film_if_any(y, n_layer=2, **kwargs)
        y = self.act2(y)

        return y + residual


class PML_AttentionBlock(nn.Module):
    """
    Lightweight spatial self-attention over HxW tokens (per batch, per channel group).
    """

    def __init__(self, n_channels: int, **kwargs: Any) -> None:
        super().__init__()
        n_heads = int(PML_kw("n_heads", kwargs, 1))
        f_dropout = float(PML_kw("f_dropout", kwargs, 0.0))
        n_qkv = int(PML_kw("n_channels_qkv", kwargs, n_channels))

        self.n_channels = int(n_channels)
        self.n_heads = max(1, n_heads)
        self.scale = (n_qkv // self.n_heads) ** -0.5 if (n_qkv // self.n_heads) > 0 else 1.0

        self.to_q = nn.Conv2d(n_channels, n_qkv, 1, bias=False)
        self.to_k = nn.Conv2d(n_channels, n_qkv, 1, bias=False)
        self.to_v = nn.Conv2d(n_channels, n_qkv, 1, bias=False)
        self.proj = nn.Conv2d(n_qkv, n_channels, 1, bias=False)
        self.drop = nn.Dropout(p=f_dropout) if f_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        def _reshape_heads(t: torch.Tensor) -> torch.Tensor:
            b, c, h, w = t.shape
            return t.view(b, self.n_heads, c // self.n_heads, h * w)

        qh = _reshape_heads(q)
        kh = _reshape_heads(k)
        vh = _reshape_heads(v)

        attn = torch.einsum("bhcn,bhcm->bhnm", qh * self.scale, kh)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.einsum("bhnm,bhcm->bhcn", attn, vh)
        out = out.reshape(b, -1, h, w)
        out = self.proj(out)
        return out + x


class PML_UNet(nn.Module):
    """
    Configurable U-Net with residual blocks and optional mid-resolution attention.
    """

    def __init__(self, n_in: int, n_out: int, **kwargs: Any) -> None:
        super().__init__()
        l_n_channels = list(PML_kw("l_n_channels", kwargs, [64, 128, 256, 512]))
        n_groups_gn = int(PML_kw("n_groups_gn", kwargs, 8))
        b_use_attention = bool(PML_kw("b_use_attention", kwargs, True))
        f_dropout = float(PML_kw("f_dropout", kwargs, 0.0))
        c_upsample = PML_kw("c_upsample", kwargs, "interp")
        b_final_activation = bool(PML_kw("b_final_activation", kwargs, False))

        ksize = int(PML_kw("ksize", kwargs, 3))
        c_padding = PML_kw("c_padding", kwargs, "same")
        n_ksize_pad = PML_kw("n_ksize_pad", kwargs, None)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        n_prev = n_in
        for n_ch in l_n_channels:
            self.enc_blocks.append(
                PML_ResidualBlock(
                    n_prev, n_ch,
                    n_groups_gn=n_groups_gn,
                    f_dropout=f_dropout,
                    ksize=ksize, c_padding=c_padding, n_ksize_pad=n_ksize_pad,
                )
            )
            self.downs.append(nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=2, padding=1, bias=False))
            n_prev = n_ch

        # Bottleneck
        self.mid_block1 = PML_ResidualBlock(
            l_n_channels[-1], l_n_channels[-1],
            n_groups_gn=n_groups_gn, f_dropout=f_dropout,
            ksize=ksize, c_padding=c_padding, n_ksize_pad=n_ksize_pad,
        )
        self.attn = PML_AttentionBlock(l_n_channels[-1]) if b_use_attention else nn.Identity()
        self.mid_block2 = PML_ResidualBlock(
            l_n_channels[-1], l_n_channels[-1],
            n_groups_gn=n_groups_gn, f_dropout=f_dropout,
            ksize=ksize, c_padding=c_padding, n_ksize_pad=n_ksize_pad,
        )

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(len(l_n_channels))):
            n_ch = l_n_channels[i]
            in_ch_up = l_n_channels[i if i == len(l_n_channels)-1 else i+1]
            if c_upsample == "deconv":
                self.ups.append(nn.ConvTranspose2d(in_ch_up, n_ch, 2, 2))
            else:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(in_ch_up, n_ch, kernel_size=1, bias=False),
                    )
                )
            self.dec_blocks.append(
                PML_ResidualBlock(
                    n_ch * 2, n_ch,
                    n_groups_gn=n_groups_gn,
                    f_dropout=f_dropout,
                    ksize=ksize, c_padding=c_padding, n_ksize_pad=n_ksize_pad,
                )
            )

        # Head
        self.head = nn.Conv2d(l_n_channels[0], n_out, kernel_size=1, bias=True)
        self.head_act = nn.SiLU(inplace=False) if b_final_activation else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Encoder
        skips: List[torch.Tensor] = []
        y = x
        for block, down in zip(self.enc_blocks, self.downs):
            y = block(y, **kwargs)
            skips.append(y)
            y = down(y)

        # Bottleneck
        y = self.mid_block1(y, **kwargs)
        y = self.attn(y, **kwargs)
        y = self.mid_block2(y, **kwargs)

        # Decoder
        for up, block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            y = up(y)
            y = _match_size_like(y, skip)
            y = torch.cat([y, skip], dim=1)
            y = block(y, **kwargs)

        y = self.head(y)
        y = self.head_act(y)
        y = _match_size_like(y, x)
        return y
