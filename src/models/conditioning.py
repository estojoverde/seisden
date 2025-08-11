# src/models/conditioning.py
from __future__ import annotations

from typing import Dict, List, Sequence, Any
import torch
from torch import nn

from ..utils import PML_kw

__all__ = ["PML_MetadataConditioner", "PML_FiLM"]


class PML_MetadataConditioner(nn.Module):
    r"""
    Build a conditioning vector z_meta from a dict of geophysical metadata.

    Typical inputs (examples):
      - f_peak        : peak source wavelet frequency [Hz]
      - f_flow        : low-cut frequency [Hz]
      - f_dt          : sample interval [s]
      - f_d_spatial   : spatial sample interval [m] or [trace index units]
      - (optionally others, see kwargs)

    Required:
        l_keys (Sequence[str]):
            Ordered list of metadata keys to extract from the input dict.

    Optional **kwargs (typed prefixes):
        c_transform (str) = "auto"
            How to transform numeric values: "auto"|"identity"|"log1p".
            - "auto" uses "log1p" for keys starting with "f_" else "identity".
        f_eps (float) = 1e-6
            Stabilizer for log1p.
        b_keep_dims (bool) = False
            If True, returns shape (B, len(l_keys), 1, 1) instead of (B, len(l_keys)).

    Forward:
        dic_meta: Dict[str, torch.Tensor | float | int]
            For each key in l_keys:
              - torch.Tensor shape (B,) or (B,1) or scalar, will be broadcast to (B,)
              - python float/int scalars allowed if consistent across batch

    Returns:
        z_meta: (B, n_feats) or (B, n_feats, 1, 1) if b_keep_dims=True
    """
    def __init__(self, l_keys: Sequence[str], **kwargs) -> None:
        super().__init__()
        self.l_keys: List[str] = list(l_keys)
        self.c_transform: str = PML_kw("c_transform", kwargs, "auto")
        self.f_eps: float = float(PML_kw("f_eps", kwargs, 1e-6))
        self.b_keep_dims: bool = bool(PML_kw("b_keep_dims", kwargs, False))

    def _transform_scalar(self, c_key: str, t: torch.Tensor) -> torch.Tensor:
        if self.c_transform == "identity":
            return t
        if self.c_transform == "log1p":
            return torch.log1p(torch.abs(t) + self.f_eps) * torch.sign(t)
        # auto
        if c_key.startswith("f_"):
            return torch.log1p(torch.abs(t) + self.f_eps) * torch.sign(t)
        return t

    def forward(self, dic_meta: Dict[str, Any]) -> torch.Tensor:
        # Determine batch size B from first tensor-like value
        n_B = None
        l_vals: List[torch.Tensor] = []
        for c_key in self.l_keys:
            v = dic_meta.get(c_key, None)
            if v is None:
                raise KeyError(f"[PML_MetadataConditioner] Missing key '{c_key}' in dic_meta.")
            if not torch.is_tensor(v):
                v = torch.as_tensor(v, dtype=torch.float32)
            if v.ndim == 0:
                v = v.view(1)  # scalar -> (1,)
            if v.ndim == 2 and v.shape[1] == 1:
                v = v.view(v.shape[0])  # (B,1) -> (B,)
            if n_B is None:
                n_B = v.shape[0]
            elif v.shape[0] != n_B:
                # Broadcast if v is scalar (1,) else error
                if v.numel() == 1:
                    v = v.expand(n_B)
                else:
                    raise ValueError(f"[PML_MetadataConditioner] Batch mismatch for key '{c_key}'.")
            l_vals.append(self._transform_scalar(c_key, v.to(dtype=torch.float32)))

        assert n_B is not None, "Empty l_keys in PML_MetadataConditioner."
        z = torch.stack(l_vals, dim=1)  # (B, n_feats)
        if self.b_keep_dims:
            z = z.view(n_B, len(self.l_keys), 1, 1)
        return z


class PML_FiLM(nn.Module):
    r"""
    Feature-wise Linear Modulation (FiLM) head.

    Produces per-channel (γ, β) pairs for two normalization points within a
    residual block (i.e., γ1/β1 after GN1, γ2/β2 after GN2).

    Required:
        n_in (int):  size of conditioning vector z_meta (last dimension)
        n_out (int): number of channels to modulate (block's output channels)

    Optional **kwargs (typed prefixes):
        n_hidden (int) = 128      # hidden size for MLP
        n_layers (int) = 2        # number of hidden layers (>=0)
        f_dropout (float) = 0.0   # dropout in hidden layers
        b_init_gamma_to_one (bool) = True
            If True, initialize gamma heads to output ~0 so we return (1 + 0) at start.
        b_init_beta_to_zero (bool) = True
            If True, initialize beta heads to 0 at start.
        c_activation (str) = "silu"  # "silu" | "gelu" | "relu"

    Forward:
        z_meta: (B, n_in) or (B, n_in, 1, 1)

    Returns:
        gamma1, beta1, gamma2, beta2:
            each shaped (B, n_out, 1, 1)
    """
    def __init__(self, n_in: int, n_out: int, **kwargs) -> None:
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)

        n_hidden = int(PML_kw("n_hidden", kwargs, 128))
        n_layers = int(PML_kw("n_layers", kwargs, 2))
        f_dropout = float(PML_kw("f_dropout", kwargs, 0.0))
        self.b_init_gamma_to_one = bool(PML_kw("b_init_gamma_to_one", kwargs, True))
        self.b_init_beta_to_zero = bool(PML_kw("b_init_beta_to_zero", kwargs, True))
        c_activation = PML_kw("c_activation", kwargs, "silu").lower()

        if c_activation == "relu":
            act = nn.ReLU()
        elif c_activation == "gelu":
            act = nn.GELU()
        else:
            act = nn.SiLU()

        layers: List[nn.Module] = []
        n_prev = self.n_in
        for _ in range(max(0, n_layers)):
            layers += [nn.Linear(n_prev, n_hidden), act]
            if f_dropout > 0:
                layers += [nn.Dropout(p=f_dropout)]
            n_prev = n_hidden
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        # Four heads for gamma1, beta1, gamma2, beta2
        self.head_gamma1 = nn.Linear(n_prev, self.n_out)
        self.head_beta1  = nn.Linear(n_prev, self.n_out)
        self.head_gamma2 = nn.Linear(n_prev, self.n_out)
        self.head_beta2  = nn.Linear(n_prev, self.n_out)

        # --- NEW: ensure z_meta == 0 -> backbone output == 0 ---
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

        # Init so that at start: gamma ≈ 1, beta ≈ 0 for z_meta == 0
        # Keep default (Kaiming) weights so outputs depend on z_meta.
        if self.b_init_gamma_to_one:
            nn.init.zeros_(self.head_gamma1.bias)
            nn.init.zeros_(self.head_gamma2.bias)
        if self.b_init_beta_to_zero:
            nn.init.zeros_(self.head_beta1.bias)
            nn.init.zeros_(self.head_beta2.bias)
        # NOTE: Do NOT zero the weights — leave default init to enable modulation.


    def forward(self, z_meta: torch.Tensor):
        if z_meta.ndim == 4:
            # (B, n_in, 1, 1) -> (B, n_in)
            z_meta = z_meta.view(z_meta.shape[0], z_meta.shape[1])
        if z_meta.ndim != 2 or z_meta.shape[1] != self.n_in:
            raise ValueError(f"[PML_FiLM] Expected z_meta (B, {self.n_in}), got {tuple(z_meta.shape)}")

        h = self.backbone(z_meta)  # (B, H) or (B, n_in)
        g1 = self.head_gamma1(h)   # (B, n_out)
        b1 = self.head_beta1(h)
        g2 = self.head_gamma2(h)
        b2 = self.head_beta2(h)

        if self.b_init_gamma_to_one:
            g1 = 1.0 + g1
            g2 = 1.0 + g2

        # reshape to (B, C, 1, 1)
        g1 = g1.view(g1.shape[0], self.n_out, 1, 1)
        b1 = b1.view(b1.shape[0], self.n_out, 1, 1)
        g2 = g2.view(g2.shape[0], self.n_out, 1, 1)
        b2 = b2.view(b2.shape[0], self.n_out, 1, 1)
        return g1, b1, g2, b2
