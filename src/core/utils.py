# src/utils.py
from __future__ import annotations
from typing import Any, Dict

__all__ = ["PML_kw", "PML_effective_gn_groups"]


def PML_kw(c_key: str, dic_kwargs: Dict[str, Any], v_default: Any) -> Any:
    """
    Universal kwarg getter.

    Args:
        c_key: key to look up in kwargs
        dic_kwargs: kwargs dict
        v_default: default value if key missing

    Returns:
        Any: dic_kwargs.get(c_key, v_default)
    """
    return dic_kwargs.get(c_key, v_default)


def PML_effective_gn_groups(n_channels: int, n_groups_requested: int) -> int:
    """
    Choose a GroupNorm group count that cleanly divides n_channels.
    Falls back to 1 if necessary.

    Args:
        n_channels: number of channels in the tensor to be normalized
        n_groups_requested: preferred number of groups

    Returns:
        int: valid group count (>=1) that divides n_channels
    """
    n_g = min(int(n_groups_requested), int(n_channels))
    while n_g > 1 and (n_channels % n_g != 0):
        n_g -= 1
    return max(n_g, 1)


