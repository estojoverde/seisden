
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Approximate model size in MB (parameters only)."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
