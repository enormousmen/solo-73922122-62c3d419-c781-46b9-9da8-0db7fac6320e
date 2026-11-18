"""Two-layer fully-connected classifier with Dropout."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_model(input_dim: int, hidden_dim: int, dropout_p: float, num_classes: int,
                device: str | torch.device = "cpu") -> nn.Sequential:
    """Build the classifier.

    ``device="meta"`` lets the learning-rate probe construct an architecture
    without consuming the global torch RNG stream (nothing is initialized);
    the caller then materializes it with ``to_empty`` and loads weights.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim, device=device),
        nn.ReLU(),
        nn.Dropout(dropout_p),
        nn.Linear(hidden_dim, num_classes, device=device),
    )
