"""
MLP Fusion — Multi-Layer Perceptron fusion of vision and language embeddings.

Architecture
------------
    concat(v, t) → FC → Act → Dropout → FC → Act → ... → (B, embed_dim)

Configurable depth, activation, and dropout.  Lightweight and fast;
serves as a strong classical baseline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from fusion_core.fusion_methods.base import BaseFusion

logger = logging.getLogger(__name__)

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


class MLPFusion(BaseFusion):
    """Multi-Layer Perceptron fusion.

    Parameters
    ----------
    vision_dim : int
        Dimensionality of the raw vision embedding.
    text_dim : int
        Dimensionality of the raw text embedding.
    embed_dim : int
        Common projection / output dimensionality of the fusion block.
    output_dim : int
        Final fused output dimensionality.
    hidden_dim : int
        Width of hidden layers inside the MLP.
    num_layers : int
        Number of FC layers (minimum 1).  A single layer degenerates to a
        linear fusion.
    activation : str
        One of ``{"relu", "gelu", "tanh", "silu"}``.
    dropout : float
        Dropout probability applied after each activation.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            vision_dim=vision_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            output_dim=output_dim,
            name="mlp",
        )
        self.hidden_dim = hidden_dim
        self.num_layers = max(num_layers, 1)
        self.activation_name = activation
        self.dropout_p = dropout

        act_cls = _ACTIVATIONS.get(activation, nn.GELU)

        # Build MLP:  2*embed_dim (concat) → hidden → ... → embed_dim
        layers: list[nn.Module] = []
        in_dim = embed_dim * 2  # concat vision + text
        for i in range(self.num_layers):
            out = embed_dim if i == self.num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out))
            if i < self.num_layers - 1:
                layers.append(act_cls())
                layers.append(nn.Dropout(dropout))
            in_dim = out

        self.mlp = nn.Sequential(*layers)

        logger.info(
            "MLPFusion init — layers=%d, hidden=%d, act=%s, params=%s",
            self.num_layers, hidden_dim, activation, f"{self.param_count:,}",
        )

    # ── core fusion ──────────────────────────────────────────────────

    def _fuse(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate and pass through MLP.  Returns ``(B, embed_dim)``."""
        x = torch.cat([vision_embed, text_embed], dim=-1)  # (B, 2*embed_dim)
        return self.mlp(x)                                   # (B, embed_dim)

    # ── meta ─────────────────────────────────────────────────────────

    def _extra_meta(self) -> Dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "activation": self.activation_name,
            "dropout": self.dropout_p,
        }
