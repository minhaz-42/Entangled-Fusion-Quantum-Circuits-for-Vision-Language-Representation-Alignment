"""
Classical Fallback Fusion — lightweight non-parametric baselines.

Provides three strategies that require *no* learning (useful for ablation):

1. **concat_tanh** — ``tanh(W·[v; t])``
2. **hadamard**   — element-wise product ``v ⊙ t``
3. **avg**        — simple mean ``(v + t) / 2``

When PennyLane is unavailable the framework can route through this module
transparently, guaranteeing the rest of the pipeline never breaks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from fusion_core.fusion_methods.base import BaseFusion

logger = logging.getLogger(__name__)


class ClassicalFallbackFusion(BaseFusion):
    """Minimal classical fusion for baselines and fallback.

    Parameters
    ----------
    method : str
        ``"concat_tanh"`` | ``"hadamard"`` | ``"avg"``.
    """

    VALID_METHODS = {"concat_tanh", "hadamard", "avg"}

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        output_dim: int,
        *,
        method: str = "concat_tanh",
    ) -> None:
        super().__init__(
            vision_dim=vision_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            output_dim=output_dim,
            name="classical_fallback",
        )
        if method not in self.VALID_METHODS:
            raise ValueError(f"Unknown method '{method}', choose from {self.VALID_METHODS}")

        self.method = method

        if method == "concat_tanh":
            self.linear = nn.Linear(embed_dim * 2, embed_dim)
        else:
            # hadamard / avg need no extra params
            self.linear = None

        self.ln = nn.LayerNorm(embed_dim)

        logger.info(
            "ClassicalFallbackFusion init — method=%s, params=%s",
            method, f"{self.param_count:,}",
        )

    # ── core fusion ──────────────────────────────────────────────────

    def _fuse(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        if self.method == "concat_tanh":
            x = torch.cat([vision_embed, text_embed], dim=-1)
            x = torch.tanh(self.linear(x))
        elif self.method == "hadamard":
            x = vision_embed * text_embed
        elif self.method == "avg":
            x = (vision_embed + text_embed) / 2.0
        else:
            raise RuntimeError(f"Unknown method: {self.method}")

        return self.ln(x)

    # ── meta ─────────────────────────────────────────────────────────

    def _extra_meta(self) -> Dict[str, Any]:
        return {"method": self.method}
