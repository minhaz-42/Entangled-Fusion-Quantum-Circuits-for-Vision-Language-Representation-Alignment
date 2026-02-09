"""
BaseFusion — Abstract interface for all fusion methods.

Every fusion module in the EFC framework *must* subclass ``BaseFusion`` so
that benchmarking, ablation, and comparison code can treat all methods
uniformly.

Design Contracts
----------------
* ``forward(vision_embed, text_embed) -> fused_embed``
    - ``vision_embed``:  ``(B, D_v)``  or  ``(D_v,)``
    - ``text_embed``:    ``(B, D_t)``  or  ``(D_t,)``
    - ``fused_embed``:   ``(B, D_out)`` or ``(D_out,)``
    Both inputs are projected to a common ``embed_dim`` inside the module.

* ``meta()`` returns a dict with at least:
    ``{"name", "param_count", "last_forward_ms", "extra"}``
"""

from __future__ import annotations

import abc
import time
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseFusion(nn.Module, abc.ABC):
    """Abstract base class for all fusion methods.

    Subclasses **must** implement:
        ``_fuse(vision_embed, text_embed) -> Tensor``

    The public ``forward`` wraps ``_fuse`` with automatic latency measurement,
    input validation, and optional projection layers.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        output_dim: int,
        *,
        name: str = "base",
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self._name = name

        # Projection heads to unify input dimensions  ─────────────────
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # Output projection  ──────────────────────────────────────────
        self.output_proj = nn.Linear(embed_dim, output_dim)

        # Timing bookkeeping
        self._last_forward_ms: float = 0.0

    # ── public interface ─────────────────────────────────────────────

    def forward(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Project → fuse → project-out.  Records latency."""
        # Ensure batch dimension
        squeeze = False
        if vision_embed.dim() == 1:
            vision_embed = vision_embed.unsqueeze(0)
            text_embed = text_embed.unsqueeze(0)
            squeeze = True

        # Project both modalities to common embed_dim
        v = self.vision_proj(vision_embed)   # (B, embed_dim)
        t = self.text_proj(text_embed)       # (B, embed_dim)

        t0 = time.perf_counter()
        fused = self._fuse(v, t)             # (B, embed_dim)
        self._last_forward_ms = (time.perf_counter() - t0) * 1000.0

        out = self.output_proj(fused)        # (B, output_dim)

        if squeeze:
            out = out.squeeze(0)
        return out

    @abc.abstractmethod
    def _fuse(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Core fusion operation.  Must return ``(B, embed_dim)``."""
        ...

    # ── introspection helpers ────────────────────────────────────────

    @property
    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def last_forward_ms(self) -> float:
        return self._last_forward_ms

    def meta(self) -> Dict[str, Any]:
        """Return a JSON-serialisable metadata dict for logging."""
        return {
            "name": self._name,
            "param_count": self.param_count,
            "vision_dim": self.vision_dim,
            "text_dim": self.text_dim,
            "embed_dim": self.embed_dim,
            "output_dim": self.output_dim,
            "last_forward_ms": round(self._last_forward_ms, 4),
            "extra": self._extra_meta(),
        }

    def _extra_meta(self) -> Dict[str, Any]:
        """Override in subclass to add method-specific metadata."""
        return {}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"v={self.vision_dim}, t={self.text_dim}, "
            f"e={self.embed_dim}, o={self.output_dim}, "
            f"params={self.param_count:,})"
        )
