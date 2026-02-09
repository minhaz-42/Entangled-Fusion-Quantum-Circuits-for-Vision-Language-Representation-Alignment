"""
Bilinear & Tensor Fusion — Higher-order interaction models.

Implements two variants:

1. **Low-Rank Bilinear Fusion** (``BilinearFusion``)
       f = σ( (U·v) ⊙ (V·t) ) W + b
   where U, V project to a low-rank space and ⊙ is Hadamard product.
   Inspired by LXMERT / Tucker decomposition.

2. **Tensor Fusion Network** (``TensorFusion``)
       f = MLP( outer(v̄, t̄).flatten )
   where v̄ = [v; 1], t̄ = [t; 1] (bias trick) and the outer product
   captures all pairwise interactions.  Uses a learnable rank-reduction
   to keep dimensionality tractable.

References
----------
- Ben-Younes et al., "MUTAN: Multimodal Tucker Fusion", ICCV 2017.
- Zadeh et al., "Tensor Fusion Network for Multimodal Sentiment Analysis",
  EMNLP 2017.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_core.fusion_methods.base import BaseFusion

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  1.  Low-Rank Bilinear Fusion
# ═══════════════════════════════════════════════════════════════════════

class BilinearFusion(BaseFusion):
    """Low-rank bilinear pooling for vision–language fusion.

    Parameters
    ----------
    rank : int
        Rank of the bilinear projections.  Lower → fewer params,
        higher → more expressive.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        output_dim: int,
        *,
        rank: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            vision_dim=vision_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            output_dim=output_dim,
            name="bilinear",
        )
        self.rank = rank
        self.dropout_p = dropout

        # Low-rank projections
        self.U = nn.Linear(embed_dim, rank, bias=False)
        self.V = nn.Linear(embed_dim, rank, bias=False)

        # Project back from rank → embed_dim
        self.W = nn.Linear(rank, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

        logger.info(
            "BilinearFusion init — rank=%d, params=%s",
            rank, f"{self.param_count:,}",
        )

    def _fuse(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Low-rank Hadamard product.  Returns ``(B, embed_dim)``."""
        u = torch.tanh(self.U(vision_embed))  # (B, rank)
        v = torch.tanh(self.V(text_embed))    # (B, rank)
        z = u * v                              # Hadamard  (B, rank)
        z = self.drop(z)
        out = self.W(z)                        # (B, embed_dim)
        out = self.ln(out + (vision_embed + text_embed) / 2.0)  # residual
        return out

    def _extra_meta(self) -> Dict[str, Any]:
        return {"rank": self.rank}


# ═══════════════════════════════════════════════════════════════════════
#  2.  Tensor Fusion Network
# ═══════════════════════════════════════════════════════════════════════

class TensorFusion(BaseFusion):
    """Tensor Fusion Network (full outer-product with rank reduction).

    The raw outer product of (embed_dim+1) × (embed_dim+1) vectors is
    flattened and projected through a small MLP to ``embed_dim``.

    Parameters
    ----------
    hidden_dim : int
        Hidden size of the rank-reduction MLP.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            vision_dim=vision_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            output_dim=output_dim,
            name="tensor",
        )
        self.tfn_hidden = hidden_dim
        outer_size = (embed_dim + 1) ** 2  # with bias trick

        self.rank_reduction = nn.Sequential(
            nn.Linear(outer_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.ln = nn.LayerNorm(embed_dim)

        logger.info(
            "TensorFusion init — outer_size=%d, hidden=%d, params=%s",
            outer_size, hidden_dim, f"{self.param_count:,}",
        )

    def _fuse(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Full outer product → rank-reduce → (B, embed_dim)."""
        B = vision_embed.size(0)

        # Bias trick: append 1
        v_bar = torch.cat(
            [vision_embed, torch.ones(B, 1, device=vision_embed.device)], dim=-1,
        )  # (B, D+1)
        t_bar = torch.cat(
            [text_embed, torch.ones(B, 1, device=text_embed.device)], dim=-1,
        )  # (B, D+1)

        # Outer product → flatten
        outer = torch.bmm(v_bar.unsqueeze(2), t_bar.unsqueeze(1))  # (B, D+1, D+1)
        outer_flat = outer.view(B, -1)  # (B, (D+1)^2)

        out = self.rank_reduction(outer_flat)  # (B, embed_dim)
        out = self.ln(out)
        return out

    def _extra_meta(self) -> Dict[str, Any]:
        return {"hidden_dim": self.tfn_hidden}
