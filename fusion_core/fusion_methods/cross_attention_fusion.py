"""
Cross-Attention Fusion — Multi-head cross-attention for vision–language alignment.

Architecture
------------
    v_proj, t_proj → MultiHeadCrossAttention × L → LayerNorm → (B, embed_dim)

Each layer applies bidirectional cross-attention:
    v' = Attn(Q=v, K=t, V=t)
    t' = Attn(Q=t, K=v, V=v)
    fused = LayerNorm(v' + t')

This is the mechanism used in Flamingo / BridgeTower style vision-language
models, adapted here for embedding-level fusion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from fusion_core.fusion_methods.base import BaseFusion

logger = logging.getLogger(__name__)


class _CrossAttentionBlock(nn.Module):
    """Single bidirectional cross-attention block with residual + LN."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.v2t_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.t2v_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.ln_v = nn.LayerNorm(embed_dim)
        self.ln_t = nn.LayerNorm(embed_dim)
        self.ffn_v = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln_ffn_v = nn.LayerNorm(embed_dim)
        self.ln_ffn_t = nn.LayerNorm(embed_dim)

    def forward(
        self,
        v: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        v : (B, S_v, D)   — vision tokens  (S_v can be 1 for single-vector)
        t : (B, S_t, D)   — text tokens

        Returns
        -------
        v', t' : same shapes, cross-attended.
        """
        # Cross-attend
        v_att, _ = self.v2t_attn(query=v, key=t, value=t)
        t_att, _ = self.t2v_attn(query=t, key=v, value=v)

        # Residual + LN
        v = self.ln_v(v + v_att)
        t = self.ln_t(t + t_att)

        # FFN + Residual + LN
        v = self.ln_ffn_v(v + self.ffn_v(v))
        t = self.ln_ffn_t(t + self.ffn_t(t))

        return v, t


class CrossAttentionFusion(BaseFusion):
    """Multi-layer bidirectional cross-attention fusion.

    After all layers the two streams are averaged to produce a single
    fused representation of shape ``(B, embed_dim)``.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of stacked cross-attention blocks.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        output_dim: int,
        *,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            vision_dim=vision_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            output_dim=output_dim,
            name="cross_attn",
        )
        self.num_heads = num_heads
        self.num_layers_ca = num_layers  # avoid shadowing nn.Module attr

        self.blocks = nn.ModuleList([
            _CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_dim)

        logger.info(
            "CrossAttentionFusion init — heads=%d, layers=%d, params=%s",
            num_heads, num_layers, f"{self.param_count:,}",
        )

    # ── core fusion ──────────────────────────────────────────────────

    def _fuse(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend then mean-pool.  Returns ``(B, embed_dim)``."""
        # Add sequence dimension:  (B, D) → (B, 1, D)
        v = vision_embed.unsqueeze(1)
        t = text_embed.unsqueeze(1)

        for block in self.blocks:
            v, t = block(v, t)

        # Mean-pool the two streams
        fused = (v.squeeze(1) + t.squeeze(1)) / 2.0
        fused = self.final_ln(fused)
        return fused  # (B, embed_dim)

    # ── meta ─────────────────────────────────────────────────────────

    def _extra_meta(self) -> Dict[str, Any]:
        return {
            "num_heads": self.num_heads,
            "num_layers": self.num_layers_ca,
        }
