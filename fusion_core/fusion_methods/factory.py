"""
Fusion Factory — centralised construction of fusion modules from config.

Usage
-----
    from fusion_core.config import cfg
    from fusion_core.fusion_methods.factory import create_fusion

    cfg.FUSION_TYPE = "cross_attn"
    model = create_fusion(cfg)
    fused = model(vision_embed, text_embed)
"""

from __future__ import annotations

import logging
from typing import Dict, Type

from fusion_core.config import FusionConfig
from fusion_core.fusion_methods.base import BaseFusion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FUSION_REGISTRY: Dict[str, Type[BaseFusion]] = {}


def _register(key: str, cls: Type[BaseFusion]) -> None:
    FUSION_REGISTRY[key] = cls


def _populate_registry() -> None:
    """Lazy import + register all built-in fusion classes."""
    from fusion_core.fusion_methods.mlp_fusion import MLPFusion
    from fusion_core.fusion_methods.cross_attention_fusion import CrossAttentionFusion
    from fusion_core.fusion_methods.bilinear_fusion import BilinearFusion, TensorFusion
    from fusion_core.fusion_methods.vqc_fusion import VQCFusion
    from fusion_core.fusion_methods.classical_fallback import ClassicalFallbackFusion

    _register("mlp", MLPFusion)
    _register("cross_attn", CrossAttentionFusion)
    _register("bilinear", BilinearFusion)
    _register("tensor", TensorFusion)
    _register("vqc", VQCFusion)
    _register("classical_fallback", ClassicalFallbackFusion)


_populate_registry()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_fusion(
    cfg: FusionConfig,
    *,
    vision_dim: int | None = None,
    text_dim: int | None = None,
) -> BaseFusion:
    """Instantiate a fusion module from a :class:`FusionConfig`.

    Parameters
    ----------
    cfg : FusionConfig
        Master config — ``cfg.FUSION_TYPE`` selects which class to build.
    vision_dim, text_dim : int, optional
        Override the input dimensionalities (default: ``cfg.EMBED_DIM``).

    Returns
    -------
    BaseFusion
        Ready-to-use fusion module (on CPU by default).

    Raises
    ------
    ValueError
        If ``cfg.FUSION_TYPE`` is unknown.
    """
    cfg.validate()

    key = cfg.FUSION_TYPE
    if key not in FUSION_REGISTRY:
        raise ValueError(
            f"Unknown FUSION_TYPE '{key}'.  Available: {sorted(FUSION_REGISTRY)}"
        )

    cls = FUSION_REGISTRY[key]
    v_dim = vision_dim or cfg.EMBED_DIM
    t_dim = text_dim or cfg.EMBED_DIM

    # Build keyword args specific to each fusion type
    kwargs: dict = {}

    if key == "mlp":
        kwargs.update(
            hidden_dim=cfg.HIDDEN_DIM,
            num_layers=cfg.MLP_NUM_LAYERS,
            activation=cfg.MLP_ACTIVATION,
            dropout=cfg.DROPOUT,
        )
    elif key == "cross_attn":
        kwargs.update(
            num_heads=cfg.CROSS_ATTN_NUM_HEADS,
            num_layers=cfg.CROSS_ATTN_NUM_LAYERS,
            dropout=cfg.DROPOUT,
        )
    elif key == "bilinear":
        kwargs.update(
            rank=cfg.BILINEAR_RANK,
            dropout=cfg.DROPOUT,
        )
    elif key == "tensor":
        kwargs.update(
            hidden_dim=cfg.TENSOR_FUSION_DIM,
            dropout=cfg.DROPOUT,
        )
    elif key == "vqc":
        kwargs.update(
            num_qubits=cfg.NUM_QUBITS,
            num_layers=cfg.VQC_NUM_LAYERS,
            topology=cfg.VQC_ENTANGLE_TOPOLOGY,
            encoding=cfg.VQC_ENCODING,
            backend=cfg.VQC_BACKEND,
            diff_method=cfg.VQC_DIFF_METHOD,
            measure_entanglement=cfg.VQC_MEASURE_ENTANGLEMENT,
            dropout=cfg.DROPOUT,
        )
    elif key == "classical_fallback":
        kwargs.update(method=cfg.CLASSICAL_METHOD)

    model = cls(
        vision_dim=v_dim,
        text_dim=t_dim,
        embed_dim=cfg.EMBED_DIM,
        output_dim=cfg.OUTPUT_DIM,
        **kwargs,
    )

    logger.info(
        "Created fusion module: %s  (%s params)",
        key, f"{model.param_count:,}",
    )
    return model
