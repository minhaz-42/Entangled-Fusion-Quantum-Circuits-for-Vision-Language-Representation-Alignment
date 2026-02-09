"""
Quantum Fusion Layer for Q-FuseVision AI Lab

Backward-compatible wrapper around the ``fusion_core`` research framework.
All fusion logic now lives in ``fusion_core/fusion_methods/`` and is
selected via ``fusion_core.config.cfg.FUSION_TYPE``.

This file re-exports the public API so that existing imports in views.py
continue to work unchanged:
    from .quantum_fusion import fuse_vision_language
"""

# ── Re-exports from the new framework ────────────────────────────────
from fusion_core.bridge import (       # noqa: F401
    fuse_vision_language,
    create_image_embedding,
    create_text_embedding,
    get_last_fusion_info,
    reset_fusion_model,
)
from fusion_core.config import cfg as fusion_config  # noqa: F401

# Legacy class kept for any code that instantiates it directly.
# New code should use ``fusion_core.fusion_methods.create_fusion(cfg)``
# instead.
from fusion_core.fusion_methods.vqc_fusion import VQCFusion as QuantumFusionLayer  # noqa: F401
