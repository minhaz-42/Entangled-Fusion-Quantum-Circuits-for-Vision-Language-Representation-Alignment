"""
Fusion Methods — Package Initialisation

Exposes all concrete fusion implementations and the factory function.
"""

from fusion_core.fusion_methods.base import BaseFusion
from fusion_core.fusion_methods.mlp_fusion import MLPFusion
from fusion_core.fusion_methods.cross_attention_fusion import CrossAttentionFusion
from fusion_core.fusion_methods.bilinear_fusion import BilinearFusion, TensorFusion
from fusion_core.fusion_methods.vqc_fusion import VQCFusion
from fusion_core.fusion_methods.classical_fallback import ClassicalFallbackFusion
from fusion_core.fusion_methods.factory import create_fusion, FUSION_REGISTRY

__all__ = [
    "BaseFusion",
    "MLPFusion",
    "CrossAttentionFusion",
    "BilinearFusion",
    "TensorFusion",
    "VQCFusion",
    "ClassicalFallbackFusion",
    "create_fusion",
    "FUSION_REGISTRY",
]
