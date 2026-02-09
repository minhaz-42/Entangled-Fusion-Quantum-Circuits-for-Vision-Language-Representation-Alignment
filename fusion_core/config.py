"""
Fusion Core — Global Configuration System

Centralised, type-safe configuration for the entire EFC framework.
Every experiment picks up settings from here so that results are
deterministic and reproducible.

Usage:
    from fusion_core.config import cfg
    cfg.FUSION_TYPE = "cross_attn"
    cfg.NUM_QUBITS = 6

    # Or load from dict / JSON
    cfg.update({"FUSION_TYPE": "vqc", "EMBED_DIM": 256})
"""

from __future__ import annotations

import json
import copy
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported fusion types
# ---------------------------------------------------------------------------
VALID_FUSION_TYPES = frozenset({
    "mlp",
    "cross_attn",
    "bilinear",
    "tensor",
    "vqc",
    "classical_fallback",
})

VALID_VQC_TOPOLOGIES = frozenset({
    "ring",
    "full",
    "linear",
    "ladder",
})

VALID_VQC_ENCODINGS = frozenset({
    "angle",
    "iqp",
})


@dataclass
class FusionConfig:
    """Master configuration for the EFC framework.

    Every field has a sensible default so that experiments can be launched
    with zero user configuration.  Fields are grouped by sub-system.
    """

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    SEED: int = 42
    DEVICE: str = "cpu"               # "cpu" | "cuda" | "mps"
    LOG_LEVEL: str = "INFO"
    RESULTS_DIR: str = "results"

    # ------------------------------------------------------------------
    # Fusion Method Selection
    # ------------------------------------------------------------------
    FUSION_TYPE: str = "vqc"           # key into VALID_FUSION_TYPES
    EMBED_DIM: int = 128               # unified embedding dimensionality
    HIDDEN_DIM: int = 256              # hidden layer width (MLP / cross-attn)
    OUTPUT_DIM: int = 128              # fused output dimensionality
    DROPOUT: float = 0.1

    # ------------------------------------------------------------------
    # MLP Fusion
    # ------------------------------------------------------------------
    MLP_NUM_LAYERS: int = 3
    MLP_ACTIVATION: str = "gelu"       # "relu" | "gelu" | "tanh"

    # ------------------------------------------------------------------
    # Cross-Attention Fusion
    # ------------------------------------------------------------------
    CROSS_ATTN_NUM_HEADS: int = 4
    CROSS_ATTN_NUM_LAYERS: int = 2

    # ------------------------------------------------------------------
    # Bilinear / Tensor Fusion
    # ------------------------------------------------------------------
    BILINEAR_RANK: int = 16            # rank for low-rank bilinear
    TENSOR_FUSION_DIM: int = 64        # hidden dim for tensor network

    # ------------------------------------------------------------------
    # Quantum VQC Fusion  (PennyLane)
    # ------------------------------------------------------------------
    NUM_QUBITS: int = 8                # total qubits (split A/B for cross-modal)
    VQC_NUM_LAYERS: int = 4            # variational layers
    VQC_ENTANGLE_TOPOLOGY: str = "full"  # "ring" | "full" | "linear" | "ladder"
    VQC_ENCODING: str = "angle"          # "angle" | "iqp"
    VQC_BACKEND: str = "default.qubit"   # PennyLane device name
    VQC_DIFF_METHOD: str = "best"        # "best" | "parameter-shift" | "adjoint"
    VQC_MEASURE_ENTANGLEMENT: bool = True

    # ------------------------------------------------------------------
    # Classical Fallback
    # ------------------------------------------------------------------
    CLASSICAL_METHOD: str = "concat_tanh"  # "concat_tanh" | "hadamard" | "avg"

    # ------------------------------------------------------------------
    # Robustness (Phase 4 placeholder — config lives here for completeness)
    # ------------------------------------------------------------------
    NOISE_LEVELS: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2])
    NOISE_TYPES: List[str] = field(default_factory=lambda: ["gaussian", "token_mask", "fgsm"])

    # ------------------------------------------------------------------
    # Benchmarking (Phase 5 placeholder)
    # ------------------------------------------------------------------
    DATASET: str = "custom"            # "vqa_subset" | "mscoco" | "custom"
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` if any setting is invalid."""
        if self.FUSION_TYPE not in VALID_FUSION_TYPES:
            raise ValueError(
                f"FUSION_TYPE '{self.FUSION_TYPE}' not in {VALID_FUSION_TYPES}"
            )
        if self.EMBED_DIM <= 0:
            raise ValueError("EMBED_DIM must be > 0")
        if self.OUTPUT_DIM <= 0:
            raise ValueError("OUTPUT_DIM must be > 0")
        if self.NUM_QUBITS < 2:
            raise ValueError("NUM_QUBITS must be >= 2 (need at least 1 per modality)")
        if self.NUM_QUBITS % 2 != 0:
            raise ValueError("NUM_QUBITS must be even (split equally between modalities)")

        if self.VQC_ENTANGLE_TOPOLOGY not in VALID_VQC_TOPOLOGIES:
            raise ValueError(
                f"VQC_ENTANGLE_TOPOLOGY '{self.VQC_ENTANGLE_TOPOLOGY}' not in {sorted(VALID_VQC_TOPOLOGIES)}"
            )
        if self.VQC_ENCODING not in VALID_VQC_ENCODINGS:
            raise ValueError(
                f"VQC_ENCODING '{self.VQC_ENCODING}' not in {sorted(VALID_VQC_ENCODINGS)}"
            )

    def update(self, overrides: Dict[str, Any]) -> None:
        """Update config from a dictionary (e.g. CLI args or JSON)."""
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning("Unknown config key: %s", key)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(s)
        return s

    @classmethod
    def from_json(cls, path: str) -> "FusionConfig":
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def copy(self) -> "FusionConfig":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        lines = [f"FusionConfig("]
        for k, v in self.to_dict().items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton instance — import this everywhere
# ---------------------------------------------------------------------------
cfg = FusionConfig()
