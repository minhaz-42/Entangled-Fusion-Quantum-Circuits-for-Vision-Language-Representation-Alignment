"""
Bridge module — connects the ``fusion_core`` research framework to the
existing ``fusionapp`` Django application.

This replaces the old monolithic ``fuse_vision_language()`` function with
one that routes through the pluggable fusion factory, while keeping the
exact same public API so that ``views.py`` needs zero changes.

Usage (unchanged from before):
    from fusionapp.quantum_fusion import fuse_vision_language
    fused_vec = fuse_vision_language(image_path, question)

Configuration:
    Set ``FUSION_TYPE`` in ``fusion_core/config.py`` or at runtime:
        from fusion_core.config import cfg
        cfg.FUSION_TYPE = "cross_attn"
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch

from fusion_core.config import cfg
from fusion_core.fusion_methods.factory import create_fusion
from fusion_core.utils import seed_everything

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton fusion model (lazy-initialised)
# ---------------------------------------------------------------------------
_fusion_model = None
_last_fusion_info = None


def _get_fusion_model():
    """Lazy-create the fusion module from the current config."""
    global _fusion_model
    if _fusion_model is None:
        seed_everything(cfg.SEED)
        _fusion_model = create_fusion(cfg)
        _fusion_model.eval()
        logger.info("Fusion model created: %s", _fusion_model.meta())
    return _fusion_model


def reset_fusion_model() -> None:
    """Force re-creation (e.g. after changing ``cfg.FUSION_TYPE``)."""
    global _fusion_model
    _fusion_model = None


def get_last_fusion_info() -> dict:
        """Return diagnostics from the most recent fusion call.

        Includes:
            - ``meta``: BaseFusion.meta() output
            - Optional ``entanglement_report`` (if available)
            - Optional ``circuit_analysis`` (if available)
            - ``config_snapshot``: cfg.to_dict() at call time
        """
        global _last_fusion_info
        return dict(_last_fusion_info or {})


# ---------------------------------------------------------------------------
# Embedding helpers  (same signatures as the original quantum_fusion.py)
# ---------------------------------------------------------------------------

def create_image_embedding(image_path: str) -> np.ndarray:
    """Create a simple statistical embedding from an image.

    In a publication-grade pipeline you would swap this for CLIP / DINOv2.
    The dimensionality is ``cfg.EMBED_DIM``.
    """
    try:
        from PIL import Image

        img = Image.open(image_path).convert("RGB").resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0

        feats: list[float] = []

        # Channel statistics
        for c in range(3):
            ch = arr[:, :, c]
            feats.extend([
                float(np.mean(ch)),
                float(np.std(ch)),
                float(np.median(ch)),
                float(np.percentile(ch, 25)),
                float(np.percentile(ch, 75)),
            ])

        # Spatial quadrant means
        h, w = arr.shape[:2]
        for i in range(2):
            for j in range(2):
                quad = arr[i * h // 2:(i + 1) * h // 2, j * w // 2:(j + 1) * w // 2]
                feats.append(float(np.mean(quad)))

        embedding = np.array(feats, dtype=np.float32)

        # Resize to embed_dim
        target = cfg.EMBED_DIM
        if len(embedding) < target:
            embedding = np.pad(embedding, (0, target - len(embedding)))
        else:
            embedding = embedding[:target]

        return embedding

    except Exception as exc:
        logger.warning("Image embedding fallback: %s", exc)
        return np.random.randn(cfg.EMBED_DIM).astype(np.float32)


def create_text_embedding(text: str) -> np.ndarray:
    """Create a simple statistical embedding from text.

    In a publication-grade pipeline you would swap this for a sentence
    transformer (e.g. ``all-MiniLM-L6-v2``).
    """
    try:
        words = text.lower().split()
        chars = list(text.lower())

        feats: list[float] = []
        feats.append(len(text) / 1000.0)
        feats.append(len(words) / 100.0)
        feats.append(np.mean([len(w) for w in words]) / 10.0 if words else 0.0)

        common = "etaoinshrdlcumwfgypbvkjxqz"
        for ch in common[:10]:
            feats.append(chars.count(ch) / max(len(chars), 1))

        feats.append(1.0 if "?" in text else 0.0)
        feats.append(1.0 if any(w in text.lower() for w in ["what", "how", "why", "where", "when", "who"]) else 0.0)
        feats.append(1.0 if any(w in text.lower() for w in ["describe", "explain", "tell"]) else 0.0)

        embedding = np.array(feats, dtype=np.float32)

        target = cfg.EMBED_DIM
        if len(embedding) < target:
            embedding = np.pad(embedding, (0, target - len(embedding)))
        else:
            embedding = embedding[:target]

        return embedding

    except Exception as exc:
        logger.warning("Text embedding fallback: %s", exc)
        return np.random.randn(cfg.EMBED_DIM).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API  (drop-in replacement)
# ---------------------------------------------------------------------------

def fuse_vision_language(image_path: str, question: str) -> List[float]:
    """Fuse vision and language inputs using the configured fusion method.

    Returns a plain Python list of floats (JSON-serialisable) for storage
    in the Django ``FusionExperiment.fused_vector`` field.
    """
    model = _get_fusion_model()

    img_emb = create_image_embedding(image_path)
    txt_emb = create_text_embedding(question)

    v = torch.tensor(img_emb, dtype=torch.float32)
    t = torch.tensor(txt_emb, dtype=torch.float32)

    with torch.no_grad():
        fused = model(v, t)  # (output_dim,)

    # Log metadata for the experiment
    meta = model.meta()

    # Capture full Phase-2 diagnostics for UI + experiment logging
    global _last_fusion_info
    info: dict = {
        "meta": meta,
        "config_snapshot": cfg.to_dict(),
    }
    if hasattr(model, "get_entanglement_report"):
        try:
            info["entanglement_report"] = model.get_entanglement_report()
        except Exception:
            pass
    if hasattr(model, "get_circuit_analysis"):
        try:
            info["circuit_analysis"] = model.get_circuit_analysis()
        except Exception:
            pass
    _last_fusion_info = info

    logger.info(
        "Fusion complete — method=%s, latency=%.2f ms, params=%s",
        meta["name"], meta["last_forward_ms"], f'{meta["param_count"]:,}',
    )

    return fused.cpu().tolist()
