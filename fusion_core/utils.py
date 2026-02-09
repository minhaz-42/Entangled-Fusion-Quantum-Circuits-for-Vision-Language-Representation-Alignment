"""
Utility helpers — seeding, timing, and logging setup.

Usage
-----
    from fusion_core.utils import seed_everything, Timer

    seed_everything(42)

    with Timer("forward pass"):
        out = model(v, t)
    # prints:  [Timer] forward pass — 12.34 ms
"""

from __future__ import annotations

import os
import time
import random
import logging
from contextlib import contextmanager
from typing import Generator

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False      # type: ignore[attr-defined]


class Timer:
    """Context-manager timer that logs elapsed time.

    Usage
    -----
        with Timer("quantum forward") as t:
            out = model(v, t_embed)
        print(t.elapsed_ms)
    """

    def __init__(self, label: str = "", log: bool = True) -> None:
        self.label = label
        self.log = log
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if self.log:
            logging.getLogger(__name__).info(
                "[Timer] %s — %.2f ms", self.label, self.elapsed_ms,
            )


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the framework."""
    fmt = "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_device(preference: str = "cpu") -> torch.device:
    """Return a torch device, falling back gracefully."""
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
