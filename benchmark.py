#!/usr/bin/env python3
"""
benchmark.py — Standalone CLI for running EFC fusion benchmarks.

This script makes ``fusion_core`` usable *outside* Django for
reproducible research experiments.

Usage
-----
    # Run VQC fusion with default settings
    python benchmark.py --fusion vqc

    # Compare all fusion methods
    python benchmark.py --fusion all

    # Specific config
    python benchmark.py --fusion cross_attn --embed-dim 256 --seed 123

    # Dump config
    python benchmark.py --dump-config
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_core.config import cfg, VALID_FUSION_TYPES
from fusion_core.fusion_methods.factory import create_fusion, FUSION_REGISTRY
from fusion_core.utils import seed_everything, setup_logging, Timer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EFC Benchmark Runner — Entangled Fusion Circuits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--fusion", type=str, default="vqc",
        help=f"Fusion type or 'all'.  Options: {sorted(VALID_FUSION_TYPES)}",
    )
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--output-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--num-qubits", type=int, default=None)
    p.add_argument("--vqc-layers", type=int, default=None)
    p.add_argument("--topology", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=5, help="Forward passes to average")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dump-config", action="store_true", help="Print config and exit")
    p.add_argument("--output", type=str, default=None, help="JSON output file")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _apply_overrides(args: argparse.Namespace) -> None:
    """Push CLI args into the global config."""
    cfg.SEED = args.seed
    cfg.DEVICE = args.device
    if args.embed_dim is not None:
        cfg.EMBED_DIM = args.embed_dim
    if args.output_dim is not None:
        cfg.OUTPUT_DIM = args.output_dim
    if args.hidden_dim is not None:
        cfg.HIDDEN_DIM = args.hidden_dim
    if args.num_qubits is not None:
        cfg.NUM_QUBITS = args.num_qubits
    if args.vqc_layers is not None:
        cfg.VQC_NUM_LAYERS = args.vqc_layers
    if args.topology is not None:
        cfg.VQC_ENTANGLE_TOPOLOGY = args.topology


def benchmark_single(
    fusion_type: str,
    batch_size: int,
    num_batches: int,
) -> dict:
    """Run a single fusion method and collect metrics."""
    local_cfg = cfg.copy()
    local_cfg.FUSION_TYPE = fusion_type
    local_cfg.validate()

    seed_everything(local_cfg.SEED)

    model = create_fusion(local_cfg)
    model.eval()

    device = torch.device(local_cfg.DEVICE)
    model = model.to(device)

    # Synthetic data
    latencies: list[float] = []
    for _ in range(num_batches):
        v = torch.randn(batch_size, local_cfg.EMBED_DIM, device=device)
        t = torch.randn(batch_size, local_cfg.EMBED_DIM, device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(v, t)
        elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed)

    meta = model.meta()

    result = {
        "fusion_type": fusion_type,
        "param_count": meta["param_count"],
        "output_shape": list(out.shape),
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "batch_size": batch_size,
        "num_batches": num_batches,
        "extra": meta.get("extra", {}),
    }
    return result


def main() -> None:
    args = parse_args()
    setup_logging("DEBUG" if args.verbose else "INFO")
    _apply_overrides(args)

    if args.dump_config:
        print(cfg.to_json())
        return

    fusion_types = sorted(FUSION_REGISTRY) if args.fusion == "all" else [args.fusion]

    results: list[dict] = []
    for ft in fusion_types:
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {ft}")
        print(f"{'='*60}")
        try:
            r = benchmark_single(ft, args.batch_size, args.num_batches)
            results.append(r)
            print(f"  Params:      {r['param_count']:>10,}")
            print(f"  Output:      {r['output_shape']}")
            print(f"  Latency:     {r['mean_latency_ms']:>8.2f} ± {r['std_latency_ms']:.2f} ms")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append({"fusion_type": ft, "error": str(exc)})

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    header = f"{'Method':<20} {'Params':>10} {'Latency (ms)':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['fusion_type']:<20} {'ERROR':>10} {r['error']}")
        else:
            print(
                f"{r['fusion_type']:<20} {r['param_count']:>10,} "
                f"{r['mean_latency_ms']:>8.2f} ± {r['std_latency_ms']:.2f}"
            )

    # Save
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
