"""
Circuit Analysis & Diagnostics

Provides structural analysis of entangled quantum circuits:
  - Gate count decomposition (single-qubit, two-qubit, total)
  - Circuit depth estimation
  - Parameter budget analysis
  - Expressibility estimation (Sim et al., 2019)
  - Entangling capability estimation
  - Resource cost projections for real hardware

These metrics are computed from the ``EntangledCircuitSpec`` dataclass
without requiring PennyLane or circuit execution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .entangled_circuit import EntangledCircuitSpec, TOPOLOGY_REGISTRY

logger = logging.getLogger(__name__)

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  Gate analysis
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GateAnalysis:
    """Detailed gate-level analysis of a circuit spec."""

    single_qubit_gates: int = 0
    two_qubit_gates: int = 0
    total_gates: int = 0
    cnot_count: int = 0
    cz_count: int = 0
    rotation_count: int = 0
    encoding_gates: int = 0
    total_parameters: int = 0
    depth_estimate: int = 0
    gates_per_layer: int = 0
    entangling_pairs_per_layer: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "single_qubit_gates": self.single_qubit_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "total_gates": self.total_gates,
            "cnot_count": self.cnot_count,
            "cz_count": self.cz_count,
            "rotation_count": self.rotation_count,
            "encoding_gates": self.encoding_gates,
            "total_parameters": self.total_parameters,
            "depth_estimate": self.depth_estimate,
            "gates_per_layer": self.gates_per_layer,
            "entangling_pairs_per_layer": self.entangling_pairs_per_layer,
        }


def analyse_gates(spec: EntangledCircuitSpec) -> GateAnalysis:
    """Static gate analysis from a circuit specification."""
    n = spec.total_qubits
    n_a = spec.n_qubits_a
    n_b = spec.n_qubits_b
    L = spec.num_layers

    # ── Encoding ─────────────────────────────────────────────────
    if spec.encoding == "iqp":
        # Hadamard(n) + Rz(n) + IsingZZ(n_a-1 + n_b-1)
        enc_single = n + n  # Hadamard + Rz
        enc_two = max(n_a - 1, 0) + max(n_b - 1, 0)  # IsingZZ
    else:  # angle
        enc_single = n * 2  # Rx, Ry per qubit
        enc_two = 0

    # ── Per variational layer ────────────────────────────────────
    rot_per_layer = n * 3  # Rx, Ry, Rz
    intra_a_cnot = max(n_a - 1, 0) + (1 if n_a > 1 else 0)  # ring
    intra_b_cnot = max(n_b - 1, 0) + (1 if n_b > 1 else 0)  # ring
    cross_cnot = len(spec.entangle_pairs)
    cz_per_layer = n // 2

    # Only apply CZ inter-layer between layers (not after last)
    total_cz = cz_per_layer * max(L - 1, 0)

    total_cnot = (intra_a_cnot + intra_b_cnot + cross_cnot) * L
    total_rot = enc_single + rot_per_layer * L
    total_two = enc_two + total_cnot + total_cz
    total_single = total_rot

    # Parameters: variational weights + encoding is data-dependent
    total_params = L * n * 3  # weight tensor shape (L, n, 3)

    # Depth: encoding depth + per-layer sequential blocks
    depth = 2 + L * 5  # conservative estimate

    gates_per_layer = rot_per_layer + intra_a_cnot + intra_b_cnot + cross_cnot + cz_per_layer

    return GateAnalysis(
        single_qubit_gates=total_single,
        two_qubit_gates=total_two,
        total_gates=total_single + total_two,
        cnot_count=total_cnot,
        cz_count=total_cz,
        rotation_count=total_rot,
        encoding_gates=enc_single + enc_two,
        total_parameters=total_params,
        depth_estimate=depth,
        gates_per_layer=gates_per_layer,
        entangling_pairs_per_layer=cross_cnot,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Expressibility estimate (Sim et al., 2019)
# ═══════════════════════════════════════════════════════════════════════

def estimate_expressibility(
    spec: EntangledCircuitSpec,
    n_samples: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Estimate circuit expressibility via fidelity distribution.

    Expressibility is quantified by the KL divergence between the
    fidelity distribution of random circuit instances and the Haar-random
    distribution (Porter-Thomas).

    For efficiency, this uses a numerical sampling approach:
      1. Sample random weight vectors θ₁, θ₂
      2. Compute |⟨ψ(θ₁)|ψ(θ₂)⟩|² (the fidelity)
      3. Compare histogram to Haar-random F ~ Beta(1, 2^n − 1)

    When PennyLane is unavailable, returns a heuristic estimate based
    on circuit structure.
    """
    n = spec.total_qubits
    dim = 2 ** n

    if PENNYLANE_AVAILABLE and n <= 12:
        # ---------- Numerical estimation with PennyLane ----------
        try:
            from .entangled_circuit import PennyLaneEntangledCircuit
            circuit = PennyLaneEntangledCircuit(spec)
            rng = np.random.default_rng(seed)

            fidelities: list[float] = []
            weight_shape = spec.weight_shape

            for _ in range(n_samples):
                # Random encoding angles + weights
                v1 = rng.uniform(-np.pi, np.pi, spec.n_qubits_a * 2)
                t1 = rng.uniform(-np.pi, np.pi, spec.n_qubits_b * 2)
                w1 = rng.uniform(-np.pi, np.pi, weight_shape)
                v2 = rng.uniform(-np.pi, np.pi, spec.n_qubits_a * 2)
                t2 = rng.uniform(-np.pi, np.pi, spec.n_qubits_b * 2)
                w2 = rng.uniform(-np.pi, np.pi, weight_shape)

                sv1 = circuit.get_statevector(v1, t1, w1)
                sv2 = circuit.get_statevector(v2, t2, w2)

                sv1 = np.asarray(sv1).flatten()
                sv2 = np.asarray(sv2).flatten()
                fid = float(np.abs(np.dot(sv1.conj(), sv2)) ** 2)
                fidelities.append(fid)

            fidelities_arr = np.array(fidelities)

            # KL divergence from Haar-random (Beta(1, dim-1))
            from scipy import stats
            hist, bin_edges = np.histogram(fidelities_arr, bins=50, range=(0, 1), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            haar_pdf = stats.beta.pdf(bin_centers, 1, dim - 1)

            # Regularise
            hist = hist + 1e-10
            haar_pdf = haar_pdf + 1e-10
            hist /= hist.sum()
            haar_pdf /= haar_pdf.sum()

            kl_div = float(stats.entropy(hist, haar_pdf))

            return {
                "expressibility_kl": round(kl_div, 6),
                "mean_fidelity": round(float(np.mean(fidelities_arr)), 6),
                "std_fidelity": round(float(np.std(fidelities_arr)), 6),
                "n_samples": n_samples,
                "method": "numerical",
            }
        except Exception as e:
            logger.warning("Expressibility numerical estimation failed: %s", e)

    # ---------- Heuristic fallback ----------
    gate_info = analyse_gates(spec)
    # More parameters and depth → higher expressibility
    param_density = gate_info.total_parameters / max(n, 1)
    depth_factor = min(gate_info.depth_estimate / (2 * n), 2.0)
    entangle_density = gate_info.entangling_pairs_per_layer / max(n * (n - 1) / 2, 1)

    # Lower KL → more expressive (closer to Haar)
    heuristic_kl = max(0.01, 2.0 / (1 + param_density * depth_factor * (1 + entangle_density)))

    return {
        "expressibility_kl": round(heuristic_kl, 6),
        "param_density": round(param_density, 4),
        "depth_factor": round(depth_factor, 4),
        "entangle_density": round(entangle_density, 4),
        "method": "heuristic",
    }


# ═══════════════════════════════════════════════════════════════════════
#  Entangling capability
# ═══════════════════════════════════════════════════════════════════════

def estimate_entangling_capability(
    spec: EntangledCircuitSpec,
    n_samples: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Estimate entangling capability via Meyer-Wallach Q measure.

    Q ∈ [0, 1]: average bipartite entanglement over random instances.
    Q = 0 → product states only; Q → 1 → high entanglement.

    When PennyLane is unavailable, uses a heuristic based on the
    entangling gate density.
    """
    n = spec.total_qubits

    if PENNYLANE_AVAILABLE and n <= 10:
        try:
            from .entangled_circuit import PennyLaneEntangledCircuit
            from .entropy import reduced_density_matrix, linear_entropy

            circuit = PennyLaneEntangledCircuit(spec)
            rng = np.random.default_rng(seed)

            q_values: list[float] = []

            for _ in range(n_samples):
                v = rng.uniform(-np.pi, np.pi, spec.n_qubits_a * 2)
                t = rng.uniform(-np.pi, np.pi, spec.n_qubits_b * 2)
                w = rng.uniform(-np.pi, np.pi, spec.weight_shape)

                sv = np.asarray(circuit.get_statevector(v, t, w)).flatten()

                # Meyer-Wallach: average linear entropy over single-qubit reductions
                ent_sum = 0.0
                for k in range(n):
                    # Partial trace over all except qubit k
                    rho_k = _single_qubit_rdm(sv, k, n)
                    ent_sum += linear_entropy(rho_k)
                q_values.append(2.0 * ent_sum / n)

            return {
                "entangling_capability_Q": round(float(np.mean(q_values)), 6),
                "std_Q": round(float(np.std(q_values)), 6),
                "n_samples": n_samples,
                "method": "numerical",
            }
        except Exception as e:
            logger.warning("Entangling capability estimation failed: %s", e)

    # Heuristic fallback
    gate_info = analyse_gates(spec)
    two_qubit_ratio = gate_info.two_qubit_gates / max(gate_info.total_gates, 1)
    cross_ratio = gate_info.entangling_pairs_per_layer / max(spec.total_qubits ** 2 / 4, 1)
    heuristic_Q = min(1.0, two_qubit_ratio * (1 + cross_ratio) *
                       min(spec.num_layers / 3, 1.5))

    return {
        "entangling_capability_Q": round(heuristic_Q, 6),
        "two_qubit_ratio": round(two_qubit_ratio, 4),
        "cross_modal_ratio": round(cross_ratio, 4),
        "method": "heuristic",
    }


def _single_qubit_rdm(statevector: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """
    Compute the reduced density matrix of a single qubit by tracing
    out all other qubits.
    """
    sv = statevector.reshape([2] * n_qubits)
    # Move target qubit to axis 0
    sv = np.moveaxis(sv, qubit, 0)
    # Reshape: (2, 2^(n-1))
    sv = sv.reshape(2, -1)
    rho = sv @ sv.conj().T
    return rho


# ═══════════════════════════════════════════════════════════════════════
#  Hardware resource estimation
# ═══════════════════════════════════════════════════════════════════════

def hardware_resource_estimate(spec: EntangledCircuitSpec) -> Dict[str, Any]:
    """
    Estimate resources needed for execution on real quantum hardware.

    Returns estimates for:
      - Qubit count (physical, with error correction overhead)
      - Gate time estimates (superconducting / trapped-ion)
      - Estimated execution time
      - CNOT decomposition overhead
    """
    gate_info = analyse_gates(spec)
    n = spec.total_qubits

    # Typical gate times (nanoseconds)
    SINGLE_GATE_NS = 30    # superconducting
    TWO_GATE_NS = 300      # superconducting CNOT
    T1_US = 100            # typical T1
    READOUT_NS = 500

    seq_time_ns = (
        gate_info.single_qubit_gates * SINGLE_GATE_NS +
        gate_info.two_qubit_gates * TWO_GATE_NS +
        n * READOUT_NS
    )

    # Parallelised time (rough: depth * max gate time)
    parallel_time_ns = (
        gate_info.depth_estimate * max(SINGLE_GATE_NS, TWO_GATE_NS) +
        n * READOUT_NS
    )

    # Error correction overhead (surface code, ~1000:1 physical:logical)
    ec_physical_qubits = n * 1000

    # 2-qubit gate fidelity budget
    CNOT_FIDELITY = 0.995
    circuit_fidelity = CNOT_FIDELITY ** gate_info.two_qubit_gates

    return {
        "logical_qubits": n,
        "physical_qubits_surface_code": ec_physical_qubits,
        "sequential_time_us": round(seq_time_ns / 1000, 2),
        "parallel_time_us": round(parallel_time_ns / 1000, 2),
        "estimated_circuit_fidelity": round(circuit_fidelity, 6),
        "shots_for_1pct_error": max(100, int(math.ceil(1 / (0.01 ** 2)))),
        "gate_analysis": gate_info.to_dict(),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Full diagnostic report
# ═══════════════════════════════════════════════════════════════════════

def full_circuit_analysis(
    spec: EntangledCircuitSpec,
    run_expressibility: bool = False,
    run_entangling_cap: bool = False,
) -> Dict[str, Any]:
    """
    Comprehensive circuit analysis combining all diagnostics.

    Parameters
    ----------
    spec : EntangledCircuitSpec
    run_expressibility : if True, run (possibly slow) expressibility estimate
    run_entangling_cap : if True, run (possibly slow) entangling capability

    Returns
    -------
    Dict with keys: circuit_spec, gate_analysis, hardware_estimate,
    and optionally expressibility, entangling_capability
    """
    report: Dict[str, Any] = {
        "circuit_spec": spec.to_dict(),
        "gate_analysis": analyse_gates(spec).to_dict(),
        "hardware_estimate": hardware_resource_estimate(spec),
    }

    if run_expressibility:
        report["expressibility"] = estimate_expressibility(spec)

    if run_entangling_cap:
        report["entangling_capability"] = estimate_entangling_capability(spec)

    return report
