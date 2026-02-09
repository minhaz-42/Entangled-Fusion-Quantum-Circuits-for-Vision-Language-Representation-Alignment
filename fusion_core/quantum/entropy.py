"""
Quantum Entropy & Entanglement Measures

Provides information-theoretic diagnostics for the entangled fusion
circuit, computed from the output state |ψ⟩ of ``PennyLaneEntangledCircuit``.

Metrics implemented
-------------------
1. **Von Neumann entropy** S(ρ_A)
   – Tracing out the text register to get ρ_A = Tr_B(|ψ⟩⟨ψ|)
   – S = − Σ λ_i log₂ λ_i
   – Measures entanglement in the bipartite pure state

2. **Subsystem purity** Tr(ρ_A²)
   – 1 → separable, 1/d_A → maximally mixed/entangled

3. **Mutual information proxy**  I(A:B) = S(A) + S(B) − S(AB)
   – For a pure state |ψ⟩ of A+B,  S(AB)=0  so  I(A:B)=2·S(A).

4. **Linear entropy** S_L = 1 − Tr(ρ_A²)
   – Operationally simpler surrogate for von Neumann entropy.

5. **Entanglement spectrum** {λ_i}
   – Sorted eigenvalues of ρ_A; shape of the distribution reveals
     the entanglement structure.

6. **Schmidt rank (effective)**
   – Number of eigenvalues > ε threshold — practical measure of
     the number of entangled modes.

Classical-surrogate mode
------------------------
When PennyLane is unavailable, the module can still estimate
entanglement characteristics from the classical surrogate weights
(using SVD on weight matrices as a proxy).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EntanglementReport:
    """Aggregated entanglement diagnostics for a single forward pass."""

    von_neumann_entropy: float = 0.0
    linear_entropy: float = 0.0
    subsystem_purity: float = 1.0
    mutual_information: float = 0.0
    schmidt_rank: int = 1
    eigenvalue_spectrum: List[float] = field(default_factory=list)
    max_possible_entropy: float = 0.0
    normalised_entropy: float = 0.0  # S / log₂(d_A)
    source: str = "unknown"          # "pennylane" | "classical_surrogate"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "von_neumann_entropy": round(self.von_neumann_entropy, 6),
            "linear_entropy": round(self.linear_entropy, 6),
            "subsystem_purity": round(self.subsystem_purity, 6),
            "mutual_information": round(self.mutual_information, 6),
            "schmidt_rank": self.schmidt_rank,
            "eigenvalue_spectrum": [round(e, 8) for e in self.eigenvalue_spectrum],
            "max_possible_entropy": round(self.max_possible_entropy, 6),
            "normalised_entropy": round(self.normalised_entropy, 6),
            "source": self.source,
        }

    def __repr__(self) -> str:
        return (
            f"EntanglementReport("
            f"S={self.von_neumann_entropy:.4f}, "
            f"S_L={self.linear_entropy:.4f}, "
            f"I(A:B)={self.mutual_information:.4f}, "
            f"schmidt={self.schmidt_rank}, "
            f"source={self.source})"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Core entropy computations (NumPy)
# ═══════════════════════════════════════════════════════════════════════

def _safe_log2(x: float) -> float:
    """log₂(x) with protection against log(0)."""
    return math.log2(x) if x > 1e-15 else 0.0


def reduced_density_matrix(
    statevector: np.ndarray,
    n_qubits_a: int,
    n_qubits_b: int,
) -> np.ndarray:
    """
    Compute ρ_A = Tr_B(|ψ⟩⟨ψ|) from a full statevector.

    Parameters
    ----------
    statevector : (2^n,) complex array
    n_qubits_a  : size of subsystem A (vision)
    n_qubits_b  : size of subsystem B (text)

    Returns
    -------
    rho_a : (d_A, d_A) complex array
    """
    d_a = 2 ** n_qubits_a
    d_b = 2 ** n_qubits_b

    sv = np.asarray(statevector).reshape(d_a, d_b)
    rho_a = sv @ sv.conj().T
    return rho_a


def von_neumann_entropy(rho: np.ndarray) -> float:
    """S(ρ) = − Σ λ_i log₂ λ_i from eigenvalues of a density matrix."""
    eigvals = np.linalg.eigvalsh(rho).real
    eigvals = np.clip(eigvals, 0, None)
    return -sum(_safe_log2(lam) * lam for lam in eigvals if lam > 1e-15)


def linear_entropy(rho: np.ndarray) -> float:
    """S_L(ρ) = 1 − Tr(ρ²)."""
    return 1.0 - np.real(np.trace(rho @ rho))


def purity(rho: np.ndarray) -> float:
    """Tr(ρ²)."""
    return float(np.real(np.trace(rho @ rho)))


def eigenvalue_spectrum(rho: np.ndarray) -> np.ndarray:
    """Sorted (descending) eigenvalues of ρ."""
    eigvals = np.linalg.eigvalsh(rho).real
    eigvals = np.clip(eigvals, 0, None)
    return np.sort(eigvals)[::-1]


def schmidt_rank(spectrum: np.ndarray, threshold: float = 1e-8) -> int:
    """Effective Schmidt rank (number of significant eigenvalues)."""
    return int(np.sum(spectrum > threshold))


# ═══════════════════════════════════════════════════════════════════════
#  High-level analyser
# ═══════════════════════════════════════════════════════════════════════

def compute_entanglement(
    statevector: np.ndarray,
    n_qubits_a: int,
    n_qubits_b: int,
) -> EntanglementReport:
    """
    Full entanglement analysis of a pure bipartite state |ψ⟩.

    Parameters
    ----------
    statevector : complex (2^(n_a+n_b),) array
    n_qubits_a  : qubits assigned to vision
    n_qubits_b  : qubits assigned to language

    Returns
    -------
    EntanglementReport with all metrics populated.
    """
    rho_a = reduced_density_matrix(statevector, n_qubits_a, n_qubits_b)

    d_a = 2 ** n_qubits_a
    max_S = math.log2(d_a)

    S = von_neumann_entropy(rho_a)
    S_L = linear_entropy(rho_a)
    P = purity(rho_a)
    spec = eigenvalue_spectrum(rho_a)
    sr = schmidt_rank(spec)
    I_AB = 2 * S  # For pure bipartite state: S(A)=S(B), S(AB)=0

    return EntanglementReport(
        von_neumann_entropy=S,
        linear_entropy=S_L,
        subsystem_purity=P,
        mutual_information=I_AB,
        schmidt_rank=sr,
        eigenvalue_spectrum=spec.tolist(),
        max_possible_entropy=max_S,
        normalised_entropy=S / max_S if max_S > 0 else 0.0,
        source="pennylane",
    )


# ═══════════════════════════════════════════════════════════════════════
#  Classical surrogate entanglement estimate
# ═══════════════════════════════════════════════════════════════════════

def classical_surrogate_entanglement(
    vision_embed: np.ndarray,
    text_embed: np.ndarray,
    n_qubits_a: int = 4,
    n_qubits_b: int = 4,
) -> EntanglementReport:
    """
    Estimate entanglement characteristics from classical embeddings.

    Uses SVD of the cross-modal correlation matrix as a proxy for the
    genuine quantum entanglement spectrum.  This does **not** represent
    true quantum entanglement but provides a heuristically meaningful
    correlation measure that is compatible with the same API.

    Parameters
    ----------
    vision_embed, text_embed : 1-D float arrays
    n_qubits_a, n_qubits_b  : qubit counts (for max entropy reference)
    """
    v = np.asarray(vision_embed).flatten()
    t = np.asarray(text_embed).flatten()

    # Build cross-correlation matrix M = v^T ⊗ t
    M = np.outer(v[:min(len(v), 64)], t[:min(len(t), 64)])
    _, sigma, _ = np.linalg.svd(M, full_matrices=False)

    # Normalise singular values to a probability distribution
    sigma = np.clip(sigma, 0, None)
    total = sigma.sum()
    if total > 1e-15:
        p = sigma / total
    else:
        p = np.ones_like(sigma) / len(sigma)

    # Proxy entropy
    S = -sum(_safe_log2(pi) * pi for pi in p if pi > 1e-15)
    max_S = math.log2(2 ** n_qubits_a) if n_qubits_a > 0 else 1.0
    S_L = 1.0 - sum(pi ** 2 for pi in p)
    P = sum(pi ** 2 for pi in p)
    sr = int(np.sum(p > 1e-8))

    return EntanglementReport(
        von_neumann_entropy=S,
        linear_entropy=S_L,
        subsystem_purity=P,
        mutual_information=2 * S,
        schmidt_rank=sr,
        eigenvalue_spectrum=p.tolist(),
        max_possible_entropy=max_S,
        normalised_entropy=S / max_S if max_S > 0 else 0.0,
        source="classical_surrogate",
    )
