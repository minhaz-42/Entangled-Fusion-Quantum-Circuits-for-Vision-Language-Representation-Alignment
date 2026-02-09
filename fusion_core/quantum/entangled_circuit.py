"""
Entangled Cross-Modal Quantum Circuit

Implements the core EFC (Entangled Fusion Circuit) for vision–language
alignment.  The circuit encodes two modality embeddings into separate
qubit registers and couples them via parameterised entangling layers.

Mathematical formulation:

    |ψ⟩ = U_entangle(θ) ( |vision⟩ ⊗ |text⟩ )

where:
    |vision⟩ = R_enc(v) |0⟩^{⊗n_A}          Amplitude encoding on group A
    |text⟩   = R_enc(t) |0⟩^{⊗n_B}          Amplitude encoding on group B
    U_entangle(θ) = ∏_{l} [ V_l(θ_l) · E_l ] Variational + entangling layers

Supports configurable:
    - Number of qubits per modality
    - Variational layer count
    - Entangling topology (ring, full, linear, ladder)
    - Encoding strategy (angle, amplitude, IQP)
    - Measurement basis

When PennyLane is unavailable, provides a classical surrogate that
preserves the same tensor shapes and API contract.

References
----------
- Schuld et al., "Circuit-centric quantum classifiers", PRA 101, 2020.
- Havlíček et al., "Supervised learning with quantum-enhanced feature
  spaces", Nature 567, 2019.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PennyLane availability
# ---------------------------------------------------------------------------
try:
    import pennylane as qml

    PENNYLANE_AVAILABLE = True
except ImportError:
    qml = None  # type: ignore[assignment]
    PENNYLANE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  Topology generators
# ═══════════════════════════════════════════════════════════════════════

def ring_topology(n_a: int, n_b: int) -> List[Tuple[int, int]]:
    """Ring coupling:  A_i ↔ B_i and B_i ↔ A_{i+1 mod n_a}."""
    pairs: list[tuple[int, int]] = []
    for i in range(min(n_a, n_b)):
        pairs.append((i, n_a + i))
        pairs.append((n_a + i, (i + 1) % n_a))
    return pairs


def full_topology(n_a: int, n_b: int) -> List[Tuple[int, int]]:
    """All A×B pairs."""
    return [(i, n_a + j) for i in range(n_a) for j in range(n_b)]


def linear_topology(n_a: int, n_b: int) -> List[Tuple[int, int]]:
    """One-to-one nearest-neighbour: A_i ↔ B_i."""
    return [(i, n_a + i) for i in range(min(n_a, n_b))]


def ladder_topology(n_a: int, n_b: int) -> List[Tuple[int, int]]:
    """Ladder: A_i ↔ B_i plus A_i ↔ B_{i+1}."""
    pairs: list[tuple[int, int]] = []
    m = min(n_a, n_b)
    for i in range(m):
        pairs.append((i, n_a + i))
        if i + 1 < n_b:
            pairs.append((i, n_a + i + 1))
    return pairs


TOPOLOGY_REGISTRY = {
    "ring": ring_topology,
    "full": full_topology,
    "linear": linear_topology,
    "ladder": ladder_topology,
}


# ═══════════════════════════════════════════════════════════════════════
#  Circuit specification (data-only, framework-agnostic)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EntangledCircuitSpec:
    """Declarative specification for a cross-modal entangled circuit.

    This is a *pure data object* — it describes the circuit but does not
    execute it.  PennyLane or a classical surrogate consumes the spec.
    """
    n_qubits_a: int = 4              # vision register size
    n_qubits_b: int = 4              # text register size
    num_layers: int = 4              # variational + entangling layers
    topology: str = "full"           # key into TOPOLOGY_REGISTRY
    encoding: str = "angle"          # "angle" | "amplitude" | "iqp"
    measurement: str = "pauli_z"     # "pauli_z" | "probs"
    backend: str = "default.qubit"
    diff_method: str = "best"

    @property
    def total_qubits(self) -> int:
        return self.n_qubits_a + self.n_qubits_b

    @property
    def entangle_pairs(self) -> List[Tuple[int, int]]:
        fn = TOPOLOGY_REGISTRY.get(self.topology, full_topology)
        return fn(self.n_qubits_a, self.n_qubits_b)

    @property
    def weight_shape(self) -> Tuple[int, int, int]:
        """Shape of the variational weight tensor: (layers, total_qubits, 3)."""
        return (self.num_layers, self.total_qubits, 3)

    def gate_count(self) -> Dict[str, int]:
        """Count gates by type."""
        n = self.total_qubits
        n_a, n_b = self.n_qubits_a, self.n_qubits_b

        # Encoding: 2 rotations per qubit
        encoding = n * 2

        # Per variational layer
        rotations_per_layer = n * 3  # Rx, Ry, Rz
        intra_a = max(n_a - 1, 0) + (1 if n_a > 1 else 0)  # ring CNOT
        intra_b = max(n_b - 1, 0) + (1 if n_b > 1 else 0)
        cross = len(self.entangle_pairs)
        cz_per_layer = n // 2

        total_cnot = (intra_a + intra_b + cross) * self.num_layers
        total_cz = cz_per_layer * max(self.num_layers - 1, 0)
        total_rotations = encoding + rotations_per_layer * self.num_layers

        return {
            "encoding_gates": encoding,
            "rotation_gates": total_rotations,
            "cnot_gates": total_cnot,
            "cz_gates": total_cz,
            "total_gates": total_rotations + total_cnot + total_cz,
            "entangle_pairs": cross,
        }

    def circuit_depth(self) -> int:
        """Approximate depth (sequential gate layers)."""
        # encoding (1) + per var layer: rotations(1) + intra-A(~n_a) +
        # intra-B(~n_b) + cross(~1) + CZ(1)
        return 1 + self.num_layers * 5

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "n_qubits_a": self.n_qubits_a,
            "n_qubits_b": self.n_qubits_b,
            "total_qubits": self.total_qubits,
            "num_layers": self.num_layers,
            "topology": self.topology,
            "encoding": self.encoding,
            "measurement": self.measurement,
            "gate_count": self.gate_count(),
            "circuit_depth": self.circuit_depth(),
        }
        return d


# ═══════════════════════════════════════════════════════════════════════
#  PennyLane circuit builder
# ═══════════════════════════════════════════════════════════════════════

class PennyLaneEntangledCircuit:
    """Builds and executes entangled circuits on a PennyLane device.

    This class is used internally by ``VQCFusion`` — other code should
    not need to interact with it directly.
    """

    def __init__(self, spec: EntangledCircuitSpec) -> None:
        if not PENNYLANE_AVAILABLE:
            raise RuntimeError("PennyLane is required for PennyLaneEntangledCircuit")

        self.spec = spec
        self.dev = qml.device(spec.backend, wires=spec.total_qubits)

        # Build the main measurement QNode
        self._qnode = qml.QNode(
            self._circuit_fn,
            self.dev,
            interface="torch",
            diff_method=spec.diff_method,
        )

        # Build state-vector QNode for entropy analysis
        self._sv_dev = qml.device("default.qubit", wires=spec.total_qubits)
        self._state_qnode = qml.QNode(
            self._state_circuit_fn,
            self._sv_dev,
            interface="torch",
        )

    # ── Encoding sub-routines ────────────────────────────────────────

    def _encode_angles(
        self,
        vision_angles,   # (n_a * 2,)
        text_angles,     # (n_b * 2,)
    ) -> None:
        """Angle encoding: Rx, Ry per qubit."""
        n_a = self.spec.n_qubits_a
        n_b = self.spec.n_qubits_b
        for i in range(n_a):
            qml.RX(vision_angles[2 * i], wires=i)
            qml.RY(vision_angles[2 * i + 1], wires=i)
        for j in range(n_b):
            qml.RX(text_angles[2 * j], wires=n_a + j)
            qml.RY(text_angles[2 * j + 1], wires=n_a + j)

    def _encode_iqp(
        self,
        vision_angles,
        text_angles,
    ) -> None:
        """IQP-style encoding: Hadamard + Rz + ZZ interactions."""
        n_a = self.spec.n_qubits_a
        n_b = self.spec.n_qubits_b
        # Hadamard layer
        for i in range(self.spec.total_qubits):
            qml.Hadamard(wires=i)
        # Rz encoding
        for i in range(n_a):
            qml.RZ(vision_angles[i % len(vision_angles)], wires=i)
        for j in range(n_b):
            qml.RZ(text_angles[j % len(text_angles)], wires=n_a + j)
        # ZZ interactions within groups
        for i in range(n_a - 1):
            qml.IsingZZ(vision_angles[i % len(vision_angles)] * vision_angles[(i + 1) % len(vision_angles)],
                        wires=[i, i + 1])
        for j in range(n_b - 1):
            qml.IsingZZ(text_angles[j % len(text_angles)] * text_angles[(j + 1) % len(text_angles)],
                        wires=[n_a + j, n_a + j + 1])

    def _apply_encoding(self, vision_angles, text_angles) -> None:
        if self.spec.encoding == "iqp":
            self._encode_iqp(vision_angles, text_angles)
        else:
            self._encode_angles(vision_angles, text_angles)

    # ── Variational + entangling layers ──────────────────────────────

    def _variational_layer(self, weights_layer, layer_idx: int) -> None:
        """Single variational + entangling layer."""
        n_a = self.spec.n_qubits_a
        n_b = self.spec.n_qubits_b
        n = self.spec.total_qubits

        # ---------- Per-qubit rotations ----------
        for q in range(n):
            qml.RX(weights_layer[q, 0], wires=q)
            qml.RY(weights_layer[q, 1], wires=q)
            qml.RZ(weights_layer[q, 2], wires=q)

        # ---------- Intra-group A entanglement (ring CNOT) ----------
        for i in range(n_a - 1):
            qml.CNOT(wires=[i, i + 1])
        if n_a > 1:
            qml.CNOT(wires=[n_a - 1, 0])

        # ---------- Intra-group B entanglement (ring CNOT) ----------
        for j in range(n_b - 1):
            qml.CNOT(wires=[n_a + j, n_a + j + 1])
        if n_b > 1:
            qml.CNOT(wires=[n_a + n_b - 1, n_a])

        # ---------- Cross-modal entanglement (A ↔ B) ----------
        for a_wire, b_wire in self.spec.entangle_pairs:
            qml.CNOT(wires=[a_wire, b_wire])

        # ---------- CZ inter-layer coupling ----------
        if layer_idx < self.spec.num_layers - 1:
            for i in range(0, n - 1, 2):
                qml.CZ(wires=[i, i + 1])

    # ── Full circuit functions ───────────────────────────────────────

    def _circuit_fn(self, vision_angles, text_angles, weights):
        """Full parameterised circuit → Pauli-Z expectations."""
        self._apply_encoding(vision_angles, text_angles)
        for l in range(self.spec.num_layers):
            self._variational_layer(weights[l], l)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.spec.total_qubits)]

    def _state_circuit_fn(self, vision_angles, text_angles, weights):
        """Same circuit but returns the full state vector."""
        self._apply_encoding(vision_angles, text_angles)
        for l in range(self.spec.num_layers):
            self._variational_layer(weights[l], l)
        return qml.state()

    # ── Public execution API ─────────────────────────────────────────

    def execute(self, vision_angles, text_angles, weights):
        """Run circuit and return Pauli-Z expectation values."""
        return self._qnode(vision_angles, text_angles, weights)

    def get_statevector(self, vision_angles, text_angles, weights):
        """Run circuit and return full state vector (for entropy computation)."""
        return self._state_qnode(vision_angles, text_angles, weights)

    def draw(self, vision_angles, text_angles, weights) -> str:
        """Return a text drawing of the circuit."""
        return qml.draw(self._qnode)(vision_angles, text_angles, weights)
