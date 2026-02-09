"""
VQC Fusion — Variational Quantum Circuit fusion with entangled cross-modal encoding.

Implements the core research contribution:

    |ψ⟩ = U_entangle(θ) ( |vision⟩ ⊗ |text⟩ )

Architecture
------------
1. **Amplitude encoding** — project each modality embedding into qubit
   rotation angles via a learned linear map.
2. **Modality-specific ansatz** — apply parameterised rotations to qubit
   group A (vision) and qubit group B (text) independently.
3. **Entangling layer** — CNOT / CZ gates couple A ↔ B according to a
   configurable topology (ring, full, linear, ladder).
4. **Measurement** — Pauli-Z expectations on all qubits are concatenated
   and projected to ``embed_dim``.

Phase 2 enhancements:
    - Full integration with ``fusion_core.quantum`` analysis modules
    - ``EntangledCircuitSpec`` declarative circuit description
    - ``EntanglementReport`` with von Neumann / linear entropy, purity,
      mutual information, Schmidt rank, eigenvalue spectrum
    - Static gate analysis and hardware resource estimation
    - Ladder topology support

Diagnostics logged per forward pass:
    - Entanglement entropy (von Neumann S of reduced ρ_A)
    - Linear entropy and subsystem purity
    - Mutual information proxy  I(A;B) = 2·S(A)
    - Schmidt rank of the bipartite state
    - Circuit depth and gate count
    - Normalised entropy (S / S_max)

Falls back to a differentiable classical surrogate when PennyLane is not
installed (``ClassicalVQCSurrogate``).

References
----------
- Schuld & Petruccione, "Supervised Learning with Quantum Computers", 2018.
- Havlíček et al., "Supervised learning with quantum-enhanced feature spaces",
  Nature 567, 2019.
- Sim et al., "Expressibility and entangling capability of PQCs", Adv. QT, 2019.
"""

from __future__ import annotations

import logging
import time
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from fusion_core.fusion_methods.base import BaseFusion

# Phase 2 quantum analysis modules
from fusion_core.quantum.entangled_circuit import (
    EntangledCircuitSpec,
    TOPOLOGY_REGISTRY,
    PENNYLANE_AVAILABLE as _PL_AVAILABLE,
)
from fusion_core.quantum.entropy import (
    EntanglementReport,
    compute_entanglement,
    classical_surrogate_entanglement,
)
from fusion_core.quantum.circuit_analysis import (
    analyse_gates,
    full_circuit_analysis,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PennyLane availability
# ---------------------------------------------------------------------------
try:
    import pennylane as qml
    from pennylane import numpy as pnp

    PENNYLANE_AVAILABLE = True
except ImportError:
    qml = None  # type: ignore[assignment]
    pnp = None  # type: ignore[assignment]
    PENNYLANE_AVAILABLE = False
    logger.info("PennyLane not installed — VQCFusion will use classical surrogate.")


# ═══════════════════════════════════════════════════════════════════════
#  Entangling topologies  (legacy helpers, now backed by quantum module)
# ═══════════════════════════════════════════════════════════════════════

def _ring_entangle(n_a: int, n_b: int) -> List[Tuple[int, int]]:
    """Ring coupling between group A (0..n_a-1) and group B (n_a..n_a+n_b-1)."""
    return TOPOLOGY_REGISTRY["ring"](n_a, n_b)


def _full_entangle(n_a: int, n_b: int) -> List[Tuple[int, int]]:
    """All-to-all coupling between groups."""
    return TOPOLOGY_REGISTRY["full"](n_a, n_b)


def _linear_entangle(n_a: int, n_b: int) -> List[Tuple[int, int]]:
    """Linear nearest-neighbour coupling."""
    return TOPOLOGY_REGISTRY["linear"](n_a, n_b)


def _ladder_entangle(n_a: int, n_b: int) -> List[Tuple[int, int]]:
    """Ladder topology: A_i↔B_i and A_i↔B_{i+1}."""
    return TOPOLOGY_REGISTRY["ladder"](n_a, n_b)


_TOPOLOGY_MAP = {
    "ring": _ring_entangle,
    "full": _full_entangle,
    "linear": _linear_entangle,
    "ladder": _ladder_entangle,
}


# ═══════════════════════════════════════════════════════════════════════
#  Classical Surrogate (when PennyLane is missing)
# ═══════════════════════════════════════════════════════════════════════

class _ClassicalVQCSurrogate(nn.Module):
    """Differentiable classical surrogate that mimics the VQC structure.

    Uses sinusoidal non-linearities to approximate the periodicity of
    parameterised quantum gates.
    """

    def __init__(self, embed_dim: int, num_qubits: int, num_layers: int) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        half = num_qubits // 2

        # Encoding projections (embed_dim → angles per qubit group)
        self.vision_enc = nn.Linear(embed_dim, half * 3)  # Rx, Ry, Rz per qubit
        self.text_enc = nn.Linear(embed_dim, half * 3)

        # "Variational" layers
        layers: list[nn.Module] = []
        dim = num_qubits * 3
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            ])
        self.var_layers = nn.Sequential(*layers)

        # Readout: simulate Pauli-Z expectations → one scalar per qubit
        self.readout = nn.Linear(dim, num_qubits)

    def forward(self, vision: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        v_angles = self.vision_enc(vision)   # (B, half*3)
        t_angles = self.text_enc(text)       # (B, half*3)
        x = torch.cat([v_angles, t_angles], dim=-1)  # (B, n_qubits*3)
        x = torch.sin(x)                    # periodic non-linearity
        x = self.var_layers(x)
        x = torch.tanh(x)                   # simulate bounded expectation
        out = self.readout(x)                # (B, num_qubits)
        return out


# ═══════════════════════════════════════════════════════════════════════
#  VQCFusion  (main class)
# ═══════════════════════════════════════════════════════════════════════

class VQCFusion(BaseFusion):
    """Variational Quantum Circuit fusion with entangled cross-modal encoding.

    When PennyLane is available the circuit is executed on the specified
    device.  Otherwise an equivalent classical surrogate is used so that
    the rest of the framework never breaks.

    Parameters
    ----------
    num_qubits : int
        Total qubits.  Split 50/50 between vision (group A) and text (B).
    num_layers : int
        Number of variational + entangling layers.
    topology : str
        Entangling topology — ``"ring"`` | ``"full"`` | ``"linear"`` | ``"ladder"``.
    encoding : str
        Encoding strategy — ``"angle"`` (default) | ``"iqp"``.
    backend : str
        PennyLane device name, e.g. ``"default.qubit"``.
    diff_method : str
        Differentiation method for the QNode.
    measure_entanglement : bool
        If True, compute entanglement entropy on every forward (slower).
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        output_dim: int,
        *,
        num_qubits: int = 8,
        num_layers: int = 4,
        topology: str = "full",
        encoding: str = "angle",
        backend: str = "default.qubit",
        diff_method: str = "best",
        measure_entanglement: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            vision_dim=vision_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            output_dim=output_dim,
            name="vqc",
        )
        assert num_qubits >= 2 and num_qubits % 2 == 0, \
            "num_qubits must be even and >= 2"

        self.num_qubits = num_qubits
        self.n_a = num_qubits // 2  # vision qubits
        self.n_b = num_qubits // 2  # text qubits
        self.num_vqc_layers = num_layers
        self.topology_name = topology
        self.encoding_name = encoding
        self.measure_entanglement = measure_entanglement
        self._using_pennylane = PENNYLANE_AVAILABLE

        # ── Phase 2: declarative circuit specification ───────────────
        self.circuit_spec = EntangledCircuitSpec(
            n_qubits_a=self.n_a,
            n_qubits_b=self.n_b,
            num_layers=num_layers,
            topology=topology,
            encoding=encoding,
            backend=backend,
            diff_method=diff_method,
        )
        self._gate_analysis = analyse_gates(self.circuit_spec)

        # Entangling pairs (from spec)
        self.entangle_pairs: List[Tuple[int, int]] = self.circuit_spec.entangle_pairs

        # Encoding projections  (embed_dim → rotation angles)
        self.vision_encoder = nn.Linear(embed_dim, self.n_a * 2)  # Rx, Ry
        self.text_encoder = nn.Linear(embed_dim, self.n_b * 2)

        # Post-measurement projection  (num_qubits → embed_dim)
        self.post_measure = nn.Sequential(
            nn.Linear(num_qubits, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

        # Diagnostics (updated each forward)
        self._diag: Dict[str, Any] = {}
        self._last_entanglement_report: Optional[EntanglementReport] = None

        if PENNYLANE_AVAILABLE:
            self._init_pennylane(backend, diff_method, num_layers)
        else:
            self.surrogate = _ClassicalVQCSurrogate(embed_dim, num_qubits, num_layers)

        logger.info(
            "VQCFusion init — qubits=%d (A=%d B=%d), layers=%d, topo=%s, "
            "encoding=%s, pennylane=%s, params=%s, gates=%d",
            num_qubits, self.n_a, self.n_b, num_layers, topology,
            encoding, PENNYLANE_AVAILABLE, f"{self.param_count:,}",
            self._gate_analysis.total_gates,
        )

    # ── PennyLane initialisation ─────────────────────────────────────

    def _init_pennylane(self, backend: str, diff_method: str, num_layers: int) -> None:
        """Build the PennyLane device, QNode, and trainable weights."""
        self.dev = qml.device(backend, wires=self.num_qubits)

        # Trainable variational weights: (num_layers, num_qubits, 3)  for Rx,Ry,Rz
        weight_shape = (num_layers, self.num_qubits, 3)
        self.var_weights = nn.Parameter(
            torch.empty(*weight_shape).uniform_(-np.pi, np.pi)
        )

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def _circuit(
            vision_angles: torch.Tensor,
            text_angles: torch.Tensor,
            weights: torch.Tensor,
        ) -> list:
            """Parameterised quantum circuit with entangled cross-modal encoding."""
            # ── 1. Amplitude Encoding ────────────────────────────────
            for i in range(self.n_a):
                qml.RX(vision_angles[2 * i], wires=i)
                qml.RY(vision_angles[2 * i + 1], wires=i)

            for j in range(self.n_b):
                qml.RX(text_angles[2 * j], wires=self.n_a + j)
                qml.RY(text_angles[2 * j + 1], wires=self.n_a + j)

            # ── 2. Variational + Entangling Layers ───────────────────
            for layer in range(self.num_vqc_layers):
                # Per-qubit rotations
                for q in range(self.num_qubits):
                    qml.RX(weights[layer, q, 0], wires=q)
                    qml.RY(weights[layer, q, 1], wires=q)
                    qml.RZ(weights[layer, q, 2], wires=q)

                # Intra-group entanglement (ring within each group)
                for i in range(self.n_a - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_a > 1:
                    qml.CNOT(wires=[self.n_a - 1, 0])

                for j in range(self.n_b - 1):
                    qml.CNOT(wires=[self.n_a + j, self.n_a + j + 1])
                if self.n_b > 1:
                    qml.CNOT(wires=[self.n_a + self.n_b - 1, self.n_a])

                # Cross-modal entanglement (A ↔ B)
                for a_wire, b_wire in self.entangle_pairs:
                    qml.CNOT(wires=[a_wire, b_wire])

                # Additional CZ for depth
                if layer < self.num_vqc_layers - 1:
                    for i in range(0, self.num_qubits - 1, 2):
                        qml.CZ(wires=[i, i + 1])

            # ── 3. Measurement ───────────────────────────────────────
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self._qnode = _circuit

        # State-vector QNode for entanglement entropy (no gradients needed)
        if self.measure_entanglement:
            dev_sv = qml.device("default.qubit", wires=self.num_qubits)

            @qml.qnode(dev_sv, interface="torch")
            def _state_circuit(
                vision_angles: torch.Tensor,
                text_angles: torch.Tensor,
                weights: torch.Tensor,
            ):
                """Same circuit but return state vector."""
                for i in range(self.n_a):
                    qml.RX(vision_angles[2 * i], wires=i)
                    qml.RY(vision_angles[2 * i + 1], wires=i)
                for j in range(self.n_b):
                    qml.RX(text_angles[2 * j], wires=self.n_a + j)
                    qml.RY(text_angles[2 * j + 1], wires=self.n_a + j)
                for layer in range(self.num_vqc_layers):
                    for q in range(self.num_qubits):
                        qml.RX(weights[layer, q, 0], wires=q)
                        qml.RY(weights[layer, q, 1], wires=q)
                        qml.RZ(weights[layer, q, 2], wires=q)
                    for i in range(self.n_a - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if self.n_a > 1:
                        qml.CNOT(wires=[self.n_a - 1, 0])
                    for j in range(self.n_b - 1):
                        qml.CNOT(wires=[self.n_a + j, self.n_a + j + 1])
                    if self.n_b > 1:
                        qml.CNOT(wires=[self.n_a + self.n_b - 1, self.n_a])
                    for a_wire, b_wire in self.entangle_pairs:
                        qml.CNOT(wires=[a_wire, b_wire])
                    if layer < self.num_vqc_layers - 1:
                        for i in range(0, self.num_qubits - 1, 2):
                            qml.CZ(wires=[i, i + 1])
                return qml.state()

            self._state_qnode = _state_circuit

    # ── core fusion ──────────────────────────────────────────────────

    def _fuse(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Encode → quantum circuit → post-measure → (B, embed_dim)."""
        B = vision_embed.size(0)

        # Encode to rotation angles
        v_angles = torch.tanh(self.vision_encoder(vision_embed)) * np.pi  # (B, n_a*2)
        t_angles = torch.tanh(self.text_encoder(text_embed)) * np.pi     # (B, n_b*2)

        if self._using_pennylane:
            measurements = self._forward_pennylane(v_angles, t_angles, B)
        else:
            measurements = self.surrogate(vision_embed, text_embed)  # (B, num_qubits)

            # Phase 2: classical surrogate entanglement estimation
            if self.measure_entanglement:
                try:
                    report = self.get_classical_surrogate_entanglement(
                        vision_embed[0:1], text_embed[0:1],
                    )
                    self._last_entanglement_report = report
                    self._diag = {
                        "entanglement_entropy": report.von_neumann_entropy,
                        "linear_entropy": report.linear_entropy,
                        "subsystem_purity": report.subsystem_purity,
                        "mutual_information": report.mutual_information,
                        "schmidt_rank": report.schmidt_rank,
                        "normalised_entropy": report.normalised_entropy,
                        "circuit_depth": self._gate_analysis.depth_estimate,
                        "gate_count": self._gate_analysis.total_gates,
                        "cnot_count": self._gate_analysis.cnot_count,
                        "num_entangle_pairs": len(self.entangle_pairs),
                        "source": report.source,
                    }
                except Exception as exc:
                    logger.debug("Classical surrogate entanglement failed: %s", exc)

        fused = self.post_measure(measurements)  # (B, embed_dim)
        return fused

    def _forward_pennylane(
        self,
        v_angles: torch.Tensor,
        t_angles: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Execute QNode per sample (PennyLane doesn't natively batch)."""
        results = []
        entanglement_reports: list[EntanglementReport] = []

        for i in range(batch_size):
            expvals = self._qnode(v_angles[i], t_angles[i], self.var_weights)
            # expvals is a list of tensors
            row = torch.stack([e if isinstance(e, torch.Tensor) else torch.tensor(e, dtype=torch.float32)
                               for e in expvals])
            results.append(row)

            # Entanglement analysis (first sample only per batch, for speed)
            if self.measure_entanglement and i == 0:
                try:
                    report = self._compute_entanglement_report(v_angles[i], t_angles[i])
                    entanglement_reports.append(report)
                except Exception as exc:
                    logger.debug("Entanglement computation failed: %s", exc)

        measurements = torch.stack(results)  # (B, num_qubits)

        # Record diagnostics  (Phase 2 enhanced)
        if entanglement_reports:
            rep = entanglement_reports[0]
            self._last_entanglement_report = rep
            self._diag = {
                "entanglement_entropy": rep.von_neumann_entropy,
                "linear_entropy": rep.linear_entropy,
                "subsystem_purity": rep.subsystem_purity,
                "mutual_information": rep.mutual_information,
                "schmidt_rank": rep.schmidt_rank,
                "normalised_entropy": rep.normalised_entropy,
                "max_possible_entropy": rep.max_possible_entropy,
                "circuit_depth": self._gate_analysis.depth_estimate,
                "gate_count": self._gate_analysis.total_gates,
                "cnot_count": self._gate_analysis.cnot_count,
                "num_entangle_pairs": len(self.entangle_pairs),
                "source": rep.source,
            }
        else:
            self._diag = {
                "entanglement_entropy": None,
                "circuit_depth": self._gate_analysis.depth_estimate,
                "gate_count": self._gate_analysis.total_gates,
                "cnot_count": self._gate_analysis.cnot_count,
                "num_entangle_pairs": len(self.entangle_pairs),
            }

        return measurements

    # ── entanglement diagnostics (Phase 2) ───────────────────────────

    def _compute_entanglement_report(
        self,
        v_angles: torch.Tensor,
        t_angles: torch.Tensor,
    ) -> EntanglementReport:
        """Full entanglement report using fusion_core.quantum.entropy.

        Returns an ``EntanglementReport`` with von Neumann entropy,
        linear entropy, purity, mutual information, Schmidt rank, and
        the full eigenvalue spectrum of ρ_A.
        """
        if not hasattr(self, "_state_qnode"):
            # No state QNode → return empty report
            return EntanglementReport(source="unavailable")

        with torch.no_grad():
            state = self._state_qnode(v_angles, t_angles, self.var_weights)
            state_np = state.detach().cpu().numpy().flatten()

        return compute_entanglement(state_np, self.n_a, self.n_b)

    def _compute_entanglement(
        self,
        v_angles: torch.Tensor,
        t_angles: torch.Tensor,
    ) -> float:
        """Legacy API — returns scalar von Neumann entropy."""
        report = self._compute_entanglement_report(v_angles, t_angles)
        return report.von_neumann_entropy

    def _estimate_depth(self) -> int:
        """Circuit depth from static gate analysis."""
        return self._gate_analysis.depth_estimate

    def _estimate_gate_count(self) -> int:
        """Total gate count from static gate analysis."""
        return self._gate_analysis.total_gates

    def get_circuit_analysis(self) -> Dict[str, Any]:
        """Full circuit analysis report (Phase 2).

        Returns gate counts, hardware resource estimates, and the
        declarative circuit specification. Suitable for experiment logging.
        """
        return full_circuit_analysis(self.circuit_spec)

    def get_entanglement_report(self) -> Optional[Dict[str, Any]]:
        """Return the last entanglement report as a dict, or None."""
        if self._last_entanglement_report is not None:
            return self._last_entanglement_report.to_dict()
        return None

    def get_classical_surrogate_entanglement(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> EntanglementReport:
        """Compute a classical proxy entanglement report from embeddings.

        Useful when PennyLane is not available — provides a heuristic
        entanglement estimate based on SVD of the cross-correlation matrix.
        """
        v_np = vision_embed.detach().cpu().numpy().flatten()
        t_np = text_embed.detach().cpu().numpy().flatten()
        return classical_surrogate_entanglement(
            v_np, t_np,
            n_qubits_a=self.n_a,
            n_qubits_b=self.n_b,
        )

    # ── meta ─────────────────────────────────────────────────────────

    def _extra_meta(self) -> Dict[str, Any]:
        meta = {
            "num_qubits": self.num_qubits,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "num_layers": self.num_vqc_layers,
            "topology": self.topology_name,
            "encoding": self.encoding_name,
            "using_pennylane": self._using_pennylane,
            "entangle_pairs": len(self.entangle_pairs),
            # Static analysis
            "total_gates": self._gate_analysis.total_gates,
            "single_qubit_gates": self._gate_analysis.single_qubit_gates,
            "two_qubit_gates": self._gate_analysis.two_qubit_gates,
            "cnot_count": self._gate_analysis.cnot_count,
            "depth_estimate": self._gate_analysis.depth_estimate,
            "total_parameters": self._gate_analysis.total_parameters,
        }
        # Forward-pass diagnostics
        meta.update(self._diag)
        return meta
