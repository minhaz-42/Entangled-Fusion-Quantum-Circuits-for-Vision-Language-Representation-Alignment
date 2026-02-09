"""
Quantum — advanced quantum circuit utilities (Phase 2).

Provides:
  - ``EntangledCircuitSpec`` — declarative circuit specification
  - ``PennyLaneEntangledCircuit`` — executable PennyLane circuit builder
  - ``EntanglementReport`` / ``compute_entanglement`` — entropy diagnostics
  - ``analyse_gates`` / ``full_circuit_analysis`` — structural analysis
  - Topology generators (ring, full, linear, ladder)
"""

from .entangled_circuit import (
    EntangledCircuitSpec,
    TOPOLOGY_REGISTRY,
    ring_topology,
    full_topology,
    linear_topology,
    ladder_topology,
    PENNYLANE_AVAILABLE,
)

from .entropy import (
    EntanglementReport,
    compute_entanglement,
    classical_surrogate_entanglement,
    reduced_density_matrix,
    von_neumann_entropy,
    linear_entropy,
    purity,
    eigenvalue_spectrum,
    schmidt_rank,
)

from .circuit_analysis import (
    GateAnalysis,
    analyse_gates,
    estimate_expressibility,
    estimate_entangling_capability,
    hardware_resource_estimate,
    full_circuit_analysis,
)

# Conditionally export PennyLane circuit builder
if PENNYLANE_AVAILABLE:
    from .entangled_circuit import PennyLaneEntangledCircuit

__all__ = [
    "EntangledCircuitSpec",
    "TOPOLOGY_REGISTRY",
    "ring_topology",
    "full_topology",
    "linear_topology",
    "ladder_topology",
    "EntanglementReport",
    "compute_entanglement",
    "classical_surrogate_entanglement",
    "reduced_density_matrix",
    "von_neumann_entropy",
    "linear_entropy",
    "purity",
    "eigenvalue_spectrum",
    "schmidt_rank",
    "GateAnalysis",
    "analyse_gates",
    "estimate_expressibility",
    "estimate_entangling_capability",
    "hardware_resource_estimate",
    "full_circuit_analysis",
]
