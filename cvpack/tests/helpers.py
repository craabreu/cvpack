"""
Helpers shared by the cvpack test modules.
"""

import numpy as np
import openmm
from openmm import app


def make_context(system: openmm.System, positions) -> openmm.Context:
    """Create a Reference-platform context for a system and set its positions."""
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, openmm.VerletIntegrator(0), platform)
    context.setPositions(positions)
    return context


def particle_system(num_particles: int, mass: float = 1.0) -> openmm.System:
    """Create a non-periodic system of identical particles (default box: 2 nm)."""
    system = openmm.System()
    for _ in range(num_particles):
        system.addParticle(mass)
    return system


def single_atom_topology(num_residues: int, residue_name: str = "ALA") -> app.Topology:
    """Create a single-chain topology in which every residue has one carbon atom."""
    topology = app.Topology()
    chain = topology.addChain()
    for _ in range(num_residues):
        residue = topology.addResidue(residue_name, chain)
        topology.addAtom("CA", app.element.carbon, residue)
    return topology


def shortest_distance_reference(positions, group1, group2, beta, rc, box=None):
    """Evaluate the ShortestDistance formula with NumPy, counting each pair once."""
    positions = np.asarray(positions)
    pairs = np.array(
        sorted({(min(i, j), max(i, j)) for i in group1 for j in group2 if i != j})
    )
    deltas = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    if box is not None:
        deltas -= box * np.round(deltas / box)
    distances = np.linalg.norm(deltas, axis=1)
    distances = distances[distances < rc]
    weights = np.exp(-beta * distances / rc)
    numerator = rc * np.exp(-beta) + np.sum(distances * weights)
    denominator = np.exp(-beta) + np.sum(weights)
    return numerator / denominator
