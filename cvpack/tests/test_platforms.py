"""
Comparisons between the Reference platform and the other OpenMM platforms.

GPU platforms accumulate forces in 64-bit fixed point, so forces outside a range of
roughly 1e-10 to 1e9 are lost. Tests for platforms that are not available are skipped.
"""

import numpy as np
import openmm
import pytest
from openmm import unit
from openmmtools import testsystems

import cvpack
from cvpack.tests.cv_cases import CASES

PLATFORMS = [
    (name, precision)
    for name, precisions in [
        ("CPU", [None]),
        ("CUDA", ["single", "mixed", "double"]),
        ("OpenCL", ["single", "mixed", "double"]),
    ]
    for precision in precisions
]


def _context(system, positions, name, precision=None):
    try:
        platform = openmm.Platform.getPlatformByName(name)
        properties = {} if precision is None else {"Precision": precision}
        context = openmm.Context(
            system, openmm.VerletIntegrator(0), platform, properties
        )
    except openmm.OpenMMException:
        pytest.skip(f"{name} platform is not available")
    context.setPositions(positions)
    return context


def _value_and_forces(cv, context):
    state = context.getState(getForces=True, groups={cv.getForceGroup()})
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    return cv.getValue(context).value_in_unit(cv.getUnit()), forces


@pytest.fixture(scope="module")
def host_guest():
    model = testsystems.HostGuestVacuum()
    host, guest = [], []
    for residue in model.topology.residues():
        group = host if residue.name == "B2" else guest
        group.extend(atom.index for atom in residue.atoms())
    return model, host, guest


@pytest.mark.parametrize("name, precision", PLATFORMS)
def test_shortest_distance_forces_match_reference(host_guest, name, precision):
    model, host, guest = host_guest
    system = openmm.System()
    for atom in range(model.system.getNumParticles()):
        system.addParticle(model.system.getParticleMass(atom))
    cv = cvpack.ShortestDistance(host, guest, system.getNumParticles())
    cv.addToSystem(system)
    positions = model.positions.value_in_unit(unit.nanometers)
    axis = positions[guest].mean(axis=0) - positions[host].mean(axis=0)
    axis /= np.linalg.norm(axis)
    # The guest moves out of the host, so the shortest distance spans 0.03-0.8 nm
    for shift in (-0.5, 0.0, 1.1, 1.2, 1.3, 1.5, 1.7):
        coords = positions.copy()
        coords[guest] += shift * axis
        value, forces = _value_and_forces(cv, _context(system, coords, name, precision))
        expected, reference = _value_and_forces(
            cv, _context(system, coords, "Reference")
        )
        assert value == pytest.approx(expected, rel=1e-5)
        error = np.linalg.norm(forces - reference) / np.linalg.norm(reference)
        assert error < 1e-4, f"shift = {shift}"


# Relative tolerances for values and for the norm of the force error. They are about
# ten times the largest deviations measured on CUDA and OpenCL.
TOLERANCES = {None: (1e-6, 1e-5), "double": (1e-6, 1e-5), "mixed": (2e-5, 2e-4)}
TOLERANCES["single"] = TOLERANCES["mixed"]


@pytest.mark.parametrize("name, precision", PLATFORMS)
@pytest.mark.parametrize("case", CASES)
def test_cv_matches_reference(case, name, precision):
    try:
        cv, system, positions = CASES[case]()
    except ImportError:
        pytest.skip("openmm-cpp-forces is not installed")
    cv.addToSystem(system)
    value, forces = _value_and_forces(cv, _context(system, positions, name, precision))
    expected, reference = _value_and_forces(
        cv, _context(system, positions, "Reference")
    )
    value_tolerance, force_tolerance = TOLERANCES[precision]
    assert value == pytest.approx(expected, rel=value_tolerance)
    error = np.linalg.norm(forces - reference) / np.linalg.norm(reference)
    assert error < force_tolerance
