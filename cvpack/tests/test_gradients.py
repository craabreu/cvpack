"""
Finite-difference checks of the forces of cvpack collective variables.

The force of a collective variable is minus its gradient, and effective masses are
computed from it, so an error in a derivative would go unnoticed by value tests.
"""

import itertools as it

import numpy as np
import openmm
import pytest
from openmm import unit
from openmmtools import testsystems

import cvpack
from cvpack.tests.helpers import make_context, particle_system

STEP = 1e-5  # nm
MAX_ATOMS = 8  # atoms checked per CV, chosen by force magnitude


def assert_forces_match_finite_differences(cv, system, positions):
    """Compare the forces of a CV with central differences of its value."""
    cv.addToSystem(system)
    positions = np.array(positions, dtype=float)
    context = make_context(system, positions)
    state = context.getState(getForces=True, groups={cv.getForceGroup()})
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    magnitudes = np.linalg.norm(forces, axis=1)
    atoms = np.argsort(-magnitudes)[:MAX_ATOMS]
    assert magnitudes[atoms[0]] > 0
    gradient = np.empty((len(atoms), 3))
    for (n, atom), k in it.product(enumerate(atoms), range(3)):
        values = []
        for sign in (1, -1):
            displaced = positions.copy()
            displaced[atom, k] += sign * STEP
            context.setPositions(displaced)
            values.append(
                cv.getValue(context).value_in_unit_system(unit.md_unit_system)
            )
        gradient[n, k] = (values[0] - values[1]) / (2 * STEP)
    scale = np.max(np.abs(gradient))
    assert -forces[atoms] == pytest.approx(gradient, rel=1e-5, abs=1e-6 * scale)


@pytest.fixture(scope="module")
def dipeptide():
    model = testsystems.AlanineDipeptideVacuum()
    positions = model.positions.value_in_unit(unit.nanometers)
    noise = np.random.default_rng(23).normal(0, 0.01, positions.shape)
    return model, positions + noise


@pytest.fixture(scope="module")
def lysozyme():
    model = testsystems.LysozymeImplicit()
    return model, model.positions.value_in_unit(unit.nanometers)


def _phi_psi():
    return [
        cvpack.Torsion(4, 6, 8, 14, name="phi"),
        cvpack.Torsion(6, 8, 14, 16, name="psi"),
    ]


def _nonbonded(model):
    return next(
        f for f in model.system.getForces() if isinstance(f, openmm.NonbondedForce)
    )


def _carbons_and_oxygens(model):
    elements = [atom.element.symbol for atom in model.topology.atoms()]
    return [
        [i for i, symbol in enumerate(elements) if symbol == element]
        for element in ("C", "O")
    ]


DIPEPTIDE_CVS = {
    "Distance": lambda model: cvpack.Distance(0, 5),
    "Angle": lambda model: cvpack.Angle(0, 5, 10),
    "Torsion": lambda model: cvpack.Torsion(4, 6, 8, 14),
    "RadiusOfGyration": lambda model: cvpack.RadiusOfGyration(range(22)),
    "RadiusOfGyrationSq": lambda model: cvpack.RadiusOfGyrationSq(
        range(22), weighByMass=True
    ),
    "NumberOfContacts": lambda model: cvpack.NumberOfContacts(
        *_carbons_and_oxygens(model), _nonbonded(model)
    ),
    "ShortestDistance": lambda model: cvpack.ShortestDistance(
        range(10), range(12, 22), 22, beta=3
    ),
    "RMSD": lambda model: cvpack.RMSD(model.positions, range(22), 22),
    "AtomicFunction": lambda model: cvpack.AtomicFunction(
        "distance(p1, p2)*angle(p1, p2, p3)", unit.nanometer * unit.radian, [[0, 5, 10]]
    ),
    "CentroidFunction": lambda model: cvpack.CentroidFunction(
        "distance(g1, g2)", unit.nanometer, [range(0, 6), range(14, 22)]
    ),
    "TorsionSimilarity": lambda model: cvpack.TorsionSimilarity(
        [[4, 6, 8, 14]], [[6, 8, 14, 16]]
    ),
    "PathInCVSpace": lambda model: cvpack.PathInCVSpace(
        cvpack.path.progress, _phi_psi(), [[1.3, -0.2], [1.2, 3.1], [-2.7, 2.9]], 0.5
    ),
    "PathInRMSDSpace": lambda model: cvpack.PathInRMSDSpace(
        cvpack.path.deviation,
        [
            dict(enumerate(model.positions / unit.nanometer + noise))
            for noise in np.random.default_rng(31).normal(0, 0.02, (3, 22, 3))
        ],
        22,
        0.05,
    ),
    "MetaCollectiveVariable": lambda model: cvpack.MetaCollectiveVariable(
        "cos(phi)*sin(psi)", _phi_psi(), unit.dimensionless
    ),
    "OpenMMForceWrapper": lambda model: cvpack.OpenMMForceWrapper(
        openmm.CustomCompoundBondForce(3, "angle(p1, p2, p3)"), unit.radian
    ),
}


@pytest.mark.parametrize("name", DIPEPTIDE_CVS)
def test_dipeptide_cv_forces(dipeptide, name):
    model, positions = dipeptide
    cv = DIPEPTIDE_CVS[name](model)
    if name == "OpenMMForceWrapper":
        cv.addBond([0, 5, 10], [])
    system = openmm.System()
    for atom in range(model.system.getNumParticles()):
        system.addParticle(model.system.getParticleMass(atom))
    assert_forces_match_finite_differences(cv, system, positions)


def _residues(model, start, end):
    return list(it.islice(model.topology.residues(), start, end))


LYSOZYME_CVS = {
    "HelixAngleContent": lambda model: cvpack.HelixAngleContent(
        _residues(model, 59, 80)
    ),
    "HelixTorsionContent": lambda model: cvpack.HelixTorsionContent(
        _residues(model, 59, 80)
    ),
    "HelixHBondContent": lambda model: cvpack.HelixHBondContent(
        _residues(model, 59, 80)
    ),
    "HelixRMSDContent": lambda model: cvpack.HelixRMSDContent(
        _residues(model, 39, 51), model.system.getNumParticles()
    ),
    "SheetRMSDContent": lambda model: cvpack.SheetRMSDContent(
        _residues(model, 0, 14), model.system.getNumParticles()
    ),
    "ResidueCoordination": lambda model: cvpack.ResidueCoordination(
        _residues(model, 125, 132), _residues(model, 142, 150)
    ),
}


@pytest.mark.parametrize("name", LYSOZYME_CVS)
def test_lysozyme_cv_forces(lysozyme, name):
    model, positions = lysozyme
    system = particle_system(model.system.getNumParticles())
    for atom in range(system.getNumParticles()):
        system.setParticleMass(atom, model.system.getParticleMass(atom))
    assert_forces_match_finite_differences(LYSOZYME_CVS[name](model), system, positions)


def test_attraction_strength_forces():
    rng = np.random.default_rng(29)
    num_particles = 10
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    nonbonded.setCutoffDistance(1.0)
    for _ in range(num_particles):
        nonbonded.addParticle(rng.uniform(-1, 1), rng.uniform(0.2, 0.4), 1.0)
    cv = cvpack.AttractionStrength(range(5), range(5, 10), nonbonded)
    positions = rng.uniform(0, 1.2, (num_particles, 3))
    assert_forces_match_finite_differences(
        cv, particle_system(num_particles), positions
    )


# Strict, so that this test fails as a reminder to remove the marker once a fixed
# openmm-cpp-forces package is released.
@pytest.mark.xfail(
    strict=True,
    reason="openmm-cpp-forces scales the forces of each group by 1/n_j instead of 1/n "
    "(https://github.com/craabreu/openmm-cpp-forces/pull/22)",
)
def test_composite_rmsd_forces(dipeptide):
    model, positions = dipeptide
    try:
        cv = cvpack.CompositeRMSD(model.positions, [range(0, 10), range(10, 22)])
    except ImportError:
        pytest.skip("openmm-cpp-forces is not installed")
    assert_forces_match_finite_differences(cv, particle_system(22), positions)
