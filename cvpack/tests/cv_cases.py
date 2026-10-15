"""
One configuration of every collective variable class, shared by test modules that
check all of them.

Each case builds a new collective variable, a system without other forces and the
positions at which to evaluate it.
"""

import functools
import itertools as it

import numpy as np
import openmm
from openmm import unit
from openmmtools import testsystems

import cvpack
from cvpack.tests.helpers import particle_system


@functools.cache
def _dipeptide():
    model = testsystems.AlanineDipeptideVacuum()
    positions = model.positions.value_in_unit(unit.nanometers)
    noise = np.random.default_rng(23).normal(0, 0.01, positions.shape)
    return model, positions + noise


@functools.cache
def _lysozyme():
    model = testsystems.LysozymeImplicit()
    return model, model.positions.value_in_unit(unit.nanometers)


def _bare_system(model):
    system = particle_system(model.system.getNumParticles())
    for atom in range(system.getNumParticles()):
        system.setParticleMass(atom, model.system.getParticleMass(atom))
    return system


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


def _residues(model, start, end):
    return list(it.islice(model.topology.residues(), start, end))


def _angle_wrapper():
    force = openmm.CustomCompoundBondForce(3, "angle(p1, p2, p3)")
    force.addBond([0, 5, 10], [])
    return cvpack.OpenMMForceWrapper(force, unit.radian)


DIPEPTIDE_CVS = {
    "Distance": lambda model: cvpack.Distance(0, 5),
    "Angle": lambda model: cvpack.Angle(0, 5, 10),
    "Torsion": lambda model: cvpack.Torsion(4, 6, 8, 14),
    "RadiusOfGyration": lambda model: cvpack.RadiusOfGyration(range(22)),
    "RadiusOfGyrationSq": lambda model: cvpack.RadiusOfGyrationSq(range(22)),
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
    "OpenMMForceWrapper": lambda model: _angle_wrapper(),
}

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


def _attraction_strength():
    rng = np.random.default_rng(29)
    num_particles = 10
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    nonbonded.setCutoffDistance(1.0)
    for _ in range(num_particles):
        nonbonded.addParticle(rng.uniform(-1, 1), rng.uniform(0.2, 0.4), 1.0)
    cv = cvpack.AttractionStrength(range(5), range(5, 10), nonbonded)
    positions = rng.uniform(0, 1.2, (num_particles, 3))
    return cv, particle_system(num_particles), positions


def _composite_rmsd():
    """Raises ImportError if openmm-cpp-forces is not installed."""
    model, positions = _dipeptide()
    cv = cvpack.CompositeRMSD(model.positions, [range(0, 10), range(10, 22)])
    return cv, particle_system(22), positions


def _model_case(factory, model_loader):
    def case():
        model, positions = model_loader()
        return factory(model), _bare_system(model), positions

    return case


CASES = {
    **{
        name: _model_case(factory, _dipeptide)
        for name, factory in DIPEPTIDE_CVS.items()
    },
    **{name: _model_case(factory, _lysozyme) for name, factory in LYSOZYME_CVS.items()},
    "AttractionStrength": _attraction_strength,
    "CompositeRMSD": _composite_rmsd,
}
