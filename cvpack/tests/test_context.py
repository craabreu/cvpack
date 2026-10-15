"""
Tests for evaluating collective variables in OpenMM contexts and reporting them.
"""

import pytest
from openmm import unit
from openmmtools import testsystems

import cvpack
from cvpack.tests.helpers import make_context, particle_system


@pytest.fixture
def dipeptide():
    return testsystems.AlanineDipeptideVacuum()


def test_allow_reinitialization_restores_force_group(dipeptide):
    shared = cvpack.Torsion(6, 8, 14, 16, name="shared")
    shared.addToSystem(dipeptide.system, setUnusedForceGroup=False)
    alone = cvpack.Torsion(6, 8, 14, 16, name="alone")
    alone.addToSystem(dipeptide.system)
    context = make_context(dipeptide.system, dipeptide.positions)
    with pytest.raises(ValueError, match="reinitialization"):
        shared.getValue(context)
    value = shared.getValue(context, allowReinitialization=True)
    assert value / unit.radian == pytest.approx(alone.getValue(context) / unit.radian)
    mass = shared.getEffectiveMass(context, allowReinitialization=True)
    expected = alone.getEffectiveMass(context)
    assert mass / mass.unit == pytest.approx(expected / expected.unit)
    assert shared.getForceGroup() == 0


def test_set_unused_force_group_returns_group(dipeptide):
    cv = cvpack.Torsion(6, 8, 14, 16)
    group = cv._setUnusedForceGroup(dipeptide.system)
    assert group == cv.getForceGroup() == 1


def test_effective_mass_ignores_massless_particles():
    system = particle_system(2)
    system.setParticleMass(0, 0.0)
    system.setParticleMass(1, 10.0)
    cv = cvpack.Distance(0, 1)
    cv.addToSystem(system)
    context = make_context(system, [[0, 0, 0], [0.5, 0, 0]])
    assert cv.getEffectiveMass(context) / unit.dalton == pytest.approx(10.0)


def test_inner_effective_masses_follow_current_positions(dipeptide):
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    meta = cvpack.MetaCollectiveVariable("phi", [phi], unit.radian, name="meta")
    meta.addToSystem(dipeptide.system)
    direct = cvpack.Torsion(6, 8, 14, 16, name="direct")
    direct.addToSystem(dipeptide.system)
    context = make_context(dipeptide.system, dipeptide.positions)
    for shift in (0.0, 0.05):
        positions = dipeptide.positions.value_in_unit(unit.nanometers).copy()
        positions[6] += shift
        context.setPositions(positions)
        inner = meta.getInnerEffectiveMasses(context)["phi"]
        expected = direct.getEffectiveMass(context)
        assert inner / inner.unit == pytest.approx(expected / expected.unit)
