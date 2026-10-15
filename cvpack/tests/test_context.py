"""
Tests for evaluating collective variables in OpenMM contexts and reporting them.
"""

import pytest
from openmm import unit
from openmmtools import testsystems

import cvpack
from cvpack.tests.helpers import make_context


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
