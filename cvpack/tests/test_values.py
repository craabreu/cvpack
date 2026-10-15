"""
Regression tests for the values computed by cvpack collective variables.
"""

import numpy as np
import openmm
import pytest

import cvpack
from cvpack.attraction_strength import ONE_4PI_EPS0
from cvpack.tests.helpers import make_context, particle_system


@pytest.mark.parametrize("group", [[0, 2], [5, 6, 7], [7, 1, 4, 3]])
def test_radius_of_gyration_sq_with_atom_subset(group):
    positions = np.random.default_rng(1).uniform(0, 2, (8, 3))
    system = particle_system(8)
    cv = cvpack.RadiusOfGyrationSq(group)
    cv.addToSystem(system)
    context = make_context(system, positions)
    selected = positions[group]
    expected = np.mean(np.sum((selected - selected.mean(axis=0)) ** 2, axis=1))
    assert cv.getValue(context).value_in_unit(cv.getUnit()) == pytest.approx(expected)


@pytest.mark.parametrize("cutoff", [0.8, 1.0, 1.6])
def test_attraction_strength_coulomb_scales_with_cutoff(cutoff):
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    nonbonded.setCutoffDistance(cutoff)
    nonbonded.addParticle(1.0, 0.3, 0.0)
    nonbonded.addParticle(-1.0, 0.3, 0.0)
    system = particle_system(2)
    cv = cvpack.AttractionStrength([0], [1], nonbonded)
    cv.addToSystem(system)
    distance = 0.4
    context = make_context(system, [[0, 0, 0], [distance, 0, 0]])
    x = distance / cutoff
    # Docstring: -q_i q_j / (4 pi eps0 r_c) * u_elec(x), with u_elec(x) = 1/x + (x^2-3)/2
    expected = ONE_4PI_EPS0 / cutoff * (1 / x + (x**2 - 3) / 2)
    assert cv.getValue(context).value_in_unit(cv.getUnit()) == pytest.approx(expected)
