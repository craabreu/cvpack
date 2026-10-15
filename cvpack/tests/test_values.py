"""
Regression tests for the values computed by cvpack collective variables.
"""

import numpy as np
import pytest

import cvpack
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
