"""
Finite-difference checks of the forces of cvpack collective variables.

The force of a collective variable is minus its gradient, and effective masses are
computed from it, so an error in a derivative would go unnoticed by value tests.
"""

import itertools as it

import numpy as np
import pytest
from openmm import unit

from cvpack.tests.cv_cases import CASES
from cvpack.tests.helpers import make_context

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


@pytest.mark.parametrize("name", CASES)
def test_cv_forces(name):
    cv, system, positions = CASES[name]()
    assert_forces_match_finite_differences(cv, system, positions)
