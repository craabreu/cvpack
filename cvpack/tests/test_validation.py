"""
Tests for argument validation and unit conversion in cvpack collective variables.
"""

import openmm
import pytest
from openmm import unit

import cvpack


def test_openmm_force_wrapper_rejects_non_md_unit():
    angle = openmm.CustomAngleForce("theta")
    angle.addAngle(0, 1, 2)
    with pytest.raises(ValueError, match="MD unit system"):
        cvpack.OpenMMForceWrapper(angle, unit.degree)


def test_meta_cv_rejects_non_md_unit():
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    with pytest.raises(ValueError, match="MD unit system"):
        cvpack.MetaCollectiveVariable("phi", [phi], unit.kilocalorie_per_mole)
