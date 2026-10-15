"""
Tests for argument validation and unit conversion in cvpack collective variables.
"""

import numpy as np
import openmm
import pytest
from openmm import unit
from openmmtools import testsystems

import cvpack
from cvpack.tests.helpers import make_context


def test_openmm_force_wrapper_rejects_non_md_unit():
    angle = openmm.CustomAngleForce("theta")
    angle.addAngle(0, 1, 2)
    with pytest.raises(ValueError, match="MD unit system"):
        cvpack.OpenMMForceWrapper(angle, unit.degree)


def test_meta_cv_rejects_non_md_unit():
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    with pytest.raises(ValueError, match="MD unit system"):
        cvpack.MetaCollectiveVariable("phi", [phi], unit.kilocalorie_per_mole)


def _dipeptide_path(milestones, sigma, scales=None):
    model = testsystems.AlanineDipeptideVacuum()
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    psi = cvpack.Torsion(8, 14, 16, 18, name="psi")
    cv = cvpack.PathInCVSpace(
        cvpack.path.progress, [phi, psi], milestones, sigma, scales
    )
    cv.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    return cv.getValue(context).value_in_unit(unit.dimensionless)


RADIANS = np.array([[1.3, -0.2], [1.2, 3.1], [-2.7, 2.9], [-1.3, 2.7]])


def test_path_in_cv_space_converts_milestone_units():
    expected = _dipeptide_path(RADIANS, np.pi / 6)
    in_degrees = np.degrees(RADIANS) * unit.degree
    assert _dipeptide_path(in_degrees, np.pi / 6) == pytest.approx(expected)
    rows = [[x * unit.radian for x in row] for row in RADIANS]
    assert _dipeptide_path(rows, np.pi / 6) == pytest.approx(expected)


def test_path_in_cv_space_converts_scale_and_sigma_units():
    expected = _dipeptide_path(RADIANS, 0.5, scales=[0.5, 0.5])
    value = _dipeptide_path(
        RADIANS,
        0.5 * unit.dimensionless,
        scales=[0.5 * unit.radian, np.degrees(0.5) * unit.degree],
    )
    assert value == pytest.approx(expected)


def test_number_of_contacts_accepts_dimensionless_quantity_reference():
    model = testsystems.HostGuestVacuum()
    nonbonded = next(
        f for f in model.system.getForces() if isinstance(f, openmm.NonbondedForce)
    )
    residues = list(model.topology.residues())
    host = [a.index for a in residues[0].atoms()]
    guest = [a.index for a in residues[1].atoms()]
    cv1 = cvpack.NumberOfContacts(host, guest, nonbonded, reference=4.0, name="a")
    cv2 = cvpack.NumberOfContacts(
        host, guest, nonbonded, reference=4.0 * unit.dimensionless, name="b"
    )
    cv1.addToSystem(model.system)
    cv2.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    assert cv2.getValue(context) == cv1.getValue(context)
