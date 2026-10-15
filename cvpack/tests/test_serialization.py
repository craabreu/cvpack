"""
Tests that collective variables survive copying and (de)serialization unchanged.
"""

import copy
import itertools as it

import pytest
import yaml
from openmm import unit
from openmmtools import testsystems

import cvpack
from cvpack.tests.helpers import make_context


def assert_round_trip(cv, context):
    """Deep-copy a CV that is already in the context's system and compare values."""
    clone = copy.deepcopy(cv)
    assert yaml.safe_dump(clone) == yaml.safe_dump(cv)
    assert clone.getName() == cv.getName()
    clone.addToSystem(context.getSystem())
    context.reinitialize(preserveState=True)
    original = cv.getValue(context).value_in_unit(cv.getUnit())
    copied = clone.getValue(context).value_in_unit(clone.getUnit())
    assert copied == pytest.approx(original)
    return clone


@pytest.fixture
def lysozyme():
    return testsystems.LysozymeImplicit()


@pytest.fixture
def dipeptide():
    return testsystems.AlanineDipeptideVacuum()


def test_helix_torsion_content_keeps_normalization(lysozyme):
    residues = [r for r in lysozyme.topology.residues() if 59 <= r.index <= 79]
    cv = cvpack.HelixTorsionContent(residues, normalize=True)
    cv.addToSystem(lysozyme.system)
    assert_round_trip(cv, make_context(lysozyme.system, lysozyme.positions))


def test_torsion_similarity_keeps_pbc():
    cv = cvpack.TorsionSimilarity([[0, 1, 2, 3]], [[1, 2, 3, 4]], pbc=True)
    assert copy.deepcopy(cv).usesPeriodicBoundaryConditions()


def test_residue_coordination_keeps_reference_value(lysozyme):
    groups = [
        list(it.islice(lysozyme.topology.residues(), start, end))
        for start, end in [(125, 142), (142, 156)]
    ]
    cv = cvpack.ResidueCoordination(*groups)
    cv.setReferenceValue(5.0)
    cv.addToSystem(lysozyme.system)
    context = make_context(lysozyme.system, lysozyme.positions)
    clone = assert_round_trip(cv, context)
    assert clone.getReferenceValue() == cv.getReferenceValue()


def test_renamed_cv_keeps_its_name_inside_meta_cv(dipeptide):
    phi = cvpack.Torsion(6, 8, 14, 16)
    phi.setName("phi")
    assert copy.copy(phi).getName() == "phi"
    meta = cvpack.MetaCollectiveVariable("2*phi", [phi], unit.radian)
    meta.addToSystem(dipeptide.system)
    phi.addToSystem(dipeptide.system)
    context = make_context(dipeptide.system, dipeptide.positions)
    assert meta.getValue(context).value_in_unit(unit.radian) == pytest.approx(
        2 * phi.getValue(context).value_in_unit(unit.radian)
    )
    assert_round_trip(meta, context)


def test_nested_renamed_cvs_round_trip(dipeptide):
    phi = cvpack.Torsion(6, 8, 14, 16)
    psi = cvpack.Torsion(8, 14, 16, 18)
    phi.setName("phi")
    psi.setName("psi")
    path = cvpack.PathInCVSpace(
        cvpack.path.progress,
        [phi, psi],
        [[1.3, -0.2], [1.2, 3.1], [-2.7, 2.9]],
        0.5,
        name="s",
    )
    meta = cvpack.MetaCollectiveVariable("10*s", [path], unit.dimensionless)
    meta.addToSystem(dipeptide.system)
    assert_round_trip(meta, make_context(dipeptide.system, dipeptide.positions))
