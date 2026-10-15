"""
Tests that collective variables survive copying and (de)serialization unchanged.
"""

import copy
import itertools as it
from collections import namedtuple

import numpy as np
import openmm
import pytest
import yaml
from openmm import app, unit
from openmmtools import testsystems

import cvpack
from cvpack.serialization import SerializableAtom
from cvpack.tests.helpers import make_context
from cvpack.units import Unit
from cvpack.utils import preprocess_args


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


def test_preprocess_args_converts_containers():
    @preprocess_args
    def identity(data):
        return data

    Pair = namedtuple("Pair", "a b")
    assert identity(openmm.Vec3(1.0, 2.0, 3.0)) == [1.0, 2.0, 3.0]
    assert identity(range(3)) == [0, 1, 2]
    assert identity(Pair(1, 2)) == [1, 2]
    converted = identity(np.array([[1, 2], [3, 4]]))
    assert converted == [[1, 2], [3, 4]]
    assert type(converted[0][0]) is int
    assert identity((1, 2)) == (1, 2)


def test_numpy_arguments_round_trip(dipeptide):
    forces = {f.getName(): f for f in dipeptide.system.getForces()}
    cvs = [
        cvpack.NumberOfContacts(
            np.array([0, 1, 2]), np.array([10, 11]), forces["NonbondedForce"]
        ),
        cvpack.ShortestDistance(np.arange(0, 5), np.arange(10, 15), 22),
        cvpack.AtomicFunction(
            "dihedral(p1, p2, p3, p4)",
            unit.radian,
            np.array([[6, 8, 14, 16]]),
            periodicBounds=np.array([-np.pi, np.pi]) * unit.radian,
            name="phi",
        ),
    ]
    for cv in cvs:
        cv.addToSystem(dipeptide.system)
    context = make_context(dipeptide.system, dipeptide.positions)
    for cv in cvs:
        clone = assert_round_trip(cv, context)
        assert clone.getPeriodicBounds() == cv.getPeriodicBounds()


def test_one_shot_iterables_round_trip(dipeptide):
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    psi = cvpack.Torsion(8, 14, 16, 18, name="psi")
    meta = cvpack.MetaCollectiveVariable(
        "phi+psi", (cv for cv in [phi, psi]), unit.radian
    )
    path = cvpack.PathInCVSpace(
        cvpack.path.progress,
        [phi, psi],
        [[1.3, -0.2], [1.2, 3.1]],
        0.5,
        scales=(s for s in [1.0, 2.0]),
    )
    for cv in (meta, path):
        cv.addToSystem(dipeptide.system)
    context = make_context(dipeptide.system, dipeptide.positions)
    assert_round_trip(meta, context)
    assert_round_trip(path, context)


@pytest.mark.parametrize("as_vec3", [False, True])
def test_path_in_rmsd_space_with_quantity_milestones(dipeptide, as_vec3):
    positions = dipeptide.positions  # numpy-backed Quantity
    atoms = list(range(10))

    def milestone(shift):
        if as_vec3:
            return {
                i: openmm.Vec3(*(positions[i].value_in_unit(unit.angstrom) + shift))
                * unit.angstrom
                for i in atoms
            }
        return {i: positions[i] + shift * unit.angstrom for i in atoms}

    cv = cvpack.PathInRMSDSpace(
        cvpack.path.progress, [milestone(0.0), milestone(0.5)], 22, 0.5 * unit.angstrom
    )
    cv.addToSystem(dipeptide.system)
    assert_round_trip(cv, make_context(dipeptide.system, dipeptide.positions))


def test_serializable_atom_without_element():
    topology = app.Topology()
    residue = topology.addResidue("HOH", topology.addChain())
    topology.addAtom("EPW", None, residue)
    atom = SerializableAtom(next(topology.atoms()))
    assert atom.element is None
    assert yaml.safe_load(yaml.safe_dump(atom)).element is None


@pytest.mark.parametrize(
    "mm_unit",
    [
        unit.elementary_charge,
        unit.year,
        unit.picosecond**-1,
        unit.kilojoule_per_mole / unit.radian**2,
        unit.nanometer**2 * unit.dalton / unit.radian**2,
        unit.dimensionless,
        unit.angstrom,
    ],
)
def test_unit_round_trip(mm_unit):
    loaded = yaml.safe_load(yaml.safe_dump(Unit(mm_unit)))
    assert isinstance(loaded, Unit)
    assert (1 * loaded) / mm_unit == pytest.approx(1.0)


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "nanometer**2*dalton/(radian**2)",
            unit.nanometer**2 * unit.dalton / unit.radian**2,
        ),
        ("kilojoule/(mole*radian**2)", unit.kilojoule_per_mole / unit.radian**2),
        ("/picosecond", unit.picosecond**-1),
        ("dimensionless", unit.dimensionless),
    ],
)
def test_legacy_unit_strings_still_load(text, expected):
    assert (1 * Unit(text)) / expected == pytest.approx(1.0)


@pytest.mark.parametrize(
    "text",
    [
        "nanometer.__class__",
        "__import__('os')",
        "nanometer.__class__.__init__.__globals__['__builtins__']['print']('x')"
        " or nanometer",
        "not_a_unit",
        "9**9**9",
        "2*nanometer",
    ],
)
def test_unit_rejects_unsafe_or_invalid_expressions(text):
    with pytest.raises(ValueError):
        Unit(text)
    with pytest.raises(ValueError):
        yaml.safe_load(f'!cvpack.Unit {{data: "{text}"}}')


def test_residue_coordination_numpy_reference_value_round_trips(lysozyme):
    groups = [
        list(it.islice(lysozyme.topology.residues(), start, end))
        for start, end in [(125, 142), (142, 156)]
    ]
    cv = cvpack.ResidueCoordination(*groups)
    cv.setReferenceValue(np.float64(7.0))
    clone = copy.deepcopy(cv)
    assert clone.getReferenceValue() == cv.getReferenceValue()


@pytest.mark.parametrize(
    "mm_unit, expected",
    [
        (unit.dalton, "dalton"),
        (unit.kilojoule_per_mole, "mole**-1*md_kilojoule"),
        (unit.nanometer**2, "nanometer**2"),
    ],
)
def test_units_serialize_with_their_own_names(mm_unit, expected):
    assert Unit(mm_unit).__getstate__()["data"] == expected
