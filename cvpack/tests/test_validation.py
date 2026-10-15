"""
Tests for argument validation and unit conversion in cvpack collective variables.
"""

import itertools as it
import sys
import types

import numpy as np
import openmm
import pytest
from openmm import app, unit
from openmmtools import testsystems

import cvpack
from cvpack.tests.helpers import make_context, single_atom_topology


def test_openmm_force_wrapper_rejects_non_md_unit():
    angle = openmm.CustomAngleForce("theta")
    angle.addAngle(0, 1, 2)
    with pytest.raises(ValueError, match="MD unit system"):
        cvpack.OpenMMForceWrapper(angle, unit.degree)


def test_meta_cv_rejects_non_md_unit():
    phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
    with pytest.raises(ValueError, match="MD unit system"):
        cvpack.MetaCollectiveVariable("phi", [phi], unit.kilocalorie_per_mole)


def _dipeptide_path(milestones, sigma, scales=None):
    model = testsystems.AlanineDipeptideVacuum()
    phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
    psi = cvpack.Torsion(6, 8, 14, 16, name="psi")
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


@pytest.mark.parametrize("k", [[1.0, 2.0, 3.0], [1.0]])
def test_atomic_function_rejects_wrong_parameter_length(k):
    with pytest.raises(ValueError, match="k"):
        cvpack.AtomicFunction(
            "k*distance(p1, p2)", unit.kilojoule_per_mole, [[0, 1], [2, 3]], k=k
        )


@pytest.mark.parametrize("cls", ["RMSD", "CompositeRMSD"])
def test_rmsd_dict_reference_requires_num_atoms(cls):
    reference = {0: [0, 0, 0], 3: [1, 0, 0], 5: [0, 1, 0]}
    try:
        with pytest.raises(ValueError, match="numAtoms"):
            if cls == "RMSD":
                cvpack.RMSD(reference, [0, 3, 5])
            else:
                cvpack.CompositeRMSD(reference, [[0, 3], [5]])
    except ImportError:
        pytest.skip("openmm-cpp-forces is not installed")


@pytest.mark.parametrize(
    "version, accepted",
    [("0.1.1", False), (None, False), ("0.2.0", True), ("0.2.0.post2", True)],
)
def test_composite_rmsd_requires_fixed_openmm_cpp_forces(
    monkeypatch, version, accepted
):
    # Versions before 0.2.0 compute wrong forces for groups of different sizes
    class FakeForce:
        pass

    fake = types.ModuleType("openmmcppforces")
    fake.CompositeRMSDForce = FakeForce
    if version is not None:
        fake.__version__ = version
    monkeypatch.setitem(sys.modules, "openmmcppforces", fake)
    force_class = cvpack.composite_rmsd._get_force_class()
    if accepted:
        assert force_class is FakeForce
    else:
        with pytest.raises(ImportError, match="0.2.0 or later"):
            force_class(None)


def test_residue_coordination_rejects_overlapping_groups():
    residues = list(single_atom_topology(5).residues())
    with pytest.raises(ValueError, match="disjoint"):
        cvpack.ResidueCoordination(residues[0:3], residues[2:5])


def test_attraction_strength_rejects_overlapping_contrast_group():
    nonbonded = openmm.NonbondedForce()
    for _ in range(4):
        nonbonded.addParticle(0.0, 0.3, 1.0)
    with pytest.raises(ValueError, match="contrast"):
        cvpack.AttractionStrength([0], [1, 2], nonbonded, contrastGroup=[2, 3])
    with pytest.raises(ValueError, match="contrastScaling"):
        cvpack.AttractionStrength(
            [0], [1], nonbonded, contrastGroup=[3], contrastScaling=-1.0
        )


def test_attraction_strength_rejects_nonbonded_force_without_cutoff():
    nonbonded = openmm.NonbondedForce()
    for _ in range(2):
        nonbonded.addParticle(0.0, 0.3, 1.0)
    with pytest.raises(ValueError, match="cutoff"):
        cvpack.AttractionStrength([0], [1], nonbonded)


def test_torsion_similarity_rejects_lists_of_different_lengths():
    with pytest.raises(ValueError, match="same length"):
        cvpack.TorsionSimilarity([[0, 1, 2, 3], [1, 2, 3, 4]], [[2, 3, 4, 5]])


@pytest.fixture(scope="module")
def lysozyme_residues():
    return list(testsystems.LysozymeImplicit().topology.residues())


@pytest.mark.parametrize(
    "factory",
    [
        cvpack.HelixAngleContent,
        cvpack.HelixTorsionContent,
        cvpack.HelixHBondContent,
        lambda residues: cvpack.HelixRMSDContent(residues, 5000),
    ],
)
def test_helix_cvs_reject_non_contiguous_residues(lysozyme_residues, factory):
    residues = lysozyme_residues[59:66] + lysozyme_residues[67:75]
    with pytest.raises(ValueError, match="contiguous"):
        factory(residues)


def test_sheet_rmsd_content_checks_each_block(lysozyme_residues):
    residues = lysozyme_residues[0:8] + lysozyme_residues[20:28]
    cvpack.SheetRMSDContent(residues, 5000, blockSizes=[8, 8])
    with pytest.raises(ValueError, match="contiguous"):
        cvpack.SheetRMSDContent(residues, 5000)


def test_contiguity_check_detects_chain_breaks():
    topology = app.Topology()
    for _ in range(2):
        chain = topology.addChain()
        for _ in range(3):
            residue = topology.addResidue("ALA", chain)
            topology.addAtom("CA", app.element.carbon, residue)
    with pytest.raises(ValueError, match="contiguous"):
        cvpack.utils.check_contiguous_residues(list(topology.residues()))


@pytest.mark.parametrize(
    "factory, size",
    [
        (cvpack.HelixAngleContent, 2),
        (cvpack.HelixTorsionContent, 2),
        (cvpack.HelixHBondContent, 4),
    ],
)
def test_normalized_helix_cvs_reject_short_sequences(lysozyme_residues, factory, size):
    with pytest.raises(ValueError, match="at least"):
        factory(lysozyme_residues[60 : 60 + size], normalize=True)


def test_helix_hbond_content_skips_proline_donors(lysozyme_residues):
    proline = next(r for r in lysozyme_residues if r.name == "PRO" and r.index >= 10)
    residues = lysozyme_residues[proline.index - 6 : proline.index + 3]
    cv = cvpack.HelixHBondContent(residues, normalize=True)
    expected_bonds = sum(1 for r in residues[4:] if r.name != "PRO")
    assert cv.getNumBonds() == expected_bonds == len(residues) - 5


def test_from_openmm_force_inherits_pbc():
    force = openmm.CustomBondForce("r")
    force.addBond(0, 1)
    force.setUsesPeriodicBoundaryConditions(True)
    cv = cvpack.AtomicFunction.fromOpenMMForce(force, unit.nanometer)
    assert cv.usesPeriodicBoundaryConditions()
    cv = cvpack.AtomicFunction.fromOpenMMForce(force, unit.nanometer, pbc=False)
    assert not cv.usesPeriodicBoundaryConditions()


def test_from_openmm_force_rejects_tabulated_functions():
    force = openmm.CustomCompoundBondForce(2, "tab(distance(p1, p2))")
    force.addTabulatedFunction(
        "tab", openmm.Continuous1DFunction([0.0, 1.0, 2.0], 0.0, 2.0)
    )
    force.addBond([0, 1], [])
    with pytest.raises(ValueError, match="tabulated"):
        cvpack.AtomicFunction.fromOpenMMForce(force, unit.kilojoule_per_mole)


def _kabsch_rmsd(p, q):
    u, s, vt = np.linalg.svd(p.T @ q)
    s[-1] *= np.sign(np.linalg.det(u @ vt))
    return np.sqrt(max((np.sum(p**2) + np.sum(q**2) - 2 * np.sum(s)) / len(p), 0.0))


def test_composite_rmsd_value():
    model = testsystems.HostGuestExplicit()
    host, guest = (
        [a.index for a in r.atoms()] for r in it.islice(model.topology.residues(), 0, 2)
    )
    reference = model.positions.value_in_unit(unit.nanometers)
    try:
        cv = cvpack.CompositeRMSD(reference, [host, guest])
    except ImportError:
        pytest.skip("openmm-cpp-forces is not installed")
    cv.addToSystem(model.system)
    rng = np.random.default_rng(7)
    current = reference + rng.normal(0, 0.02, reference.shape)
    current[guest] += 0.7
    context = make_context(model.system, current)
    centered = [
        np.vstack([x[g] - x[g].mean(axis=0) for g in (host, guest)])
        for x in (current, reference)
    ]
    expected = _kabsch_rmsd(*centered)
    value = cv.getValue(context).value_in_unit(unit.nanometer)
    assert value == pytest.approx(expected, rel=1e-5)


def test_from_openmm_force_rejects_non_forces():
    with pytest.raises(TypeError):
        cvpack.AtomicFunction.fromOpenMMForce("not a force", unit.nanometer)


def test_rmsd_rejects_empty_group():
    with pytest.raises(ValueError, match="empty"):
        cvpack.RMSD({0: [0, 0, 0]}, [], 3)


def test_composite_rmsd_rejects_empty_groups():
    reference = np.zeros((4, 3))
    with pytest.raises(ValueError, match="empty"):
        cvpack.CompositeRMSD(reference, [[0, 1], []])


def test_meta_cv_rejects_duplicate_names():
    first, second = cvpack.Distance(0, 1), cvpack.Distance(1, 2)
    with pytest.raises(ValueError, match="unique"):
        cvpack.MetaCollectiveVariable("distance", [first, second], unit.nanometer)


def test_path_in_cv_space_wraps_milestones_of_periodic_cvs():
    expected = _dipeptide_path(RADIANS, np.pi / 6)
    for shift in (2 * np.pi, 4 * np.pi, -6 * np.pi):
        assert _dipeptide_path(RADIANS + shift, np.pi / 6) == pytest.approx(expected)


def test_from_openmm_force_rejects_periodicdistance():
    force = openmm.CustomExternalForce("periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    force.addParticle(0, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="periodicdistance"):
        cvpack.AtomicFunction.fromOpenMMForce(force, unit.nanometer)
