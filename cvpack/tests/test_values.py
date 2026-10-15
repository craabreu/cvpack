"""
Regression tests for the values computed by cvpack collective variables.
"""

import numpy as np
import openmm
import pytest
from openmm import app, unit
from openmmtools import testsystems

import cvpack
from cvpack.attraction_strength import ONE_4PI_EPS0
from cvpack.base_rmsd_content import BaseRMSDContent
from cvpack.tests.helpers import (
    make_context,
    particle_system,
    shortest_distance_reference,
    single_atom_topology,
)


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
    # Docstring: -q_i q_j/(4 pi eps0 r_c) u_elec(x), with u_elec(x) = 1/x + (x^2-3)/2
    expected = ONE_4PI_EPS0 / cutoff * (1 / x + (x**2 - 3) / 2)
    assert cv.getValue(context).value_in_unit(cv.getUnit()) == pytest.approx(expected)


def test_residue_coordination_defaults_to_no_pbc():
    residues = list(single_atom_topology(2).residues())
    system = particle_system(2)
    cv = cvpack.ResidueCoordination(
        residues[:1],
        residues[1:],
        stepFunction="step(1-x)",
        thresholdDistance=0.5 * unit.nanometers,
    )
    cv.addToSystem(system)
    context = make_context(system, [[0, 0, 0], [1.9, 0, 0]])
    assert cv.getValue(context).value_in_unit(cv.getUnit()) == 0.0


def test_residue_coordination_explicit_pbc_still_wraps():
    residues = list(single_atom_topology(2).residues())
    system = particle_system(2)
    system.setDefaultPeriodicBoxVectors([2, 0, 0], [0, 2, 0], [0, 0, 2])
    cv = cvpack.ResidueCoordination(
        residues[:1],
        residues[1:],
        pbc=True,
        stepFunction="step(1-x)",
        thresholdDistance=0.5 * unit.nanometers,
    )
    cv.addToSystem(system)
    context = make_context(system, [[0, 0, 0], [1.9, 0, 0]])
    assert cv.getValue(context).value_in_unit(cv.getUnit()) == 1.0


def test_shortest_distance_defaults_to_no_pbc():
    system = particle_system(2)
    cv = cvpack.ShortestDistance([0], [1], 2)
    cv.addToSystem(system)
    context = make_context(system, [[0, 0, 0], [1.9, 0, 0]])
    # No pair within the 1 nm cutoff, so the CV evaluates to the cutoff distance
    assert cv.getValue(context).value_in_unit(cv.getUnit()) == pytest.approx(1.0)


def _shortest_distance_value(positions, group1, group2, box=None, **kwargs):
    system = particle_system(len(positions))
    if box is not None:
        system.setDefaultPeriodicBoxVectors(*(box * np.eye(3)))
    cv = cvpack.ShortestDistance(group1, group2, len(positions), **kwargs)
    cv.addToSystem(system)
    context = make_context(system, positions)
    return cv.getValue(context).value_in_unit(cv.getUnit())


# Small beta values make pairs beyond the nearest one, the cutoff and the terms
# outside the summations contribute measurably to the CV.
@pytest.mark.parametrize("beta", [3, 10, 100])
def test_shortest_distance_matches_formula(beta):
    positions = np.random.default_rng(2).uniform(0, 2, (12, 3))
    group1, group2 = list(range(6)), list(range(6, 12))
    value = _shortest_distance_value(positions, group1, group2, beta=beta)
    expected = shortest_distance_reference(positions, group1, group2, beta, 1.0)
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("cutoff", [0.8, 0.8 * unit.nanometers, 8 * unit.angstroms])
def test_shortest_distance_converts_cutoff_units(cutoff):
    positions = np.random.default_rng(3).uniform(0, 2, (12, 3))
    group1, group2 = list(range(6)), list(range(6, 12))
    value = _shortest_distance_value(
        positions, group1, group2, beta=3, cutoffDistance=cutoff
    )
    expected = shortest_distance_reference(positions, group1, group2, 3, 0.8)
    assert value == pytest.approx(expected)


def test_shortest_distance_counts_overlapping_pairs_once():
    positions = np.random.default_rng(4).uniform(0, 2, (6, 3))
    group1, group2 = [0, 1, 2, 3], [2, 3, 4, 5]
    value = _shortest_distance_value(
        positions, group1, group2, beta=3, cutoffDistance=2.0
    )
    expected = shortest_distance_reference(positions, group1, group2, 3, 2.0)
    assert value == pytest.approx(expected)


def test_shortest_distance_explicit_pbc_wraps():
    positions = [[0.05, 0, 0], [1.95, 0, 0], [1.0, 0.4, 0]]
    group1, group2 = [0], [1, 2]
    value = _shortest_distance_value(
        positions, group1, group2, box=2.0, beta=3, pbc=True
    )
    expected = shortest_distance_reference(positions, group1, group2, 3, 1.0, box=2.0)
    assert value == pytest.approx(expected)
    assert value < 0.5


def test_angle_is_not_periodic():
    assert cvpack.Angle(0, 1, 2).getPeriodicBounds() is None


@pytest.mark.parametrize(
    "force_class", [openmm.HarmonicAngleForce, openmm.PeriodicTorsionForce]
)
def test_energy_cvs_from_openmm_forces_are_not_periodic(force_class):
    force = force_class()
    if force_class is openmm.HarmonicAngleForce:
        force.addAngle(0, 1, 2, 1.9, 100.0)
    else:
        force.addTorsion(0, 1, 2, 3, 3, 0.0, 1.0)
    cv = cvpack.AtomicFunction.fromOpenMMForce(force, unit.kilojoules_per_mole)
    assert cv.getPeriodicBounds() is None


def _chirality(positions, n, ca, c, x):
    normal = np.cross(positions[n] - positions[ca], positions[c] - positions[ca])
    return np.sign(np.dot(normal, positions[x] - positions[ca]))


def test_pseudo_cb_has_the_same_chirality_as_cb():
    model = testsystems.LysozymeImplicit()
    positions = model.positions.value_in_unit(unit.nanometers)
    amino_acids = [
        residue
        for residue in model.topology.residues()
        if {"N", "CA"} <= {atom.name for atom in residue.atoms()}
    ]
    for residue in amino_acids:
        n, ca, cb, c, _ = BaseRMSDContent._getAtomList(residue)
        assert _chirality(positions, n, ca, c, cb) == 1, residue


def _glycine(hydrogen_names):
    topology = app.Topology()
    residue = topology.addResidue("GLY", topology.addChain())
    for name in ["N", "CA", "C", "O", *hydrogen_names]:
        element = app.element.hydrogen if name.startswith("H") else app.element.carbon
        topology.addAtom(name, element, residue)
    return residue, {atom.name: atom.index for atom in residue.atoms()}


def test_glycine_charmm_naming_uses_ha1():
    residue, index = _glycine(["HA1", "HA2"])
    assert BaseRMSDContent._getAtomList(residue)[2] == index["HA1"]


def test_glycine_pdb_naming_uses_ha3():
    residue, index = _glycine(["HA2", "HA3"])
    assert BaseRMSDContent._getAtomList(residue)[2] == index["HA3"]


def test_glycine_without_hydrogens_raises_value_error():
    residue, _ = _glycine([])
    with pytest.raises(ValueError, match="GLY"):
        BaseRMSDContent._getAtomList(residue)


def test_helix_torsion_content_reference_is_periodic():
    model = testsystems.LysozymeImplicit()
    residues = [r for r in model.topology.residues() if 59 <= r.index <= 79]
    cv1 = cvpack.HelixTorsionContent(residues, name="a")
    cv2 = cvpack.HelixTorsionContent(
        residues,
        phiReference=(360 - 63.8) * unit.degrees,
        psiReference=(360 - 41.1) * unit.degrees,
        name="b",
    )
    cv1.addToSystem(model.system)
    cv2.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    value1 = cv1.getValue(context).value_in_unit(cv1.getUnit())
    value2 = cv2.getValue(context).value_in_unit(cv2.getUnit())
    assert value2 == pytest.approx(value1)


@pytest.mark.parametrize("step_function", ["exp(-x^2)", "max(0, 1-x)"])
def test_rmsd_content_step_function_with_named_functions(step_function):
    model = testsystems.LysozymeImplicit()
    residues = [r for r in model.topology.residues() if 59 <= r.index <= 66]
    cv = cvpack.HelixRMSDContent(
        residues, model.system.getNumParticles(), stepFunction=step_function
    )
    cv.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    # 8 residues give 3 blocks (<= CHUNK_SIZE), so the inner CVs are the block RMSDs
    x = np.array(cv.getCollectiveVariableValues(context)) / 0.08
    if step_function.startswith("exp"):
        expected = np.sum(np.exp(-(x**2)))
    else:
        expected = np.sum(np.maximum(0, 1 - x))
    assert cv.getValue(context).value_in_unit(cv.getUnit()) == pytest.approx(expected)


def test_atomic_and_centroid_functions_default_to_no_pbc():
    system = particle_system(2)
    atomic = cvpack.AtomicFunction("distance(p1, p2)", unit.nanometer, [[0, 1]])
    centroid = cvpack.CentroidFunction(
        "distance(g1, g2)", unit.nanometer, [[0], [1]], name="centroid"
    )
    atomic.addToSystem(system)
    centroid.addToSystem(system)
    context = make_context(system, [[0, 0, 0], [1.9, 0, 0]])
    for cv in (atomic, centroid):
        assert cv.getValue(context).value_in_unit(cv.getUnit()) == pytest.approx(1.9)
