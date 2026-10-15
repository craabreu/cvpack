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
from cvpack.tests.helpers import make_context, particle_system, single_atom_topology


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
        r for r in model.topology.residues() if {"N", "CA"} <= {a.name for a in r.atoms()}
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


def test_glycine_charmm_naming_uses_ha2():
    residue, index = _glycine(["HA1", "HA2"])
    assert BaseRMSDContent._getAtomList(residue)[2] == index["HA2"]


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
