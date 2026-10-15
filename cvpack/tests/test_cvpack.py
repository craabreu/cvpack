"""
Unit and regression test for the cvpack package.
"""

import copy
import inspect
import io
import itertools as it
import sys
import typing as t

import mdtraj
import networkx as nx
import numpy as np
import openmm
import pytest
from openmm import app, unit
from openmmtools import testsystems
from openmmtools.constants import ONE_4PI_EPS0
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

import cvpack
from cvpack import serialization
from cvpack.tests.helpers import make_context, shortest_distance_reference
from cvpack.units import value_in_md_units


def test_cvpack_imported():
    """
    Sample test, will always pass so long as import statement worked.

    """
    assert "cvpack" in sys.modules


def test_effective_mass():
    """
    Test effective mass evaluation at a given context.

    """
    model = testsystems.AlanineDipeptideVacuum()
    rg_cv = cvpack.RadiusOfGyration(range(model.system.getNumParticles()))
    rg_cv.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    effective_mass = rg_cv.getEffectiveMass(context)
    assert effective_mass / effective_mass.unit == pytest.approx(30.946932)


def perform_common_tests(
    collectiveVariable: cvpack.CollectiveVariable,
    context: openmm.Context,
) -> None:
    """
    Function to be called in every individual cv test.

    """
    # Unit must conform to the default OpenMM system
    unity = 1 * collectiveVariable.getUnit()
    assert unity.value_in_unit_system(unit.md_unit_system) == 1

    # Class must have full type annotation (except for argument `self`)
    signature = inspect.signature(collectiveVariable.__init__).parameters.items()
    args = {name: annotation.annotation for name, annotation in signature}
    for _, annotation in args.items():
        assert annotation is not inspect.Parameter.empty

    # YAML tag must be defined and start with "!cvpack."
    assert hasattr(collectiveVariable, "yaml_tag")
    assert collectiveVariable.yaml_tag.startswith("!cvpack.")

    # Test serialization/deserialization
    pipe = io.StringIO()
    serialization.serialize(collectiveVariable, pipe)
    pipe.seek(0)
    new_cv = serialization.deserialize(pipe)
    new_cv.addToSystem(context.getSystem())
    context.reinitialize(preserveState=True)
    value1 = collectiveVariable.getValue(context)
    value2 = new_cv.getValue(context)
    assert value1 / value1.unit == value2 / value2.unit
    mass1 = collectiveVariable.getEffectiveMass(context)
    mass2 = new_cv.getEffectiveMass(context)
    assert mass1 / mass1.unit == mass2 / mass2.unit
    assert collectiveVariable.getName() == new_cv.getName()
    assert collectiveVariable.getPeriodicBounds() == new_cv.getPeriodicBounds()


def test_cv_is_in_context():
    """
    Test whether a collective variable is in a context.

    """
    model = testsystems.AlanineDipeptideVacuum()
    rg_cv = cvpack.RadiusOfGyration(range(model.system.getNumParticles()))
    context = make_context(model.system, model.positions)
    with pytest.raises(RuntimeError) as excinfo:
        rg_cv.getValue(context)
    assert str(excinfo.value) == "This force is not present in the given context."


def test_distance():
    """
    Test whether a distance is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    atom1, atom2 = 0, 5
    distance = cvpack.Distance(atom1, atom2)
    distance.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    value1 = distance.getValue(context).value_in_unit(distance.getUnit())
    value2 = np.sqrt(np.sum((model.positions[atom1] - model.positions[atom2]) ** 2))
    assert value1 == pytest.approx(value2)
    perform_common_tests(distance, context)


def test_angle():
    """
    Test whether an angle is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    atoms = [0, 5, 10]
    angle = cvpack.Angle(*atoms)
    angle.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    value1 = angle.getValue(context).value_in_unit(angle.getUnit())
    r21 = model.positions[atoms[0]] - model.positions[atoms[1]]
    r23 = model.positions[atoms[2]] - model.positions[atoms[1]]
    value2 = np.arccos(np.dot(r21, r23) / (np.linalg.norm(r21) * np.linalg.norm(r23)))
    assert value1 == pytest.approx(value2)
    perform_common_tests(angle, context)


def test_torsion():
    """
    Test whether a torsion angle is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    atoms = [0, 5, 10, 15]
    torsion = cvpack.Torsion(*atoms)
    torsion.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    value1 = torsion.getValue(context).value_in_unit(torsion.getUnit())
    r21 = model.positions[atoms[0]] - model.positions[atoms[1]]
    u23 = model.positions[atoms[2]] - model.positions[atoms[1]]
    u23 /= np.linalg.norm(u23)
    r34 = model.positions[atoms[3]] - model.positions[atoms[2]]
    value2 = np.arctan2(
        np.cross(r21, r34).dot(u23), r21.dot(r34) - r21.dot(u23) * r34.dot(u23)
    )
    assert value1 == pytest.approx(value2)
    perform_common_tests(torsion, context)


def test_radius_of_gyration():
    """
    Test whether a radius of gyration is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    masses = np.array(
        [value_in_md_units(atom.element.mass) for atom in model.topology.atoms()]
    )
    positions = model.positions.value_in_unit(unit.nanometers)
    centroid = positions.mean(axis=0)
    center_of_mass = np.einsum("i,ij->j", masses, positions) / np.sum(masses)
    num_atoms = model.system.getNumParticles()
    rgsq = np.sum((positions - centroid) ** 2) / num_atoms
    rg_cv = cvpack.RadiusOfGyration(range(num_atoms))
    weighted_rgsq = np.sum((positions - center_of_mass) ** 2) / num_atoms
    weighted_rg_cv = cvpack.RadiusOfGyration(range(num_atoms), weighByMass=True)
    rg_cv.addToSystem(model.system)
    weighted_rg_cv.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    rgval = rg_cv.getValue(context).value_in_unit(unit.nanometers)
    assert rgval**2 == pytest.approx(rgsq)
    weighted_rgval = weighted_rg_cv.getValue(context).value_in_unit(unit.nanometers)
    assert weighted_rgval**2 == pytest.approx(weighted_rgsq)
    perform_common_tests(rg_cv, context)


def test_radius_of_gyration_squared():
    """
    Test whether a squared radius of gyration is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    masses = np.array(
        [value_in_md_units(atom.element.mass) for atom in model.topology.atoms()]
    )
    positions = model.positions.value_in_unit(unit.nanometers)
    centroid = positions.mean(axis=0)
    center_of_mass = np.einsum("i,ij->j", masses, positions) / np.sum(masses)
    num_atoms = model.system.getNumParticles()
    rgsq = np.sum((positions - centroid) ** 2) / num_atoms
    rg_sq = cvpack.RadiusOfGyrationSq(range(num_atoms))
    weighted_rgsq = np.sum((positions - center_of_mass) ** 2) / num_atoms
    weighted_rg_sq = cvpack.RadiusOfGyrationSq(range(num_atoms), weighByMass=True)
    rg_sq.addToSystem(model.system)
    weighted_rg_sq.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    rg_sq_value = value_in_md_units(rg_sq.getValue(context))
    assert rg_sq_value == pytest.approx(rgsq)
    weighted_rg_sq_value = value_in_md_units(weighted_rg_sq.getValue(context))
    assert weighted_rg_sq_value == pytest.approx(weighted_rgsq)
    perform_common_tests(rg_sq, context)


@pytest.mark.parametrize(
    "cvClass", [cvpack.RadiusOfGyration, cvpack.RadiusOfGyrationSq]
)
def test_radius_of_gyration_with_massless_atoms(cvClass):
    """
    Test whether a radius of gyration can be computed for a group containing massless
    particles (e.g., virtual sites or rigid-body members).

    """
    model = testsystems.AlanineDipeptideVacuum()
    num_atoms = model.system.getNumParticles()
    while model.system.getNumConstraints() > 0:
        model.system.removeConstraint(0)
    for atom in range(0, num_atoms, 2):
        model.system.setParticleMass(atom, 0)
    positions = model.positions.value_in_unit(unit.nanometers)
    rgsq = np.sum((positions - positions.mean(axis=0)) ** 2) / num_atoms
    cv = cvClass(range(num_atoms))
    cv.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    value = value_in_md_units(cv.getValue(context))
    if cvClass is cvpack.RadiusOfGyration:
        value = value**2
    assert value == pytest.approx(rgsq)


def test_number_of_contacts():
    """
    Test whether a number of contacts is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    pos = model.positions
    group1 = [
        a.index for a in model.topology.atoms() if a.element == app.element.carbon
    ]
    group2 = [
        a.index for a in model.topology.atoms() if a.element == app.element.oxygen
    ]
    forces = {f.getName(): f for f in model.system.getForces()}
    nonbonded = forces["NonbondedForce"]
    exclusions = set()
    for index in range(nonbonded.getNumExceptions()):
        i, j, *_ = nonbonded.getExceptionParameters(index)
        exclusions.add((i, j) if i < j else (j, i))
    pairs = set()
    for i, j in it.product(group1, group2):
        if j != i:
            pair = (i, j) if i < j else (j, i)
            if pair not in exclusions:
                pairs.add(pair)
    distances = np.array([np.linalg.norm(pos[i] - pos[j]) for i, j in pairs])
    contacts = np.where(distances <= 0.6, 1 / (1 + (distances / 0.3) ** 6), 0)
    number_of_contacts = cvpack.NumberOfContacts(
        group1,
        group2,
        forces["NonbondedForce"],
        switchFactor=None,
    )
    number_of_contacts.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    nc_value = number_of_contacts.getValue(context)
    assert nc_value / nc_value.unit == pytest.approx(contacts.sum())
    perform_common_tests(number_of_contacts, context)


def run_rmsd_test(
    coordinates: np.ndarray,
    group: t.Sequence[int],
    passGroupOnly: bool,
    passVec3: bool,
) -> None:
    """
    Performs a specific RMSD test

    """
    model = testsystems.AlanineDipeptideVacuum()
    num_atoms = model.topology.getNumAtoms()
    reference = np.array(model.positions.value_in_unit(unit.nanometers))
    group_ref = reference[group, :] - reference[group, :].mean(axis=0)
    if passVec3:
        reference = [openmm.Vec3(*row) for row in reference]
    rmsd = cvpack.RMSD(
        dict(zip(group, group_ref)) if passGroupOnly else reference,
        group,
        num_atoms,
    )
    rmsd.addToSystem(model.system)
    context = make_context(model.system, coordinates)
    group_coords = coordinates[group, :] - coordinates[group, :].mean(axis=0)
    _, rssd = Rotation.align_vectors(group_coords, group_ref)
    rmsd_value = rmsd.getValue(context)
    assert rmsd_value / rmsd_value.unit == pytest.approx(rssd / np.sqrt(len(group)))


def test_rmsd():
    """
    Test whether an RMSD is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    num_atoms = model.topology.getNumAtoms()
    rmsd = cvpack.RMSD(model.positions, np.arange(num_atoms), num_atoms)
    rmsd.addToSystem(model.system)
    integrator = openmm.VerletIntegrator(2 * unit.femtosecond)
    integrator.setIntegrationForceGroups({0})
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    context.getIntegrator().step(10000)
    state = context.getState(getPositions=True)
    coordinates = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    rng = np.random.default_rng(17)
    for pass_vec3 in [False, True]:
        for pass_group_only in [False, True]:
            group = np.arange(num_atoms)
            run_rmsd_test(coordinates, group, pass_group_only, pass_vec3)
            run_rmsd_test(
                coordinates, group[: num_atoms // 2], pass_group_only, pass_vec3
            )
            rng.shuffle(group)
            run_rmsd_test(
                coordinates, group[: num_atoms // 2], pass_group_only, pass_vec3
            )
    perform_common_tests(rmsd, context)


def test_composite_rmsd():
    """
    Test whether a CompositeRMSD is computed correctly.

    """
    model = testsystems.HostGuestExplicit()
    host_atoms, guest_atoms = (
        [a.index for a in r.atoms()] for r in it.islice(model.topology.residues(), 0, 2)
    )
    try:
        crmsd1 = cvpack.CompositeRMSD(model.positions, [host_atoms, guest_atoms])
        atoms = host_atoms + guest_atoms
        crmsd2 = cvpack.CompositeRMSD(
            dict(zip(atoms, model.positions[atoms])),
            [host_atoms, guest_atoms],
            numAtoms=model.system.getNumParticles(),
        )
    except ImportError:
        pytest.skip("openmm-cpp-forces is not installed")
    crmsd1.addToSystem(model.system)
    crmsd2.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    perform_common_tests(crmsd1, context)
    perform_common_tests(crmsd2, context)


def test_helix_torsion_content():
    """
    Test whether a helix Ramachandran content is computed correctly.

    """
    model = testsystems.LysozymeImplicit()

    positions = model.positions.value_in_unit(unit.nanometers)
    traj = mdtraj.Trajectory(positions, mdtraj.Topology.from_openmm(model.topology))
    _, phi = mdtraj.compute_phi(traj)
    _, psi = mdtraj.compute_psi(traj)

    def wrap(angle):
        return (angle + 180) % 360 - 180

    x = wrap(np.rad2deg(phi.ravel()[:-1]) + 63.8) / 25
    y = wrap(np.rad2deg(psi.ravel()[1:]) + 41.1) / 25
    computed_value = np.sum(1 / (1 + x**6) + 1 / (1 + y**6)) / 2

    residues = list(model.topology.residues())
    with pytest.raises(ValueError) as excinfo:
        helix_content = cvpack.HelixTorsionContent(residues)
    assert str(excinfo.value) == "Could not find atom N in residue TMP163"
    helix_content = cvpack.HelixTorsionContent(residues[0:-1])
    helix_content.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    cv_value = helix_content.getValue(context)

    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(helix_content, context)


def test_helix_angle_content():
    """
    Test whether a helix angle content is computed correctly.

    """
    model = testsystems.LysozymeImplicit()

    positions = model.positions.value_in_unit(unit.nanometers)
    traj = mdtraj.Trajectory(positions, mdtraj.Topology.from_openmm(model.topology))
    alpha_carbons = traj.top.select("name CA")
    angle_atoms = np.array(
        [alpha_carbons[:-2], alpha_carbons[1:-1], alpha_carbons[2:]]
    ).T
    angles = mdtraj.compute_angles(traj, angle_atoms)
    x = (np.rad2deg(angles.ravel()) - 88) / 15
    computed_value = np.sum(1 / (1 + x**6))

    residues = list(model.topology.residues())
    with pytest.raises(ValueError) as excinfo:
        helix_content = cvpack.HelixAngleContent(residues)
    assert str(excinfo.value) == "Could not find atom CA in residue TMP163"
    helix_content = cvpack.HelixAngleContent(residues[0:-1])
    helix_content.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    cv_value = helix_content.getValue(context)
    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(helix_content, context)


def test_helix_hbond_content():
    """
    Test whether a helix hydrogen-bond content is computed correctly.

    """
    model = testsystems.LysozymeImplicit()

    positions = model.positions.value_in_unit(unit.nanometers)
    traj = mdtraj.Trajectory(positions, mdtraj.Topology.from_openmm(model.topology))
    hydrogens = traj.top.select("resSeq 59 to 79 and name H")
    oxygens = traj.top.select("resSeq 59 to 79 and name O")
    distances = mdtraj.compute_distances(
        traj, np.array([hydrogens[4:], oxygens[:-4]]).T
    )
    x = distances.ravel() / 0.33
    computed_value = np.sum(1 / (1 + x**6))

    residues = list(model.topology.residues())
    with pytest.raises(ValueError):
        helix_content = cvpack.HelixHBondContent(residues[58:70] + residues[71:79])
    helix_content = cvpack.HelixHBondContent(residues[58:79])
    helix_content.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    cv_value = helix_content.getValue(context)
    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(helix_content, context)


@pytest.mark.parametrize("normalize", [False, True])
def test_helix_rmsd_content(normalize: bool):
    """
    Test whether a helix rmsd content is computed correctly.

    """
    model = testsystems.LysozymeImplicit()

    num_atoms = model.topology.getNumAtoms()
    residues = list(model.topology.residues())
    with pytest.raises(ValueError) as excinfo:
        helix_content = cvpack.HelixRMSDContent(
            residues[len(residues) - 37 :], num_atoms, normalize=normalize
        )
    assert str(excinfo.value) == "Atom N not found in residue TMP163"

    start, end = 39, 51
    helix_content = cvpack.HelixRMSDContent(
        residues[start:end], num_atoms, normalize=normalize
    )
    helix_content.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    cv_value = helix_content.getValue(context)

    traj = mdtraj.Trajectory(
        model.positions, mdtraj.Topology.from_openmm(model.topology)
    )
    ref = copy.deepcopy(traj)
    positions = cvpack.helix_rmsd_content.ALPHA_POSITIONS

    def compute_cv_value(atoms):
        computed_value = 0
        for i in range(len(atoms) // 5 - 5):
            group = atoms[5 * i : 5 * i + 30]
            ref.xyz[:, group, :] = positions
            x4 = (mdtraj.rmsd(traj, ref, 0, group).item() / 0.08) ** 4
            computed_value += (1 + x4) / (1 + x4 + x4**2)
        if normalize:
            computed_value /= len(atoms) // 5 - 5
        return computed_value

    def select_atoms(stard, end):
        atoms = ()
        for residue in residues[stard:end]:
            residue_atoms = {atom.name: atom.index for atom in residue.atoms()}
            if residue.name == "GLY":
                residue_atoms["CB"] = residue_atoms["HA3"]
            atoms += tuple(residue_atoms[name] for name in ["N", "CA", "CB", "C", "O"])
        return atoms

    computed_value = compute_cv_value(select_atoms(start, end))
    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(helix_content, context)


def test_helix_torsion_similarity():
    """
    Test whether a torsion similarity CV is computed correctly.

    """
    model = testsystems.LysozymeImplicit()
    positions = model.positions.value_in_unit(unit.nanometers)
    traj = mdtraj.Trajectory(positions, mdtraj.Topology.from_openmm(model.topology))
    phi_atoms, phi = mdtraj.compute_phi(traj)
    psi_atoms, psi = mdtraj.compute_psi(traj)
    torsion_similarity = cvpack.TorsionSimilarity(
        np.vstack([phi_atoms[1:], psi_atoms[1:]]),
        np.vstack([phi_atoms[:-1], psi_atoms[:-1]]),
    )
    torsion_similarity.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    cv_value = torsion_similarity.getValue(context)
    phi = phi.ravel()
    psi = psi.ravel()
    deltas = np.hstack([phi[1:], psi[1:]]) - np.hstack([phi[:-1], psi[:-1]])
    assert cv_value / cv_value.unit == pytest.approx(np.sum(0.5 * (1 + np.cos(deltas))))
    perform_common_tests(torsion_similarity, context)


@pytest.mark.parametrize("normalize", [False, True])
def test_sheet_rmsd_content(normalize: bool):
    """
    Test whether a sheet rmsd content is computed correctly.

    """
    model = testsystems.SrcImplicit()
    positions = np.vstack([np.array(pos) for pos in value_in_md_units(model.positions)])
    topology = mdtraj.Topology.from_openmm(model.topology)
    residues = list(it.islice(model.topology.residues(), 68, 82))

    sheet_content = cvpack.SheetRMSDContent(
        residues, topology.n_atoms, normalize=normalize
    )
    sheet_content.setForceGroup(31)
    sheet_content.addToSystem(model.system)
    context = make_context(model.system, positions)
    cv_value = sheet_content.getValue(context)

    traj = mdtraj.Trajectory(positions, topology)
    ref = copy.deepcopy(traj)

    def compute_step_function(indices):
        group = sum((residue_atoms[i] for i in indices), ())
        ref.xyz[:, group, :] = cvpack.sheet_rmsd_content.ANTIBETA_POSITIONS
        x4 = (mdtraj.rmsd(traj, ref, 0, group).item() / 0.08) ** 4
        return (1 + x4) / (1 + x4 + x4**2)

    residue_atoms = []
    for residue in residues:
        atoms = {atom.name: atom.index for atom in residue.atoms()}
        if residue.name == "GLY":
            atoms["CB"] = atoms["HA2"]
        residue_atoms.append(tuple(atoms[name] for name in ["N", "CA", "CB", "C", "O"]))

    min_separation = 5
    terms = [
        compute_step_function([i, i + 1, i + 2, j, j + 1, j + 2])
        for i, j in it.combinations(range(len(residues) - 2), 2)
        if j - i >= min_separation
    ]
    computed_value = sum(terms)
    if normalize:
        computed_value /= len(terms)

    assert cv_value / cv_value.unit == pytest.approx(computed_value, abs=1e-5)
    perform_common_tests(sheet_content, context)


def test_atomic_function():
    """
    Test whether an atomic-function CV is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    num_atoms = model.system.getNumParticles()
    atoms = np.arange(num_atoms)
    np.random.default_rng(11).shuffle(atoms)
    function = "+".join(f"distance(p{i + 1}, p{i + 2})" for i in range(num_atoms - 1))
    with pytest.raises(ValueError) as excinfo:
        colvar = cvpack.AtomicFunction(function, unit.angstrom, atoms)
    assert (
        str(excinfo.value) == "Unit angstrom is not compatible with the MD unit system."
    )
    colvar = cvpack.AtomicFunction(function, unit.nanometers, atoms)
    colvar.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    cv_value = colvar.getValue(context)
    positions = model.positions.value_in_unit(unit.nanometers)
    computed_value = np.sum(
        [
            np.linalg.norm(positions[atoms[i + 1]] - positions[atoms[i]])
            for i in range(num_atoms - 1)
        ]
    )
    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(colvar, context)


def test_centroid_function():
    """
    Test whether a centroid-function CV is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    num_atoms = model.system.getNumParticles()
    atoms = np.arange(num_atoms)
    np.random.default_rng(11).shuffle(atoms)
    num_groups = num_atoms // 3
    groups = np.reshape(atoms[: 3 * num_groups], (num_groups, 3))
    function = "+".join(f"distance(g{i + 1}, g{i + 2})" for i in range(num_groups - 1))
    with pytest.raises(ValueError) as excinfo:
        colvar = cvpack.CentroidFunction(function, unit.angstrom, groups)
    assert (
        str(excinfo.value) == "Unit angstrom is not compatible with the MD unit system."
    )
    colvar = cvpack.CentroidFunction(
        function, unit.nanometers, groups, weighByMass=False
    )
    colvar.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    cv_value = colvar.getValue(context)
    positions = model.positions.value_in_unit(unit.nanometers)
    computed_value = np.sum(
        [
            np.linalg.norm(
                np.mean(positions[groups[i + 1]], axis=0)
                - np.mean(positions[groups[i]], axis=0)
            )
            for i in range(num_groups - 1)
        ]
    )
    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(colvar, context)


def test_attraction_strength():
    """
    Test whether an attraction strength CV is computed correctly.

    """
    num_particles = 20
    group1 = list(range(num_particles // 2))
    group2 = list(range(num_particles // 2, num_particles))
    rng = np.random.default_rng(13)
    charges = 1.0 - 2.0 * rng.random(num_particles)
    sigmas = rng.random(num_particles)
    epsilons = rng.random(num_particles)
    positions = rng.random((num_particles, 3))
    cutoff = 1.0
    strength = 0.0
    for i in group1:
        for j in group2:
            rij = np.linalg.norm(positions[i] - positions[j])
            if rij <= cutoff:
                x = np.abs((2 * rij / (sigmas[i] + sigmas[j])) ** 6 - 2.0) + 2.0
                strength -= 4 * np.sqrt(epsilons[i] * epsilons[j]) * (1 / x**2 - 1 / x)
                if charges[i] * charges[j] < 0.0:
                    x = rij / cutoff
                    strength -= (
                        ONE_4PI_EPS0
                        * charges[i]
                        * charges[j]
                        * (1 / x + (x**2 - 3) / 2)
                    )
    system = openmm.System()
    for _ in range(num_particles):
        system.addParticle(1.0)
    force = openmm.NonbondedForce()
    for params in zip(charges, sigmas, epsilons):
        force.addParticle(*params)
    force.setUseSwitchingFunction(False)
    force.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    force.setCutoffDistance(cutoff)
    colvar = cvpack.AttractionStrength(group1, group2, force)
    system.addForce(colvar)
    context = make_context(system, positions)
    assert value_in_md_units(colvar.getValue(context)) == pytest.approx(strength)
    perform_common_tests(colvar, context)


@pytest.mark.parametrize("includeHs", [False, True])
def test_residue_coordination(includeHs: bool):
    """
    Test whether a residue coordination CV is computed correctly.

    """
    model = testsystems.LysozymeImplicit()
    positions = model.positions.value_in_unit(unit.nanometers)
    groups = [
        list(it.islice(model.topology.residues(), start, end))
        for start, end in [(125, 142), (142, 156)]
    ]
    centroids = []
    for group in groups:
        residue_centroids = []
        for r in group:
            atoms = [a.index for a in r.atoms() if includeHs or a.element.symbol != "H"]
            residue_centroids.append(np.mean(positions[atoms], axis=0))
        centroids.append(np.vstack(residue_centroids))
    distances = np.linalg.norm(centroids[0][:, None, :] - centroids[1], axis=2)
    computed_cv = np.sum(1 / (1 + distances**6))

    res_coord = cvpack.ResidueCoordination(
        *groups, pbc=False, weighByMass=False, includeHydrogens=includeHs
    )
    res_coord.addToSystem(model.system)
    context = make_context(model.system, positions)
    cv_value = res_coord.getValue(context)

    assert cv_value / cv_value.unit == pytest.approx(computed_cv)
    perform_common_tests(res_coord, context)


@pytest.mark.parametrize("metric", [cvpack.path.progress, cvpack.path.deviation])
def test_path_in_cv_space(metric: cvpack.path.Metric):
    """
    Test whether a path in CV space is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    phi_atoms = ["ACE-C", "ALA-N", "ALA-CA", "ALA-C"]
    psi_atoms = ["ALA-N", "ALA-CA", "ALA-C", "NME-N"]
    atoms = [f"{a.residue.name}-{a.name}" for a in model.topology.atoms()]
    milestones = np.array(
        [[1.3, -0.2], [1.2, 3.1], [-2.7, 2.9], [-1.3, 2.7], [-1.3, -0.4]]
    )
    torsions = [
        [atoms.index(atom) for atom in group] for group in (phi_atoms, psi_atoms)
    ]
    phi, psi = (cvpack.Torsion(*indices) for indices in torsions)
    sigma = np.pi / 6
    var = cvpack.PathInCVSpace(metric, [phi, psi], milestones, sigma)
    var.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    cv_value = var.getValue(context)

    traj = mdtraj.Trajectory(
        model.positions.value_in_unit(unit.nanometers), model.mdtraj_topology
    )
    angles = mdtraj.compute_dihedrals(traj, torsions).ravel()
    deltas = np.abs(milestones - angles)
    x = -0.5 * np.sum(np.minimum(deltas, 2 * np.pi - deltas) ** 2, axis=1) / sigma**2
    if metric is cvpack.path.progress:
        n = len(x)
        computed_value = np.exp(logsumexp(x, b=np.arange(n)) - logsumexp(x)) / (n - 1)
    else:
        computed_value = -2 * sigma**2 * logsumexp(x)
    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(var, context)


@pytest.mark.parametrize("metric", [cvpack.path.progress, cvpack.path.deviation])
def test_path_in_rmsd_space(metric: cvpack.path.Metric):
    """
    Test whether a path in RMSD space is computed correctly.

    """

    def get_movable_atoms(model, atom1, atom2):
        graph = model.mdtraj_topology.to_bondgraph()
        nodes = list(graph.nodes)
        graph.remove_edge(nodes[atom1], nodes[atom2])
        movable = list(nx.connected_components(graph))[1]
        return [nodes.index(atom) for atom in movable]

    model = testsystems.AlanineDipeptideVacuum()
    atoms = get_movable_atoms(model, 8, 14)
    x = model.positions / model.positions.unit
    vector = (x[14, :] - x[8, :]) / np.linalg.norm(x[14, :] - x[8, :])
    rotation = Rotation.from_rotvec((np.pi / 6) * vector)
    frames = [x.copy()]
    for _ in range(6):
        x[atoms, :] = x[8, :] + rotation.apply(x[atoms, :] - x[8, :])
        frames.append(x.copy())
    milestones = [dict(enumerate(frame)) for frame in frames]
    sigma = 0.05
    path_cv = cvpack.PathInRMSDSpace(metric, milestones, len(x), sigma)
    path_cv.addToSystem(model.system)
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.004 * unit.picoseconds
    )
    integrator.setRandomNumberSeed(19)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)

    traj = mdtraj.Trajectory(frames, model.mdtraj_topology)
    for _ in range(10):
        integrator.step(1000)
        frame = mdtraj.Trajectory(
            context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.nanometer),
            model.mdtraj_topology,
        )
        exponents = -0.5 * (mdtraj.rmsd(traj, frame) / sigma) ** 2
        s = np.exp(logsumexp(exponents, b=np.arange(7)) - logsumexp(exponents)) / 6
        z = -2 * sigma**2 * logsumexp(exponents)
        value = path_cv.getValue(context)
        if metric is cvpack.path.progress:
            assert value / unit.dimensionless == pytest.approx(s, rel=1e-3)
        else:
            assert value / unit.nanometer**2 == pytest.approx(z, rel=1e-3)
    perform_common_tests(path_cv, context)


def test_openmm_force_wrapper():
    """
    Test whether an OpenMM force wrapper is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    angle = openmm.CustomAngleForce("theta")
    angle.addAngle(0, 1, 2)
    cv = cvpack.OpenMMForceWrapper(
        angle, unit.radian, periodicBounds=[-np.pi, np.pi] * unit.radian
    )
    cv.addToSystem(model.system)
    angle.setForceGroup(cv.getForceGroup() + 1)
    model.system.addForce(angle)
    context = make_context(model.system, model.positions)
    state = context.getState(
        getEnergy=True,
        groups={cv.getForceGroup() + 1},
    )
    angle_value = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert cv.getValue(context) / unit.radians == pytest.approx(angle_value)
    perform_common_tests(cv, context)


def test_openmm_force_wrapper_delegates_to_wrapped_force():
    """
    Test whether unknown attributes are looked up in the wrapped force.

    """
    force = openmm.CustomNonbondedForce("r")
    force.addParticle([])
    force.addParticle([])
    force.setNonbondedMethod(force.CutoffNonPeriodic)
    cv = cvpack.OpenMMForceWrapper(force, unit.nanometer)
    assert cv.getNumParticles() == 2
    assert cv.getNonbondedMethod() == cv.CutoffNonPeriodic
    assert cv.CutoffNonPeriodic == openmm.CustomNonbondedForce.CutoffNonPeriodic
    with pytest.raises(AttributeError):
        cv.notAnAttribute  # noqa: B018


def test_shortest_distance():
    """
    Test whether a shortest distance CV is computed correctly.

    """
    model = testsystems.HostGuestVacuum()
    group1, group2 = [], []
    for residue in model.topology.residues():
        group = group1 if residue.name == "B2" else group2
        group.extend(atom.index for atom in residue.atoms())
    positions = model.positions.value_in_unit(unit.nanometers)
    num_atoms = model.system.getNumParticles()
    beta = 100
    rc = 1
    min_dist = cvpack.ShortestDistance(group1, group2, num_atoms, beta, rc)
    min_dist.addToSystem(model.system)
    context = make_context(model.system, positions)
    rng = np.random.default_rng(5)
    for _ in range(5):
        coords = positions + rng.normal(0, 0.01, positions.shape)
        context.setPositions(coords)
        cv_value = min_dist.getValue(context) / unit.nanometer
        computed_value = shortest_distance_reference(coords, group1, group2, beta, rc)
        assert cv_value == pytest.approx(computed_value)
    perform_common_tests(min_dist, context)


def test_meta_collective_variable():
    """
    Test a meta-collective variable's value, parameter derivative and serialization.

    """
    model = testsystems.AlanineDipeptideVacuum()
    phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
    psi = cvpack.Torsion(6, 8, 14, 16, name="psi")
    meta = cvpack.MetaCollectiveVariable(
        "0.5*kappa*(phi^2 + psi^2)",
        [phi, psi],
        unit.kilojoules_per_mole,
        kappa=10 * unit.kilojoules_per_mole / unit.radian**2,
    )
    meta.addToSystem(model.system)
    context = make_context(model.system, model.positions)
    inner = meta.getInnerValues(context)
    squares = sum((value / unit.radian) ** 2 for value in inner.values())
    assert meta.getValue(context) / unit.kilojoules_per_mole == pytest.approx(
        5 * squares
    )
    derivative = meta.getParameterDerivatives(context)["kappa"]
    assert derivative / derivative.unit == pytest.approx(0.5 * squares)
    perform_common_tests(meta, context)
