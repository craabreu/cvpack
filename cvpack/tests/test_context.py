"""
Tests for evaluating collective variables in OpenMM contexts and reporting them.
"""

import io

import numpy as np
import openmm
import pytest
from openmm import app, unit
from openmmtools import testsystems

import cvpack
from cvpack import reporting
from cvpack.tests.helpers import make_context, particle_system
from cvpack.utils import evaluate_in_context


@pytest.fixture
def dipeptide():
    return testsystems.AlanineDipeptideVacuum()


def test_allow_reinitialization_restores_force_group(dipeptide):
    shared = cvpack.Torsion(4, 6, 8, 14, name="shared")
    shared.addToSystem(dipeptide.system, setUnusedForceGroup=False)
    alone = cvpack.Torsion(4, 6, 8, 14, name="alone")
    alone.addToSystem(dipeptide.system)
    context = make_context(dipeptide.system, dipeptide.positions)
    with pytest.raises(ValueError, match="reinitialization"):
        shared.getValue(context)
    value = shared.getValue(context, allowReinitialization=True)
    assert value / unit.radian == pytest.approx(alone.getValue(context) / unit.radian)
    mass = shared.getEffectiveMass(context, allowReinitialization=True)
    expected = alone.getEffectiveMass(context)
    assert mass / mass.unit == pytest.approx(expected / expected.unit)
    assert shared.getForceGroup() == 0


def test_set_unused_force_group_returns_group(dipeptide):
    cv = cvpack.Torsion(4, 6, 8, 14)
    group = cv._setUnusedForceGroup(dipeptide.system)
    assert group == cv.getForceGroup() == 1


def test_effective_mass_ignores_massless_particles():
    system = particle_system(2)
    system.setParticleMass(0, 0.0)
    system.setParticleMass(1, 10.0)
    cv = cvpack.Distance(0, 1)
    cv.addToSystem(system)
    context = make_context(system, [[0, 0, 0], [0.5, 0, 0]])
    assert cv.getEffectiveMass(context) / unit.dalton == pytest.approx(10.0)


def test_inner_effective_masses_follow_current_positions(dipeptide):
    phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
    meta = cvpack.MetaCollectiveVariable("phi", [phi], unit.radian, name="meta")
    meta.addToSystem(dipeptide.system)
    direct = cvpack.Torsion(4, 6, 8, 14, name="direct")
    direct.addToSystem(dipeptide.system)
    context = make_context(dipeptide.system, dipeptide.positions)
    for shift in (0.0, 0.05):
        positions = dipeptide.positions.value_in_unit(unit.nanometers).copy()
        positions[6] += shift
        context.setPositions(positions)
        inner = meta.getInnerEffectiveMasses(context)["phi"]
        expected = direct.getEffectiveMass(context)
        assert inner / inner.unit == pytest.approx(expected / expected.unit)


def test_inner_effective_masses_with_virtual_sites_follow_the_context():
    system = particle_system(5)
    for atom, mass in enumerate([12.0, 1.0, 16.0, 1.0, 0.0]):
        system.setParticleMass(atom, mass)
    system.setVirtualSite(4, openmm.TwoParticleAverageSite(0, 2, 0.3, 0.7))
    inner = cvpack.AtomicFunction(
        "k*distance(p1, p2)", unit.nanometer, [[3, 4]], k=1.0, name="d"
    )
    meta = cvpack.MetaCollectiveVariable("2*d", [inner], unit.nanometer, name="meta")
    meta.addToSystem(system)
    direct = cvpack.AtomicFunction(
        "k*distance(p1, p2)", unit.nanometer, [[3, 4]], k=1.0, name="direct"
    )
    direct.addToSystem(system)
    positions = np.array(
        [[0, 0, 0], [0.1, 0, 0], [0.12, 0.1, 0], [0.3, 0.2, 0.1], [0, 0, 0]]
    )
    context = make_context(system, positions)
    masses = []
    for shift, k in [(0.0, 1.0), (0.05, 1.0), (0.05, 2.0)]:
        context.setParameter("k", k)
        moved = positions.copy()
        moved[3] += shift
        context.setPositions(moved)
        mass = meta.getInnerEffectiveMasses(context)["d"]
        expected = direct.getEffectiveMass(context)
        assert mass / mass.unit == pytest.approx(expected / expected.unit)
        masses.append(mass / mass.unit)
    # The context with virtual sites is built once and reused
    assert meta._virtual_site_context[1] is meta._virtual_site_context[1]
    assert masses[2] == pytest.approx(masses[1] / 4)


def test_evaluate_in_context_uses_context_parameters():
    system = particle_system(2)
    force = openmm.CustomBondForce("k*r")
    force.addGlobalParameter("k", 1.0)
    force.addBond(0, 1)
    system.addForce(force)
    context = make_context(system, [[0, 0, 0], [0.5, 0, 0]])
    context.setParameter("k", 2.0)
    assert evaluate_in_context(force, context) == pytest.approx(1.0)


def _simulation(model):
    integrator = openmm.VerletIntegrator(0.001)
    integrator.setIntegrationForceGroups({0})
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    return simulation


class DuckWriter:
    def getHeaders(self):
        return ["duck"]

    def getValues(self, simulation):
        return [1.0]


def test_reporter_accepts_duck_typed_writers_and_generators(dipeptide):
    simulation = _simulation(dipeptide)
    stream = io.StringIO()
    reporter = reporting.StateDataReporter(
        stream, 1, writers=(w for w in [DuckWriter()]), step=True
    )
    simulation.reporters.append(reporter)
    simulation.step(1)
    assert stream.getvalue().splitlines() == ['#"Step","duck"', "1,1.0"]


def test_cv_writer_reports_current_values_between_timing_columns(dipeptide):
    phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
    phi.addToSystem(dipeptide.system)
    simulation = _simulation(dipeptide)
    stream = io.StringIO()
    reporter = reporting.StateDataReporter(
        stream,
        1,
        writers=[reporting.CVWriter(phi, value=True)],
        step=True,
        speed=True,
        elapsedTime=True,
    )
    simulation.reporters.append(reporter)
    simulation.step(2)
    header, *rows = stream.getvalue().splitlines()
    assert header == '#"Step","phi (rad)","Speed (ns/day)","Elapsed Time (s)"'
    last = rows[-1].split(",")
    assert float(last[1]) == pytest.approx(
        phi.getValue(simulation.context) / unit.radian
    )


def test_cv_writer_allows_reinitialization(dipeptide):
    phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
    phi.addToSystem(dipeptide.system, setUnusedForceGroup=False)
    simulation = _simulation(dipeptide)
    writer = reporting.CVWriter(phi, value=True, emass=True, allowReinitialization=True)
    stream = io.StringIO()
    simulation.reporters.append(
        reporting.StateDataReporter(stream, 1, writers=[writer])
    )
    simulation.step(1)
    assert len(stream.getvalue().splitlines()) == 2


def test_meta_cv_writer_rejects_unknown_names():
    phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
    meta = cvpack.MetaCollectiveVariable("phi", [phi], unit.radian)
    with pytest.raises(ValueError, match="chi"):
        reporting.MetaCVWriter(meta, values=["chi"])


def test_meta_cv_writer_allows_reinitialization(dipeptide):
    phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
    meta = cvpack.MetaCollectiveVariable(
        "k*phi", [phi], unit.kilojoule_per_mole, k=1.0 * unit.kilojoule_per_mole
    )
    meta.addToSystem(dipeptide.system, setUnusedForceGroup=False)
    simulation = _simulation(dipeptide)
    writer = reporting.MetaCVWriter(
        meta, values=["phi"], derivatives=["k"], allowReinitialization=True
    )
    stream = io.StringIO()
    simulation.reporters.append(
        reporting.StateDataReporter(stream, 1, writers=[writer])
    )
    simulation.step(1)
    assert len(stream.getvalue().splitlines()) == 2


def test_custom_writer_is_exported():
    assert isinstance(DuckWriter(), reporting.CustomWriter)


def test_inner_effective_masses_with_virtual_sites():
    system = particle_system(5)
    for atom, mass in enumerate([12.0, 1.0, 16.0, 1.0, 0.0]):
        system.setParticleMass(atom, mass)
    system.setVirtualSite(4, openmm.TwoParticleAverageSite(0, 2, 0.3, 0.7))
    meta = cvpack.MetaCollectiveVariable(
        "2*d", [cvpack.Distance(3, 4, name="d")], unit.nanometer, name="meta"
    )
    meta.addToSystem(system)
    direct = cvpack.Distance(3, 4, name="direct")
    direct.addToSystem(system)
    positions = [[0, 0, 0], [0.1, 0, 0], [0.12, 0.1, 0], [0.3, 0.2, 0.1], [0, 0, 0]]
    context = make_context(system, positions)
    inner = meta.getInnerEffectiveMasses(context)["d"]
    expected = direct.getEffectiveMass(context)
    assert inner / inner.unit == pytest.approx(expected / expected.unit)
