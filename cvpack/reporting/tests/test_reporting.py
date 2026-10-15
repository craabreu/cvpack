"""
Unit and regression test for the cvpack package.
"""

import os
import tempfile
from math import pi

import openmm
from openmm import app, unit
from openmmtools import testsystems

import cvpack
from cvpack import reporting


def test_cv_writer():
    """
    Test whether a reporter works as expected.
    """
    model = testsystems.AlanineDipeptideVacuum()
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    phi.addToSystem(model.system)
    psi = cvpack.Torsion(8, 14, 16, 18, name="psi")
    psi.addToSystem(model.system)
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        2 * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(1234)
    simulation = app.Simulation(model.topology, model.system, integrator)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 5678)
    with tempfile.TemporaryDirectory() as dirpath:
        with open(os.path.join(dirpath, "report.csv"), "w", encoding="utf-8") as file:
            reporter = reporting.StateDataReporter(
                file,
                1,
                writers=[
                    reporting.CVWriter(phi, value=True, emass=True),
                    reporting.CVWriter(psi, value=True, emass=True),
                ],
                step=True,
            )
            simulation.reporters.append(reporter)
            simulation.step(10)
        with open(os.path.join(dirpath, "report.csv"), encoding="utf-8") as file:
            assert file.readline() == ",".join(
                [
                    '#"Step"',
                    '"phi (rad)"',
                    '"emass[phi] (nm**2 Da/(rad**2))"',
                    '"psi (rad)"',
                    '"emass[psi] (nm**2 Da/(rad**2))"\n',
                ]
            )


def test_meta_cv_writer():
    """
    Test whether a reporter works as expected.
    """
    model = testsystems.AlanineDipeptideVacuum()
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    psi = cvpack.Torsion(8, 14, 16, 18, name="psi")
    umbrella = cvpack.MetaCollectiveVariable(
        f"0.5*kappa*(min(dphi,{2 * pi}-dphi)^2+min(dpsi,{2 * pi}-dpsi)^2)"
        "; dphi=abs(phi-phi0); dpsi=abs(psi-psi0)",
        [phi, psi],
        unit.kilojoules_per_mole,
        name="umbrella",
        kappa=100 * unit.kilojoules_per_mole / unit.radian**2,
        phi0=5 * pi / 6 * unit.radian,
        psi0=-5 * pi / 6 * unit.radian,
    )
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        2 * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(1234)
    umbrella.addToSystem(model.system)
    simulation = app.Simulation(model.topology, model.system, integrator)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 5678)

    with tempfile.TemporaryDirectory() as dirpath:
        with open(os.path.join(dirpath, "report.csv"), "w", encoding="utf-8") as file:
            reporter = reporting.StateDataReporter(
                file,
                1,
                writers=[
                    reporting.MetaCVWriter(
                        umbrella,
                        values=["phi", "psi"],
                        emasses=["phi", "psi"],
                        parameters=["phi0", "psi0"],
                        derivatives=["phi0", "psi0"],
                    ),
                ],
                step=True,
            )
            simulation.reporters.append(reporter)
            simulation.step(10)
        with open(os.path.join(dirpath, "report.csv"), encoding="utf-8") as file:
            assert file.readline() == ",".join(
                [
                    '#"Step"',
                    '"phi (rad)"',
                    '"psi (rad)"',
                    '"emass[phi] (nm**2 Da/(rad**2))"',
                    '"emass[psi] (nm**2 Da/(rad**2))"',
                    '"phi0 (rad)"',
                    '"psi0 (rad)"',
                    '"d[umbrella]/d[phi0] (kJ/(mol rad))"',
                    '"d[umbrella]/d[psi0] (kJ/(mol rad))"\n',
                ]
            )
