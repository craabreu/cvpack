"""
.. class:: CVWriter
   :platform: Linux, MacOS, Windows
   :synopsis: A custom writer for reporting collective variable data

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import openmm as mm

from ..collective_variable import CollectiveVariable
from .custom_writer import CustomWriter


class CVWriter(CustomWriter):
    """
    A custom writer for reporting collective variable data.

    Parameters
    ----------
    variable
        The collective variable whose values and/or effective masses will be reported.
    value
        If `True`, report the values of the collective variable.
    emass
        If `True`, report the effective masses of the collective variable.

    Examples
    --------
    >>> import cvpack
    >>> import openmm
    >>> from cvpack.reporting import StateDataReporter, CVWriter
    >>> from math import pi
    >>> from openmm import app, unit
    >>> from sys import stdout
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> phi.addToSystem(model.system)
    >>> psi = cvpack.Torsion(8, 14, 16, 18, name="psi")
    >>> psi.addToSystem(model.system)
    >>> reporter = StateDataReporter(
    ...     stdout,
    ...     100,
    ...     writers=[
    ...         CVWriter(phi, value=True, emass=True),
    ...         CVWriter(psi, value=True, emass=True),
    ...     ],
    ...     step=True,
    ... )
    >>> integrator = openmm.LangevinIntegrator(
    ...     300 * unit.kelvin,
    ...     1 / unit.picosecond,
    ...     2 * unit.femtosecond,
    ... )
    >>> integrator.setRandomNumberSeed(1234)
    >>> simulation = app.Simulation(model.topology, model.system, integrator)
    >>> simulation.context.setPositions(model.positions)
    >>> simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 5678)
    >>> simulation.reporters.append(reporter)
    >>> simulation.step(1000)  # doctest: +SKIP
    #"Step","phi (rad)",...,"emass[psi] (nm**2 Da/(rad**2))"
    100,2.7102...,0.04970...,3.1221...,0.05386...
    200,2.1573...,0.04481...,2.9959...,0.05664...
    300,2.0859...,0.04035...,-3.001...,0.04506...
    400,2.8061...,0.05399...,3.0792...,0.04992...
    500,-2.654...,0.04784...,3.1139...,0.05592...
    600,3.1390...,0.05137...,-3.071...,0.05063...
    700,2.1133...,0.04145...,3.1047...,0.04724...
    800,1.7348...,0.04123...,-3.004...,0.05548...
    900,1.6273...,0.03007...,3.1277...,0.05271...
    1000,1.680...,0.03749...,2.9692...,0.04450...
    """

    def __init__(
        self,
        variable: CollectiveVariable,
        value: bool = False,
        emass: bool = False,
    ) -> None:
        if not isinstance(variable, CollectiveVariable):
            raise TypeError("variable must be a CollectiveVariable object")
        if not (value or emass):
            raise ValueError("At least one of value or effective_mass must be True")
        self._cv = variable
        self._value = value
        self._emass = emass

    def getHeaders(self) -> list[str]:
        headers = []
        if self._value:
            headers.append(f"{self._cv.getName()} ({self._cv.getUnit().get_symbol()})")
        if self._emass:
            headers.append(
                f"emass[{self._cv.getName()}] ({self._cv.getMassUnit().get_symbol()})"
            )
        return headers

    def getValues(self, simulation: mm.app.Simulation) -> list[float]:
        context = simulation.context
        values = []
        if self._value:
            values.append(self._cv.getValue(context) / self._cv.getUnit())
        if self._emass:
            values.append(self._cv.getEffectiveMass(context) / self._cv.getMassUnit())
        return values
