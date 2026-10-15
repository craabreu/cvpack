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
    allowReinitialization
        Whether to allow context reinitialization when the collective variable shares
        its force group with other forces (see :meth:`CollectiveVariable.getValue`).

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
    >>> phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
    >>> phi.addToSystem(model.system)
    >>> psi = cvpack.Torsion(6, 8, 14, 16, name="psi")
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
    >>> # Integrate only the physical forces, so phi and psi do not bias the dynamics
    >>> integrator.setIntegrationForceGroups({0})
    >>> simulation = app.Simulation(model.topology, model.system, integrator)
    >>> simulation.context.setPositions(model.positions)
    >>> simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 5678)
    >>> simulation.reporters.append(reporter)
    >>> simulation.step(1000)  # doctest: +SKIP
    #"Step","phi (rad)",...,"emass[psi] (nm**2 Da/(rad**2))"
    100,-2.4309...,0.0404...,2.7368...,0.0497...
    200,-2.0470...,0.0386...,2.2298...,0.0456...
    300,-2.4721...,0.0424...,2.2528...,0.0422...
    400,-2.4032...,0.0406...,-3.1341...,0.0546...
    500,-2.8165...,0.0460...,-2.5845...,0.0458...
    600,-2.6256...,0.0401...,3.1382...,0.0509...
    700,-2.4156...,0.0463...,2.1037...,0.0406...
    800,-2.3871...,0.0372...,1.8683...,0.0408...
    900,-1.8631...,0.0322...,2.0393...,0.0374...
    1000,-1.0804...,0.0240...,1.8363...,0.0354...
    """

    def __init__(
        self,
        variable: CollectiveVariable,
        value: bool = False,
        emass: bool = False,
        allowReinitialization: bool = False,
    ) -> None:
        if not isinstance(variable, CollectiveVariable):
            raise TypeError("variable must be a CollectiveVariable object")
        if not (value or emass):
            raise ValueError("At least one of value or emass must be True")
        self._cv = variable
        self._value = value
        self._emass = emass
        self._allow_reinitialization = allowReinitialization

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
            value = self._cv.getValue(context, self._allow_reinitialization)
            values.append(value / self._cv.getUnit())
        if self._emass:
            mass = self._cv.getEffectiveMass(context, self._allow_reinitialization)
            values.append(mass / self._cv.getMassUnit())
        return values
