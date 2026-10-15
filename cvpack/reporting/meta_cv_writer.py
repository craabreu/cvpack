"""
.. class:: MetaCVWriter
   :platform: Linux, MacOS, Windows
   :synopsis: A custom writer for reporting meta-collective variable data

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm as mm
from openmm import unit as mmunit

from ..meta_collective_variable import MetaCollectiveVariable
from .custom_writer import CustomWriter


class MetaCVWriter(CustomWriter):
    """
    A custom writer for reporting meta-collective variable data.

    Parameters
    ----------
    metaCV
        The meta-collective variable whose associated values will be reported.
    values
        The names of the inner variables whose values will be reported.
    emasses
        The names of the inner variables whose effective masses will be reported.
    parameters
        The names of the parameters whose values will be reported.
    derivatives
        The names of the parameters with respect to which the derivatives of the
        meta-collective variable will be reported.
    allowReinitialization
        Whether to allow context reinitialization when the meta-collective variable
        shares its force group with other forces (needed for derivatives only).

    Raises
    ------
    ValueError
        If any name does not match an inner variable or a parameter

    Examples
    --------
    >>> import cvpack
    >>> import openmm
    >>> from math import pi
    >>> from cvpack.reporting import StateDataReporter, MetaCVWriter
    >>> from openmm import app, unit
    >>> from sys import stdout
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
    >>> psi = cvpack.Torsion(6, 8, 14, 16, name="psi")
    >>> umbrella = cvpack.MetaCollectiveVariable(
    ...     f"0.5*kappa*(min(dphi,{2*pi}-dphi)^2+min(dpsi,{2*pi}-dpsi)^2)"
    ...     "; dphi=abs(phi-phi0); dpsi=abs(psi-psi0)",
    ...     [phi, psi],
    ...     unit.kilojoules_per_mole,
    ...     name="umbrella",
    ...     kappa=100 * unit.kilojoules_per_mole/unit.radian**2,
    ...     phi0=5*pi/6 * unit.radian,
    ...     psi0=-5*pi/6 * unit.radian,
    ... )
    >>> reporter = StateDataReporter(
    ...     stdout,
    ...     100,
    ...     writers=[
    ...         MetaCVWriter(
    ...             umbrella,
    ...             values=["phi", "psi"],
    ...             emasses=["phi", "psi"],
    ...             parameters=["phi0", "psi0"],
    ...             derivatives=["phi0", "psi0"],
    ...         ),
    ...     ],
    ...     step=True,
    ... )
    >>> integrator = openmm.LangevinIntegrator(
    ...     300 * unit.kelvin,
    ...     1 / unit.picosecond,
    ...     2 * unit.femtosecond,
    ... )
    >>> integrator.setRandomNumberSeed(1234)
    >>> umbrella.addToSystem(model.system)
    >>> simulation = app.Simulation(model.topology, model.system, integrator)
    >>> simulation.context.setPositions(model.positions)
    >>> simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 5678)
    >>> simulation.reporters.append(reporter)
    >>> simulation.step(1000)  # doctest: +SKIP
    #"Step","phi (rad)",...,"d[umbrella]/d[psi0] (kJ/(mol rad))"
    100,2.6766...,-20.519...
    200,2.3694...,0.7804...
    300,3.1342...,49.272...
    400,2.8179...,-10.012...
    500,2.7565...,-8.9529...
    600,2.6956...,23.049...
    700,2.9331...,25.068...
    800,2.6988...,-0.4481...
    900,2.7477...,9.5528...
    1000,2.7744...,21.986...
    """

    def __init__(
        self,
        metaCV: MetaCollectiveVariable,
        values: t.Sequence[str] = (),
        emasses: t.Sequence[str] = (),
        parameters: t.Sequence[str] = (),
        derivatives: t.Sequence[str] = (),
        allowReinitialization: bool = False,
    ) -> None:
        inner_cvs = {cv.getName(): cv for cv in metaCV.getInnerVariables()}
        all_parameters = metaCV.getParameterDefaultValues()

        def select(names, available, kind):
            unknown = [name for name in names if name not in available]
            if unknown:
                raise ValueError(f"Unknown {kind}: {', '.join(unknown)}")
            return {name: available[name] for name in names}

        self._meta_cv = metaCV
        self._values = list(select(values, inner_cvs, "inner variables").values())
        self._masses = list(select(emasses, inner_cvs, "inner variables").values())
        self._parameters = select(parameters, all_parameters, "parameters")
        self._derivatives = select(derivatives, all_parameters, "parameters")
        self._allow_reinitialization = allowReinitialization

    def getHeaders(self) -> list[str]:
        headers = []

        def add_header(name: str, unit: mmunit.Unit) -> None:
            headers.append(f"{name} ({unit.get_symbol()})")

        for cv in self._values:
            add_header(cv.getName(), cv.getUnit())
        for cv in self._masses:
            add_header(f"emass[{cv.getName()}]", cv.getMassUnit())
        for name, quantity in self._parameters.items():
            add_header(name, quantity.unit)
        for name, quantity in self._derivatives.items():
            add_header(
                f"d[{self._meta_cv.getName()}]/d[{name}]",
                self._meta_cv.getUnit() / quantity.unit,
            )
        return headers

    def getValues(self, simulation: mm.app.Simulation) -> list[float]:
        context = simulation.context
        values = []
        if self._values:
            inner_values = self._meta_cv.getInnerValues(context)
            for cv in self._values:
                values.append(inner_values[cv.getName()] / cv.getUnit())
        if self._masses:
            inner_masses = self._meta_cv.getInnerEffectiveMasses(context)
            for cv in self._masses:
                values.append(inner_masses[cv.getName()] / cv.getMassUnit())
        if self._parameters:
            parameters = self._meta_cv.getParameterValues(context)
            for name, quantity in self._parameters.items():
                values.append(parameters[name] / quantity.unit)
        if self._derivatives:
            derivatives = self._meta_cv.getParameterDerivatives(
                context, self._allow_reinitialization
            )
            for name, quantity in self._derivatives.items():
                values.append(
                    derivatives[name] / (self._meta_cv.getUnit() / quantity.unit)
                )
        return values
