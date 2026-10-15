"""
.. class:: StateDataReporter
   :platform: Linux, MacOS, Windows
   :synopsis: An extension of the OpenMM `StateDataReporter`_ class to include writers

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import io
import typing as t

import openmm
from openmm import app as mmapp

from .custom_writer import CustomWriter


class StateDataReporter(mmapp.StateDataReporter):
    """
    An extended version of OpenMM's `openmm.app.StateDataReporter`_ class that includes
    custom writers for reporting additional simulation data.

    .. _openmm.app.StateDataReporter: http://docs.openmm.org/latest/api-python/
        generated/openmm.app.statedatareporter.StateDataReporter.html

    A custom writer is an object that includes the following methods:

    1. **getHeaders**: returns a list of strings containing the headers to be added
       to the report. It must have the following signature:

    .. code-block::

        def getHeaders(self) -> List[str]:
            pass

    2. **getValues**: returns a list of floats containing the values to be added to
       the report at a given time step. It must have the following signature:

    .. code-block::

        def getValues(self, simulation: openmm.app.Simulation) -> List[float]:
            pass

    3. **initialize** (optional): performs any necessary setup before the first report.
       If present, it must have the following signature:

    .. code-block::

        def initialize(self, simulation: openmm.app.Simulation) -> None:
            pass

    Parameters
    ----------
    file
        The file to write to. This can be a file name or a file object.
    reportInterval
        The interval (in time steps) at which to report data.
    writers
        A sequence of custom writers.
    **kwargs
        Additional keyword arguments to be passed to the `StateDataReporter`_
        constructor.

    Examples
    --------
    >>> import cvpack
    >>> import openmm
    >>> from math import pi
    >>> from cvpack import reporting
    >>> from openmm import app, unit
    >>> from sys import stdout
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> psi = cvpack.Torsion(8, 14, 16, 18, name="psi")
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
    >>> reporter = reporting.StateDataReporter(
    ...     stdout,
    ...     100,
    ...     writers=[
    ...         reporting.CVWriter(umbrella, value=True, emass=True),
    ...         reporting.MetaCVWriter(
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
    #"Step","umbrella (kJ/mol)",...,"d[umbrella]/d[psi0] (kJ/(mol rad))"
    100,11.26...,40.371...
    200,7.463...,27.910...
    300,2.558...,-12.74...
    400,6.199...,3.9768...
    500,8.827...,41.878...
    600,3.761...,25.262...
    700,3.388...,25.342...
    800,1.071...,11.349...
    900,8.586...,37.380...
    1000,5.84...,31.159...
    """

    def __init__(
        self,
        file: str | io.TextIOBase,
        reportInterval: int,
        writers: t.Sequence[CustomWriter] = (),
        **kwargs,
    ) -> None:
        super().__init__(file, reportInterval, **kwargs)
        if not all(isinstance(w, CustomWriter) for w in writers):
            raise TypeError("All items in writers must satisfy the Writer protocol")
        self._writers = writers
        self._back_steps = sum([self._speed, self._elapsedTime, self._remainingTime])

    def _expand(self, sequence: list, addition: t.Iterable) -> list:
        pos = len(sequence) - self._back_steps
        return sum(addition, sequence[:pos]) + sequence[pos:]

    def _initializeConstants(self, simulation: mmapp.Simulation) -> None:
        super()._initializeConstants(simulation)
        for writer in self._writers:
            if hasattr(writer, "initialize"):
                writer.initialize(simulation)

    def _constructHeaders(self) -> list[str]:
        return self._expand(
            super()._constructHeaders(),
            (w.getHeaders() for w in self._writers),
        )

    def _constructReportValues(
        self, simulation: mmapp.Simulation, state: openmm.State
    ) -> list[float]:
        return self._expand(
            super()._constructReportValues(simulation, state),
            (w.getValues(simulation) for w in self._writers),
        )
