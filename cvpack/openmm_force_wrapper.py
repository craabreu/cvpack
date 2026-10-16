"""
.. class:: OpenMMForceWrapper
   :platform: Linux, MacOS, Windows
   :synopsis: A collective variable built from the potential energy of an OpenMM force

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import typing as t

import openmm
from openmm import unit as mmunit

from .collective_variable import CollectiveVariable
from .units import Unit, VectorQuantity


class OpenMMForceWrapper(CollectiveVariable, openmm.Force):
    r"""
    A collective variable whose numerical value is computed from the potential energy,
    in kJ/mol, of an OpenMM force object.

    Parameters
    ----------
    openmmForce
        The OpenMM force whose potential energy will be used to define the collective
        variable. It can be passed as an instance of :OpenMM:`Force` or as a string
        containing the XML serialization of the force.
    unit
        The unit of measurement of the collective variable. It must be compatible
        with the MD unit system (mass in `daltons`, distance in `nanometers`, time
        in `picoseconds`, temperature in `kelvin`, energy in `kilojoules_per_mol`,
        angle in `radians`). If the collective variable does not have a unit, use
        `dimensionless`.
    periodicBounds
        The periodic bounds of the collective variable if it is periodic, or `None` if
        it is not.
    name
        The name of the collective variable.

    Example
    -------
    >>> import cvpack
    >>> import numpy as np
    >>> import openmm
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> torsion = openmm.CustomTorsionForce("theta")
    >>> _ = torsion.addTorsion(4, 6, 8, 14)
    >>> cv = cvpack.OpenMMForceWrapper(
    ...     torsion,
    ...     unit.radian,
    ...     periodicBounds=[-np.pi, np.pi] * unit.radian,
    ... )
    >>> cv.addToSystem(model.system)
    >>> integrator = openmm.VerletIntegrator(0)
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> cv.getValue(context)
    -3.1415... rad
    >>> cv.getEffectiveMass(context)
    0.04795... nm**2 Da/(rad**2)
    """

    def __init__(
        self,
        openmmForce: openmm.Force | str,
        unit: mmunit.Unit,
        periodicBounds: VectorQuantity | None = None,
        name: str = "openmm_force_wrapper",
    ) -> None:
        if isinstance(openmmForce, openmm.Force):
            openmmForce = openmm.XmlSerializer.serialize(openmmForce)
        unit = Unit(unit)
        self._wrapped_force = openmm.XmlSerializer.deserialize(openmmForce)
        self.this = self._wrapped_force.this
        self._registerCV(
            name,
            unit,
            openmmForce=openmmForce,
            unit=unit,
            periodicBounds=periodicBounds,
        )
        if periodicBounds is not None:
            self._registerPeriodicBounds(*periodicBounds)

    def __getattr__(self, name: str) -> t.Any:
        return getattr(self._wrapped_force, name)


OpenMMForceWrapper.registerTag("!cvpack.OpenMMForceWrapper")
