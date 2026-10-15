"""
.. class:: MetaCollectiveVariable
   :platform: Linux, MacOS, Windows
   :synopsis: A function of other collective variables

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import typing as t
from copy import copy

import openmm
from openmm import unit as mmunit

from .collective_variable import CollectiveVariable
from .units import Quantity, ScalarQuantity, VectorQuantity, in_md_units
from .utils import compute_effective_mass, get_single_force_state


class MetaCollectiveVariable(CollectiveVariable, openmm.CustomCVForce):
    r"""
    A collective variable that is a function of other collective variables:

    .. math::

        f({\bf r}) = F\left(c_1({\bf r}), c_2({\bf r}), \ldots, c_n({\bf r})\right)

    where :math:`F(c_1,c_2,\ldots,c_n)` is a user-defined function and
    :math:`c_i({\bf r})` is the value of the :math:`i`-th collective variable at a
    given configuration :math:`{\bf r}`.

    The function :math:`F` is defined as a string and can be any expression supported
    by the :OpenMM:`CustomCVForce` class. If the expression contains named parameters,
    the value of each parameter can be passed in one of two ways:

    #. By a semicolon-separated definition in the function string, such as described
       in the :OpenMM:`CustomCVForce` documentation. In this case, the parameter value
       is fixed.

    #. By a scalar passed as a keyword argument to the
       :class:`MetaCollectiveVariable` constructor. In this case, the parameter will
       become available for on-the-fly modification during a simulation via the
       ``setParameter`` method of an :OpenMM:`Context` object. **Warning**: other
       collective variables or :OpenMM:`Force` objects in the same system will share
       the same values of equal-named parameters.

    Parameters
    ----------
    function
        The function to be evaluated. It must be a valid :OpenMM:`CustomCVForce`
        expression.
    variables
        A sequence of :class:`CollectiveVariable` instances that represent the
        collective variables on which the meta-collective variable depends. The name
        of each collective variable must be unique and match a symbol used in the
        function.
    unit
        The unit of measurement of the collective variable. It must be compatible
        with the MD unit system (mass in `daltons`, distance in `nanometers`, time
        in `picoseconds`, temperature in `kelvin`, energy in `kilojoules_per_mol`,
        angle in `radians`). If the collective variable does not have a unit, use
        `unit.dimensionless`.
    periodicBounds
        The periodic bounds of the collective variable if it is periodic, or `None` if
        it is not periodic.

    Keyword Args
    ------------
    **parameters
        The named parameters of the function, if any. They will become settable context
        parameters when this meta-collective variable is added to an :OpenMM:`System`.
        The passed objects must be scalar quantities. Their values will be converted to
        OpenMM's MD unit system to serve as default values for the context parameters.

    Example
    -------
    >>> import cvpack
    >>> import math
    >>> import openmm
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
    >>> driving_force = cvpack.MetaCollectiveVariable(
    ...     f"0.5*kappa*min(delta,{2*math.pi}-delta)^2"
    ...     "; delta=abs(phi-phi0)",
    ...     [phi],
    ...     unit.kilojoules_per_mole,
    ...     kappa = 1e3 * unit.kilojoules_per_mole/unit.radian**2,
    ...     phi0 = 120 * unit.degrees
    ... )
    >>> driving_force.addToSystem(model.system)
    >>> integrator = openmm.VerletIntegrator(0)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> driving_force.getParameterDefaultValues()
    {'kappa': 1000.0 kJ/(mol rad**2), 'phi0': 2.094... rad}
    >>> driving_force.getParameterValues(context)
    {'kappa': 1000.0 kJ/(mol rad**2), 'phi0': 2.094... rad}
    >>> driving_force.getValue(context)
    548.3... kJ/mol
    >>> driving_force.getInnerValues(context)
    {'phi': -3.14... rad}
    >>> driving_force.getInnerEffectiveMasses(context)
    {'phi': 0.04795... nm**2 Da/(rad**2)}
    >>> driving_force.getParameterDerivatives(context)
    {'kappa': 0.548... rad**2, 'phi0': -1047.20... kJ/(mol rad)}
    """

    def __init__(
        self,
        function: str,
        variables: t.Iterable[CollectiveVariable],
        unit: mmunit.Unit,
        periodicBounds: VectorQuantity | None = None,
        name: str = "meta_collective_variable",
        **parameters: ScalarQuantity,
    ) -> None:
        super().__init__(function)
        variables = list(variables)
        names = [cv.getName() for cv in variables]
        if len(set(names)) != len(names):
            raise ValueError(f"The names of the variables must be unique: {names}")
        self._cvs = tuple(map(copy, variables))
        self._parameters = {k: in_md_units(v) for k, v in parameters.items()}
        for cv in self._cvs:
            self.addCollectiveVariable(cv.getName(), cv)
        for parameter, value in self._parameters.items():
            self.addGlobalParameter(parameter, value / value.unit)
            self.addEnergyParameterDerivative(parameter)
        self._registerCV(
            name,
            unit,
            function=function,
            variables=variables,
            unit=unit,
            periodicBounds=periodicBounds,
            **self._parameters,
        )
        if periodicBounds is not None:
            self._registerPeriodicBounds(*periodicBounds)

    def getInnerVariables(self) -> tuple[CollectiveVariable]:
        """
        Get the collective variables on which the meta-collective variable depends.

        Returns
        -------
        Tuple[CollectiveVariable]
            A tuple with the collective variables.
        """
        return self._cvs

    def getInnerValues(self, context: openmm.Context) -> dict[str, Quantity]:
        """
        Get the values of the collective variables on which the meta-collective variable
        depends. The values are returned as a dictionary with the names of the
        collective variables as keys.

        Parameters
        ----------
        context
            The context in which the collective variables will be evaluated.

        Returns
        -------
        Dict[str, Quantity]
            A dictionary with the names of the collective variables as keys and their
            values as values.
        """
        values = self.getCollectiveVariableValues(context)
        return {
            cv.getName(): Quantity(value, cv.getUnit())
            for cv, value in zip(self._cvs, values)
        }

    def getInnerEffectiveMasses(self, context: openmm.Context) -> dict[str, Quantity]:
        """
        Get the effective masses of the collective variables on which the
        meta-collective variable depends. The effective masses are calculated from the
        forces acting on the particles that represent the collective variables.

        Parameters
        ----------
        context
            The context in which the collective variables will be evaluated.

        Returns
        -------
        Dict[str, Quantity]
            A dictionary with the names of the collective variables as keys and their
            effective masses as values.
        """
        self.getCollectiveVariableValues(context)  # update the inner context positions
        inner_context = self.getInnerContext(context)
        forces = inner_context.getSystem().getForces()
        system = context.getSystem()
        if any(map(system.isVirtualSite, range(system.getNumParticles()))):
            # The inner context has no virtual sites, so the forces on them would not
            # be distributed to the atoms that define them
            inner_context = self._contextWithVirtualSites(forces, context)
            forces = inner_context.getSystem().getForces()
        masses = [compute_effective_mass(force, inner_context) for force in forces]
        return {
            cv.getName(): Quantity(mass, cv.getMassUnit())
            for cv, mass in zip(self._cvs, masses)
        }

    @staticmethod
    def _contextWithVirtualSites(
        forces: t.Sequence[openmm.Force], context: openmm.Context
    ) -> openmm.Context:
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(context.getSystem())
        )
        while system.getNumForces() > 0:
            system.removeForce(0)
        for group, force in enumerate(forces):
            force_copy = openmm.XmlSerializer.deserialize(
                openmm.XmlSerializer.serialize(force)
            )
            force_copy.setForceGroup(group)
            system.addForce(force_copy)
        new_context = openmm.Context(
            system, openmm.VerletIntegrator(1.0), context.getPlatform()
        )
        state = context.getState(getPositions=True)
        new_context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        new_context.setPositions(state.getPositions())
        return new_context

    def getParameterDefaultValues(self) -> dict[str, Quantity]:
        """
        Get the default values of the named parameters of this meta-collective variable.

        Returns
        -------
        Dict[str, Quantity]
            A dictionary with the names of the named parameters as keys and their
            default values as values.
        """
        return self._parameters.copy()

    def getParameterValues(self, context: openmm.Context) -> dict[str, Quantity]:
        """
        Get the values of the named parameters of this meta-collective variable. The
        values are returned as a dictionary with the names of the parameters as keys.

        Parameters
        ----------
        context
            The context in which the named parameters will be evaluated.

        Returns
        -------
        Dict[str, Quantity]
            A dictionary with the names of the named parameters as keys and their values
            as values.
        """
        return {
            name: Quantity(context.getParameter(name), parameter.unit)
            for name, parameter in self._parameters.items()
        }

    def getParameterDerivatives(
        self,
        context: openmm.Context,
        allowReinitialization: bool = False,
    ) -> dict[str, Quantity]:
        """
        Get the derivatives of the named parameters of this meta-collective variable.

        Returns
        -------
        Dict[str, Quantity]
            A dictionary with the names of the named parameters as keys and their
            derivatives as values.
        """
        state = get_single_force_state(
            self, context, allowReinitialization, getParameterDerivatives=True
        )
        derivatives = state.getEnergyParameterDerivatives()
        return {
            name: Quantity(derivatives[name], self._unit / parameter.unit)
            for name, parameter in self._parameters.items()
        }


MetaCollectiveVariable.registerTag("!cvpack.MetaCollectiveVariable")
