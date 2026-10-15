"""
.. class:: CollectiveVariable
   :platform: Linux, MacOS, Windows
   :synopsis: An abstract class with common attributes and methods for all CVs

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm
import yaml
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from .serialization import Serializable
from .units import Quantity, ScalarQuantity, Unit
from .utils import (
    compute_effective_mass,
    find_unused_force_group,
    get_single_force_state,
    preprocess_args,
)


class CollectiveVariable(Serializable):
    r"""
    An abstract class with common attributes and methods for all CVs.
    """

    _unit: Unit = Unit("dimensionless")
    _mass_unit: Unit = Unit("dalton")
    _args: dict[str, t.Any] = {}
    _periodic_bounds: Quantity | None = None

    def __getstate__(self) -> dict[str, t.Any]:
        # The force group is not serialized: addToSystem assigns it
        return {**self._args, "name": self.getName()}

    def __setstate__(self, keywords: dict[str, t.Any]) -> None:
        self.__init__(**keywords)

    def __copy__(self) -> "CollectiveVariable":
        return yaml.safe_load(yaml.safe_dump(self))

    def __deepcopy__(self, _) -> "CollectiveVariable":
        return yaml.safe_load(yaml.safe_dump(self))

    @preprocess_args
    def _registerCV(
        self,
        name: str,
        cvUnit: Unit,
        **kwargs: t.Any,
    ) -> None:
        """
        Register the newly created CollectiveVariable subclass instance.

        This method must always be called from Subclass.__init__.

        Parameters
        ----------
        name
            The name of this collective variable.
        cvUnit
            The unit of measurement of this collective variable. It must be a unit
            in the MD unit system (mass in Da, distance in nm, time in ps,
            temperature in K, energy in kJ/mol, angle in rad).
        kwargs
            The keyword arguments needed to construct this collective variable

        Raises
        ------
        ValueError
            If the unit is not compatible with the MD unit system
        """
        if (1 * cvUnit).value_in_unit_system(mmunit.md_unit_system) != 1:
            raise ValueError(
                f"Unit {cvUnit} is not compatible with the MD unit system."
            )
        mmswig.Force_setName(self, name)
        self._unit = cvUnit
        self._mass_unit = Unit(mmunit.dalton * (mmunit.nanometers / self._unit) ** 2)
        self._args = {"name": name}
        self._args.update(kwargs)

    def _registerPeriodicBounds(
        self, lower: ScalarQuantity, upper: ScalarQuantity
    ) -> None:
        """
        Register the periodic bounds of this collective variable.

        This method must be called from Subclass.__init__ if the collective variable is
        periodic.

        Parameters
        ----------
        lower
            The lower bound of this collective variable
        upper
            The upper bound of this collective variable
        """
        if mmunit.is_quantity(lower):
            lower = lower.value_in_unit(self.getUnit())
        if mmunit.is_quantity(upper):
            upper = upper.value_in_unit(self.getUnit())
        self._periodic_bounds = Quantity((lower, upper), self.getUnit())

    def _setUnusedForceGroup(self, system: openmm.System) -> int:
        """
        Set the force group of this collective variable to the lowest force group not
        used by any force in an :OpenMM:`System`.

        .. note::

            Evaluating a collective variable (see :meth:`getValue`) or computing its
            effective mass (see :meth:`getEffectiveMass`) is more efficient when the
            collective variable is the only force in its own force group.

        Parameters
        ----------
        system
            The system to search for unused force groups

        Returns
        -------
        int
            The force group assigned to this collective variable

        Raises
        ------
        RuntimeError
            If all force groups are already in use
        """
        new_group = find_unused_force_group(system)
        mmswig.Force_setForceGroup(self, new_group)
        return new_group

    def getName(self) -> str:
        """
        Get the name of this collective variable.
        """
        return mmswig.Force_getName(self)

    def getUnit(self) -> Unit:
        """
        Get the unit of measurement of this collective variable.
        """
        return self._unit

    def getMassUnit(self) -> Unit:
        """
        Get the unit of measurement of the effective mass of this collective variable.
        """
        return self._mass_unit

    def getPeriodicBounds(self) -> Quantity | None:
        """
        Get the periodic bounds of this collective variable.

        Returns
        -------
        Union[Quantity, None]
            The periodic bounds of this collective variable or None if it is not
            periodic
        """
        return self._periodic_bounds

    def addToSystem(
        self, system: openmm.System, setUnusedForceGroup: bool = True
    ) -> None:
        """
        Add this collective variable to an :OpenMM:`System`.

        Parameters
        ----------
        system
            The system to which this collective variable should be added
        setUnusedForceGroup
            If True, the force group of this collective variable will be set to the
            first available force group in the system
        """
        if setUnusedForceGroup:
            self._setUnusedForceGroup(system)
        system.addForce(self)

    def getValue(
        self, context: openmm.Context, allowReinitialization: bool = False
    ) -> Quantity:
        """
        Evaluate this collective variable at a given :OpenMM:`Context`.

        If this collective variable shares the force group with other forces, then its
        evaluation requires reinitializing the :OpenMM:`Context` twice at each call.
        This is inefficient and should be avoided. To allow this behavior, the
        ``allowReinitialization`` parameter must be set to True.

        .. note::

            By adding this collective variable to the system using the
            :meth:`addToSystem` method, the force group of this collective variable
            is set to an available force group in the system by default.

        Parameters
        ----------
        context
            The context at which this collective variable should be evaluated.
        allowReinitialization
            If True, the context will be reinitialized if necessary.

        Returns
        -------
        Quantity
            The value of this collective variable at the given context.

        Example
        -------
        In this example, we compute the values of the backbone dihedral angles and
        the radius of gyration of an alanine dipeptide molecule in water:

        >>> import cvpack
        >>> import openmm
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideExplicit()
        >>> top = model.mdtraj_topology
        >>> backbone_atoms = top.select("not water and name N CA C")
        >>> phi = cvpack.Torsion(*backbone_atoms[0:4])
        >>> psi = cvpack.Torsion(*backbone_atoms[1:5])
        >>> radius_of_gyration = cvpack.RadiusOfGyration(
        ...     top.select("not water")
        ... )
        >>> for cv in [phi, psi, radius_of_gyration]:
        ...     cv.addToSystem(model.system)
        >>> context = openmm.Context(
        ...     model.system, openmm.VerletIntegrator(0)
        ... )
        >>> context.setPositions(model.positions)
        >>> phi.getValue(context)
        -3.1415... rad
        >>> psi.getValue(context)
        3.1415... rad
        >>> radius_of_gyration.getValue(context)
        0.29514... nm
        """
        state = get_single_force_state(
            self,
            context,
            allowReinitialization,
            getEnergy=True,
        )
        value = state.getPotentialEnergy().value_in_unit(mmunit.kilojoules_per_mole)
        return Quantity(value, self.getUnit())

    def getEffectiveMass(
        self, context: openmm.Context, allowReinitialization: bool = False
    ) -> Quantity:
        r"""
        Compute the effective mass of this collective variable at a given
        :OpenMM:`Context`.

        The effective mass of a collective variable :math:`q({\bf r})` is defined as
        :cite:`Chipot_2007`:

        .. math::

            m_\mathrm{eff}({\bf r}) = \left(
                \sum_{i=1}^N \frac{1}{m_i} \left\| \frac{dq}{d{\bf r}_i} \right\|^2
            \right)^{-1}

        If this collective variable shares the force group with other forces, then
        evaluating its effective mass requires reinitializing the :OpenMM:`Context`
        twice at each call. This is inefficient and should be avoided. To allow this
        behavior, the ``allowReinitialization`` parameter must be set to True.

        .. note::

            By adding this collective variable to the system using the
            :meth:`addToSystem` method, the force group of this collective variable
            is set to an available force group in the system by default.

        Parameters
        ----------
        context
            The context at which this collective variable's effective mass should be
            evaluated.
        allowReinitialization
            If True, the context will be reinitialized if necessary.

        Returns
        -------
        Quantity
            The effective mass of this collective variable at the given context.

        Example
        -------
        In this example, we compute the effective masses of the backbone dihedral
        angles and the radius of gyration of an alanine dipeptide molecule in water:

        >>> import cvpack
        >>> import openmm
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideExplicit()
        >>> top = model.mdtraj_topology
        >>> backbone_atoms = top.select("not water and name N CA C")
        >>> phi = cvpack.Torsion(*backbone_atoms[0:4])
        >>> psi = cvpack.Torsion(*backbone_atoms[1:5])
        >>> radius_of_gyration = cvpack.RadiusOfGyration(
        ...     top.select("not water")
        ... )
        >>> for cv in [phi, psi, radius_of_gyration]:
        ...     cv.addToSystem(model.system)
        >>> context = openmm.Context(
        ...     model.system, openmm.VerletIntegrator(0)
        ... )
        >>> context.setPositions(model.positions)
        >>> phi.getEffectiveMass(context)
        0.04795... nm**2 Da/(rad**2)
        >>> psi.getEffectiveMass(context)
        0.05119... nm**2 Da/(rad**2)
        >>> radius_of_gyration.getEffectiveMass(context)
        30.946... Da
        """
        return Quantity(
            compute_effective_mass(self, context, allowReinitialization),
            self._mass_unit,
        )
