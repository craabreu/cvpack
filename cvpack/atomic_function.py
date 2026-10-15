"""
.. class:: AtomicFunction
   :platform: Linux, MacOS, Windows
   :synopsis: A generic function of the coordinates of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import numpy as np
import openmm
from numpy.typing import ArrayLike
from openmm import unit as mmunit

from .base_custom_function import BaseCustomFunction
from .units import ScalarQuantity, VectorQuantity


class AtomicFunction(BaseCustomFunction, openmm.CustomCompoundBondForce):
    r"""
    A generic function of the coordinates of :math:`m` groups of :math:`n` atoms:

    .. math::

        f({\bf r}) = \sum_{i=1}^m F\left(
            {\bf r}_{i,1}, {\bf r}_{i,2}, \dots, {\bf r}_{i,n}
        \right)

    where :math:`F` is a user-defined function and :math:`{\bf r}_{i,j}` is the
    coordinate of the :math:`j`-th atom of the :math:`i`-th group.

    The function :math:`F` is defined as a string and can be any expression supported
    by :OpenMM:`CustomCompoundBondForce`. If the expression contains named parameters,
    the value of each parameter can be passed in one of three ways:

    #. By a semicolon-separated definition in the function string, such as described
       in the :OpenMM:`CustomCompoundBondForce` documentation. In this case, the
       parameter value will be the same for all groups of atoms.

    #. By a 1D array or list of length :math:`m` passed as a keyword argument to
       the :class:`AtomicFunction` constructor. In this case, each value will be
       assigned to a different group of atoms.

    #. By a scalar passed as a keyword argument to the :class:`AtomicFunction`
       constructor. In this case, the parameter will apply to all atom groups and will
       become available for on-the-fly modification during a simulation via the
       ``setParameter`` method of an :OpenMM:`Context` object. **Warning**: other
       collective variables or :OpenMM:`Force` objects in the same system will share
       the same values of equal-named parameters.

    Parameters
    ----------
    function
        The function to be evaluated. It must be a valid
        :OpenMM:`CustomCompoundBondForce` expression.
    groups
        The indices of the atoms in each group, passed as a 2D array-like object of
        shape `(m, n)`, where `m` is the number of groups and `n` is the number of
        atoms per group. If a 1D object is passed, it is assumed that `m` is 1 and
        `n` is the length of the object.
    unit
        The unit of measurement of the collective variable. It must be compatible
        with the MD unit system (mass in `daltons`, distance in `nanometers`, time
        in `picoseconds`, temperature in `kelvin`, energy in `kilojoules_per_mol`,
        angle in `radians`). If the collective variables does not have a unit, use
        `unit.dimensionless`.
    periodicBounds
        The lower and upper bounds of the collective variable if it is periodic, or
        ``None`` if it is not.
    pbc
        Whether to use periodic boundary conditions when computing atomic distances.
        This should be ``True`` for periodic systems.
    name
        The name of the collective variable.

    Keyword Args
    ------------
    **parameters
        The named parameters of the function. Each parameter can be a 1D array-like
        object or a scalar. In the former case, the array must have length :math:`m`
        and each entry will be assigned to a different group of atoms. In the latter
        case, it will define a global :OpenMM:`Context` parameter.

    Raises
    ------
    ValueError
        If the groups are not specified as a 1D or 2D array-like object.
    ValueError
        If the unit of the collective variable is not compatible with the MD unit
        system.

    Example
    -------
    >>> import cvpack
    >>> import openmm
    >>> import numpy as np
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> angle1 = cvpack.Angle(0, 5, 10)
    >>> angle2 = cvpack.Angle(10, 15, 20)
    >>> colvar = cvpack.AtomicFunction(
    ...     "(kappa/2)*(angle(p1, p2, p3) - theta0)^2",
    ...     unit.kilojoules_per_mole,
    ...     [[0, 5, 10], [10, 15, 20]],
    ...     kappa = 1000 * unit.kilojoules_per_mole/unit.radian**2,
    ...     theta0 = [np.pi/2, np.pi/3] * unit.radian,
    ... )
    >>> for cv in [angle1, angle2, colvar]:
    ...     cv.addToSystem(model.system)
    >>> integrator = openmm.VerletIntegrator(0)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> theta1 = angle1.getValue(context) / openmm.unit.radian
    >>> theta2 = angle2.getValue(context) / openmm.unit.radian
    >>> 500*((theta1 - np.pi/2)**2 + (theta2 - np.pi/3)**2)
    429.479...
    >>> colvar.getValue(context)
    429.479... kJ/mol
    >>> context.setParameter(
    ...     "kappa",
    ...     2000 * unit.kilojoules_per_mole/unit.radian**2
    ... )
    >>> colvar.getValue(context)
    858.958... kJ/mol
    """

    def __init__(
        self,
        function: str,
        unit: mmunit.Unit,
        groups: ArrayLike,
        periodicBounds: VectorQuantity | None = None,
        pbc: bool = False,
        name: str = "atomic_function",
        **parameters: ScalarQuantity | VectorQuantity,
    ) -> None:
        groups = np.atleast_2d(groups)
        num_groups, atoms_per_group, *other_dimensions = groups.shape
        if other_dimensions:
            raise ValueError("Array `groups` cannot have more than 2 dimensions")
        super().__init__(atoms_per_group, function)
        overalls, perbonds = self._extractParameters(num_groups, **parameters)
        self._addParameters(overalls, perbonds, groups, pbc)
        groups = [[int(atom) for atom in group] for group in groups]
        self._registerCV(
            name,
            unit,
            function=function,
            unit=unit,
            groups=groups,
            periodicBounds=periodicBounds,
            pbc=pbc,
            **overalls,
            **perbonds,
        )
        if periodicBounds is not None:
            self._registerPeriodicBounds(*periodicBounds)

    @classmethod
    def _fromCustomForce(  # noqa: PLR0912
        cls,
        force: openmm.CustomAngleForce
        | openmm.CustomBondForce
        | openmm.CustomCompoundBondForce
        | openmm.CustomExternalForce
        | openmm.CustomTorsionForce,
        unit: mmunit.Unit,
        periodicBounds: VectorQuantity | None = None,
        pbc: bool = False,
    ) -> AtomicFunction:
        """
        Create a :class:`AtomicFunction` from an object of :openmm:`CustomBondForce`,
        :openmm:`CustomAngleForce`, :openmm:`CustomTorsionForce`, or
        :openmm:`CustomCompoundBondForce` class.
        """
        if isinstance(force, openmm.CustomExternalForce):
            number, item, definition = 1, "Particle", "; x=x1; y=y1; z=z1"
        elif isinstance(force, openmm.CustomBondForce):
            number, item, definition = 2, "Bond", "; r=distance(p1, p2)"
        elif isinstance(force, openmm.CustomAngleForce):
            number, item, definition = 3, "Angle", "; theta=angle(p1, p2, p3)"
        elif isinstance(force, openmm.CustomTorsionForce):
            number, item, definition = 4, "Torsion", "; theta=dihedral(p1, p2, p3, p4)"
        elif isinstance(force, openmm.CustomCompoundBondForce):
            number, item, definition = force.getNumParticlesPerBond(), "Bond", ""
        else:
            raise TypeError(f"Unsupported force type: {type(force).__name__}")
        function = force.getEnergyFunction() + definition
        parameters = {}
        for i in range(force.getNumGlobalParameters()):
            name = force.getGlobalParameterName(i)
            value = force.getGlobalParameterDefaultValue(i)
            parameters[name] = value
        per_item_parameter_names = []
        for i in range(getattr(force, f"getNumPer{item}Parameters")()):
            per_item_parameter_names.append(
                getattr(force, f"getPer{item}ParameterName")(i)
            )
        for name in per_item_parameter_names:
            parameters[name] = []
        atoms = []
        for i in range(getattr(force, f"getNum{item}s")()):
            if isinstance(force, openmm.CustomCompoundBondForce):
                indices, per_item_parameters = force.getBondParameters(i)
            else:
                *indices, per_item_parameters = getattr(
                    force,
                    f"get{item}Parameters",
                )(i)
            atoms.append(indices)
            for name, value in zip(per_item_parameter_names, per_item_parameters):
                parameters[name].append(value)
        atoms = np.asarray(atoms).reshape(-1, number)
        return cls(function, unit, atoms, periodicBounds, pbc, **parameters)

    @classmethod
    def _fromHarmonicBondForce(
        cls,
        force: openmm.HarmonicBondForce,
        unit: mmunit.Unit,
        pbc: bool = False,
    ) -> AtomicFunction:
        """
        Create a :class:`AtomicFunction` from an :OpenMM:`HarmonicBondForce`.
        """
        parameters = {"r0": [], "k": []}
        atoms = []
        for i in range(force.getNumBonds()):
            *indices, length, k = force.getBondParameters(i)
            atoms.append(indices)
            parameters["r0"].append(length)
            parameters["k"].append(k)
        return cls(
            "(k/2)*(distance(p1, p2)-r0)^2",
            unit,
            atoms,
            None,
            pbc,
            **parameters,
        )

    @classmethod
    def _fromHarmonicAngleForce(
        cls,
        force: openmm.HarmonicAngleForce,
        unit: mmunit.Unit,
        pbc: bool = False,
    ) -> AtomicFunction:
        """
        Create a :class:`AtomicFunction` from an :OpenMM:`HarmonicAngleForce`.
        """
        parameters = {"theta0": [], "k": []}
        atoms = []
        for i in range(force.getNumAngles()):
            *indices, angle, k = force.getAngleParameters(i)
            atoms.append(indices)
            parameters["theta0"].append(angle)
            parameters["k"].append(k)
        return cls(
            "(k/2)*(angle(p1, p2, p3)-theta0)^2",
            unit,
            atoms,
            None,
            pbc,
            **parameters,
        )

    @classmethod
    def _fromPeriodicTorsionForce(
        cls,
        force: openmm.PeriodicTorsionForce,
        unit: mmunit.Unit,
        pbc: bool = False,
    ) -> AtomicFunction:
        """
        Create a :class:`AtomicFunction` from an :OpenMM:`PeriodicTorsionForce`.
        """
        parameters = {"periodicity": [], "phase": [], "k": []}
        atoms = []
        for i in range(force.getNumTorsions()):
            *indices, periodicity, phase, k = force.getTorsionParameters(i)
            atoms.append(indices)
            parameters["periodicity"].append(periodicity)
            parameters["phase"].append(phase)
            parameters["k"].append(k)
        return cls(
            "k*(1 + cos(periodicity*dihedral(p1, p2, p3, p4) - phase))",
            unit,
            atoms,
            None,
            pbc,
            **parameters,
        )

    @classmethod
    def fromOpenMMForce(
        cls,
        force: openmm.Force,
        unit: mmunit.Unit,
        periodicBounds: VectorQuantity | None = None,
        pbc: bool = False,
    ) -> AtomicFunction:
        """
        Create an :class:`AtomicFunction` from an :OpenMM:`Force`.

        Parameters
        ----------
        force
            The force to be converted
        unit
            The unit of measurement of the collective variable. It must be
            compatible with the MD unit system (mass in `daltons`, distance in
            `nanometers`, time in `picoseconds`, temperature in `kelvin`, energy in
            `kilojoules_per_mol`, angle in `radians`). If the collective variables
            does not have a unit, use `unit.dimensionless`.
        periodicBounds
            The lower and upper bounds of the collective variable if it is periodic,
            or ``None`` if it is not. This parameter is considered only if `force` is
            a custom :OpenMM:`Force`.
        pbc
            Whether to use periodic boundary conditions when computing atomic
            distances.

        Raises
        ------
        TypeError
            If the force is not convertible to an :class:`AtomicFunction`

        Examples
        --------
        >>> import cvpack
        >>> import numpy as np
        >>> import openmm
        >>> from openmm import unit
        >>> from openmm import app
        >>> from openmmtools import testsystems
        >>> model = testsystems.LysozymeImplicit()
        >>> residues = [r for r in model.topology.residues() if 59 <= r.index <= 79]
        >>> helix_content = cvpack.HelixTorsionContent(residues)
        >>> model.system.addForce(helix_content)
        6
        >>> num_atoms = model.system.getNumParticles()
        >>> mean_x = openmm.CustomExternalForce("x/num_atoms")
        >>> mean_x.addGlobalParameter("num_atoms", num_atoms)
        0
        >>> for i in range(num_atoms):
        ...     _ = mean_x.addParticle(i, [])
        >>> model.system.addForce(mean_x)
        7
        >>> forces = {force.getName(): force for force in model.system.getForces()}
        >>> copies = {
        ...     name: cvpack.AtomicFunction.fromOpenMMForce(
        ...         force, unit.kilojoules_per_mole
        ...     )
        ...     for name, force in forces.items()
        ...     if name in [
        ...         "HarmonicBondForce",
        ...         "HarmonicAngleForce",
        ...         "PeriodicTorsionForce",
        ...         "CustomExternalForce"
        ...     ]
        ... }
        >>> copies["HelixTorsionContent"] = cvpack.AtomicFunction.fromOpenMMForce(
        ...     helix_content, unit.dimensionless
        ... )
        >>> indices = {}
        >>> for index, (name, force) in enumerate(copies.items(), start=1):
        ...     _ = model.system.addForce(force)
        ...     force.setForceGroup(index)
        ...     indices[name] = index
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> integrator = openmm.VerletIntegrator(0)
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> for name in copies:
        ...    state = context.getState(getEnergy=True, groups={indices[name]})
        ...    value = state.getPotentialEnergy() / unit.kilojoules_per_mole
        ...    copy_value = copies[name].getValue(context)
        ...    print(f"{name}: original={value:.6f}, copy={copy_value}")
        HarmonicBondForce: original=2094.312..., copy=2094.312... kJ/mol
        HarmonicAngleForce: original=3239.795..., copy=3239.795... kJ/mol
        PeriodicTorsionForce: original=4226.05..., copy=4226.05... kJ/mol
        CustomExternalForce: original=5.02155..., copy=5.02155... kJ/mol
        HelixTorsionContent: original=17.4528..., copy=17.4528... dimensionless
        """
        if isinstance(
            force,
            (
                openmm.CustomAngleForce,
                openmm.CustomBondForce,
                openmm.CustomCompoundBondForce,
                openmm.CustomExternalForce,
                openmm.CustomTorsionForce,
            ),
        ):
            return cls._fromCustomForce(force, unit, periodicBounds, pbc)
        if isinstance(force, openmm.HarmonicBondForce):
            return cls._fromHarmonicBondForce(force, unit, pbc)
        if isinstance(force, openmm.HarmonicAngleForce):
            return cls._fromHarmonicAngleForce(force, unit, pbc)
        if isinstance(force, openmm.PeriodicTorsionForce):
            return cls._fromPeriodicTorsionForce(force, unit, pbc)
        raise TypeError(f"Force {force} is not convertible to an AtomicFunction")


AtomicFunction.registerTag("!cvpack.AtomicFunction")
