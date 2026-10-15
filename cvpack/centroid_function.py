"""
.. class:: CentroidFunction
   :platform: Linux, MacOS, Windows
   :synopsis: A generic function of the centroids of groups of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import numpy as np
import openmm
from numpy.typing import ArrayLike
from openmm import unit as mmunit

from .base_custom_function import BaseCustomFunction
from .units import ScalarQuantity, VectorQuantity


class CentroidFunction(BaseCustomFunction, openmm.CustomCentroidBondForce):
    r"""
    A generic function of the centroids of :math:`m \times n` atoms groups split
    into `m` collections of `n` groups:

    .. math::

        f({\bf r}) = \sum_{i=1}^m F\Big(
            {\bf g}^i_1({\bf r}),
            {\bf g}^i_2({\bf r}),
            \dots,
            {\bf g}^i_n({\bf r})
        \Big)

    where :math:`F` is a user-defined function and :math:`{\bf g}^i_1({\bf r})` is the
    centroid of the :math:`j`-th group of atoms of the :math:`i`-th collection of
    groups.

    The centroid of a group of atoms is defined as:

    .. math::

        {\bf g}_j({\bf r}) = \frac{1}{N_j} \sum_{k=1}^{N_j} {\bf r}_{k,j}

    where :math:`N_j` is the number of atoms in group :math:`j` and
    :math:`{\bf r}_{k,j}` is the coordinate of the :math:`k`-th atom of group
    :math:`j`. Optionally, the centroid can be weighted by the mass of each atom
    in the group. In this case, it is redefined as:

    .. math::

        {\bf g}_j({\bf r}) = \frac{1}{M_j} \sum_{k=1}^{N_j} m_{k,j} {\bf r}_{k,j}

    where :math:`M_j` is the total mass of atom group :math:`j` and :math:`m_{k,j}` is
    the mass of the :math:`k`-th atom in group :math:`j`.

    The function :math:`F` is defined as a string and can be any expression supported
    by :OpenMM:`CustomCompoundBondForce`. If the expression contains named parameters,
    the value of each parameter can be passed in one of three ways:

    #. By a semicolon-separated definition in the function string, such as described
       in the :OpenMM:`CustomCompoundBondForce` documentation. In this case, the
       parameter value will be the same for all collections of atom groups.

    #. By a 1D array or list of length :math:`m` passed as a keyword argument to
       the :class:`AtomicFunction` constructor. In this case, each value will be
       assigned to a different collection of atom groups.

    #. By a scalar passed as a keyword argument to the :class:`AtomicFunction`
       constructor. In this case, the parameter will apply to all collections of atom
       groups and will become available for on-the-fly modification during a simulation
       via the ``setParameter`` method of an :OpenMM:`Context` object. **Warning**:
       other collective variables or :OpenMM:`Force` objects in the same system will
       share the same values of equal-named parameters.

    Parameters
    ----------
    function
        The function to be evaluated. It must be a valid
        :OpenMM:`CustomCentroidBondForce` expression.
    unit
        The unit of measurement of the collective variable. It must be compatible
        with the MD unit system (mass in `daltons`, distance in `nanometers`, time
        in `picoseconds`, temperature in `kelvin`, energy in `kilojoules_per_mol`,
        angle in `radians`). If the collective variables does not have a unit, use
        `dimensionless`.
    groups
        The groups of atoms to be used in the function. Each group must be specified
        as a list of atom indices with arbitrary length.
    collections
        The indices of the groups in each collection, passed as a 2D array-like object
        of shape `(m, n)`, where `m` is the number of collections and `n` is the number
        groups per collection. If a 1D object is passed, it is assumed that `m` is 1 and
        `n` is the length of the object.
    periodicBounds
        The periodic bounds of the collective variable if it is periodic, or `None` if
        it is not.
    pbc
        Whether to use periodic boundary conditions. This should be ``True`` for
        periodic systems.
    weighByMass
        Whether to define the centroid as the center of mass of the group instead of
        the geometric center.
    name
        The name of the collective variable.

    Keyword Args
    ------------
    **parameters
        The named parameters of the function. Each parameter can be a 1D array-like
        object or a scalar. In the former case, the array must have length :math:`m`
        and each entry will be assigned to a different collection of atom groups. In
        the latter case, it will define a global :OpenMM:`Context` parameter.

    Raises
    ------
    ValueError
        If the collections are not specified as a 1D or 2D array-like object.
    ValueError
        If group indices are out of bounds.
    ValueError
        If the unit of the collective variable is not compatible with the MD unit
        system.

    Example
    -------
    >>> import cvpack
    >>> import numpy as np
    >>> import openmm
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.LysozymeImplicit()
    >>> residues = list(model.topology.residues())
    >>> atoms = [[a.index for a in r.atoms()] for r in residues]

    Compute the residue coordination between two helices:

    >>> res_coord = cvpack.ResidueCoordination(
    ...     residues[115:124],
    ...     residues[126:135],
    ...     stepFunction="step(1-x)",
    ... )
    >>> res_coord.addToSystem(model.system)
    >>> integrator = openmm.VerletIntegrator(0)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> res_coord.getValue(context)
    28.0 dimensionless

    Recompute the residue coordination using the centroid function:

    >>> groups = [atoms[115:124], atoms[126:135]]
    >>> colvar = cvpack.CentroidFunction(
    ...    "step(1 - distance(g1, g2))",
    ...    unit.dimensionless,
    ...    atoms[115:124] + atoms[126:135],
    ...    [[i, j] for i in range(9) for j in range(9, 18)],
    ... )
    >>> colvar.addToSystem(model.system)
    >>> context.reinitialize(preserveState=True)
    >>> colvar.getValue(context)
    28.0 dimensionless
    """

    def __init__(
        self,
        function: str,
        unit: mmunit.Unit,
        groups: t.Iterable[t.Iterable[int]],
        collections: ArrayLike | None = None,
        periodicBounds: VectorQuantity | None = None,
        pbc: bool = False,
        weighByMass: bool = True,
        name: str = "centroid_function",
        **parameters: ScalarQuantity | VectorQuantity,
    ) -> None:
        groups = [[int(atom) for atom in group] for group in groups]
        num_groups = len(groups)
        collections = np.atleast_2d(
            np.arange(num_groups) if collections is None else collections
        )
        num_collections, groups_per_collection, *others = collections.shape
        if others:
            raise ValueError("Array `collections` cannot have more than 2 dimensions")
        if np.any(collections < 0) or np.any(collections >= num_groups):
            raise ValueError("Group index out of bounds")
        super().__init__(groups_per_collection, function)
        for group in groups:
            self.addGroup(group, *([] if weighByMass else [[1] * len(group)]))
        overalls, perbonds = self._extractParameters(num_collections, **parameters)
        self._addParameters(overalls, perbonds, collections, pbc, unit)
        collections = [[int(atom) for atom in collection] for collection in collections]
        self._registerCV(
            name,
            unit,
            function=function,
            unit=unit,
            groups=groups,
            collections=collections,
            periodicBounds=periodicBounds,
            pbc=pbc,
            weighByMass=weighByMass,
            **overalls,
            **perbonds,
        )
        if periodicBounds is not None:
            self._registerPeriodicBounds(*periodicBounds)


CentroidFunction.registerTag("!cvpack.CentroidFunction")
