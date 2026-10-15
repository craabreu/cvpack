"""
.. class:: RMSD
   :platform: Linux, MacOS, Windows
   :synopsis: Root-mean-square deviation of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm
from openmm import unit as mmunit

from .base_rmsd import BaseRMSD
from .units import MatrixQuantity, VectorQuantity


class RMSD(BaseRMSD, openmm.RMSDForce):
    r"""
    The root-mean-square deviation (RMSD) between the current and reference
    coordinates of a group of `n` atoms:

    .. math::

        d_{\rm rms}({\bf r}) = \sqrt{
            \frac{1}{n} \min_{
                \bf q \in \mathbb{R}^4 \atop \|{\bf q}\| = 1
            } \sum_{i=1}^n \left\|
                {\bf A}({\bf q}) \hat{\bf r}_i - \hat{\bf r}_i^{\rm ref}
            \right\|^2
        }

    where :math:`\hat{\bf r}_i` is the position of the :math:`i`-th atom in the group
    relative to the group's center of geometry (centroid), the superscript
    :math:`\rm ref` denotes the reference structure, :math:`{\bf q}` is a unit
    quaternion, and :math:`{\bf A}({\bf q})` is the rotation matrix corresponding to
    :math:`{\bf q}`.

    .. warning::

        Periodic boundary conditions are `not supported
        <https://github.com/openmm/openmm/issues/2913>`_. This is not a problem if all
        atoms in the group belong to the same molecule. If they belong to distinct
        molecules, it is possible to circumvent the issue by calling the method
        :func:`getNullBondForce` and adding the resulting force to the system.

    Parameters
    ----------
    referencePositions
        The reference coordinates, which can be either a coordinate matrix or a mapping
        from atom indices to coordinate vectors. It must contain all atoms in ``group``,
        and does not need to contain all atoms in the system. See ``numAtoms`` below.
    group
        A sequence of atom indices.
    numAtoms
        The total number of atoms in the system, including those that are not in
        ``group``. This argument is necessary only if ``referencePositions`` does not
        contain all atoms in the system.
    name
        The name of the collective variable.

    Raises
    ------
    ValueError
        If referencePositions is a dict and numAtoms is not given

    Example
    -------
    >>> import cvpack
    >>> import openmm
    >>> from openmm import app, unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideImplicit()
    >>> num_atoms = model.topology.getNumAtoms()
    >>> group = list(range(num_atoms))
    >>> rmsd = cvpack.RMSD(model.positions, group, num_atoms)
    >>> rmsd.addToSystem(model.system)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> integrator = openmm.VerletIntegrator(2*unit.femtoseconds)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> integrator.step(1000)
    >>> rmsd.getValue(context)
    0.123138... nm

    """

    def __init__(
        self,
        referencePositions: MatrixQuantity | dict[int, VectorQuantity],
        group: t.Iterable[int],
        numAtoms: int | None = None,
        name: str = "rmsd",
    ) -> None:
        group = list(map(int, group))
        if not group:
            # OpenMM's RMSDForce would interpret an empty group as all atoms
            raise ValueError("The group of atoms must not be empty")
        if numAtoms is None and isinstance(referencePositions, dict):
            raise ValueError("numAtoms is required when referencePositions is a dict")
        num_atoms = numAtoms or len(referencePositions)
        defined_coords = self._getDefinedCoords(referencePositions, group)
        all_coords = self._getAllCoords(defined_coords, num_atoms)
        super().__init__(all_coords, group)
        self._registerCV(
            name,
            mmunit.nanometers,
            referencePositions=defined_coords,
            group=group,
            numAtoms=num_atoms,
        )

    def getNullBondForce(self) -> openmm.HarmonicBondForce:
        """
        Get a null bond force that creates a connected graph with all the atoms in the
        group.

        Returns
        -------
        openmm.HarmonicBondForce
            The null bond force.

        Example
        -------
        >>> import cvpack
        >>> import openmm
        >>> from openmm import app, unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.WaterBox(
        ...     box_edge=10*unit.angstroms,
        ...     cutoff=5*unit.angstroms,
        ... )
        >>> group = [
        ...     atom.index
        ...     for atom in model.topology.atoms()
        ...     if atom.residue.index < 3
        ... ]
        >>> rmsd = cvpack.RMSD(
        ...     model.positions, group, model.topology.getNumAtoms()
        ... )
        >>> rmsd.addToSystem(model.system)
        >>> _ = model.system.addForce(rmsd.getNullBondForce())
        >>> integrator = openmm.VerletIntegrator(2*unit.femtoseconds)
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> integrator.step(100)
        >>> rmsd.getValue(context)
        0.10436... nm

        """
        force = openmm.HarmonicBondForce()
        group = self._args["group"]
        for i, j in zip(group[:-1], group[1:]):
            force.addBond(i, j, 0.0, 0.0)
        return force


RMSD.registerTag("!cvpack.RMSD")
