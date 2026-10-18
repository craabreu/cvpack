"""
.. class:: CompositeRMSD
   :platform: Linux, MacOS, Windows
   :synopsis: Deviation of multiple corotating bodies from their reference structures

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import itertools
import typing as t

import numpy as np
import openmm
from openmm import unit as mmunit

from .base_rmsd import BaseRMSD
from .units import MatrixQuantity, VectorQuantity


class CompositeRMSD(BaseRMSD, openmm.CustomCVForce):
    r"""
    The composite root-mean-square deviation (RMSD) between the current and reference
    coordinates of :math:`m` groups of atoms:

    .. math::

        d_{\rm crms}({\bf r}) = \sqrt{
            \frac{1}{n} \min_{
                \bf q \in \mathbb{R}^4 \atop \|{\bf q}\| = 1
            } \sum_{j=1}^m \sum_{i \in {\bf g}_j} \left\|
                {\bf A}({\bf q})\left({\bf r}_i - {\bf c}_j\right) -
                    {\bf r}_i^{\rm ref} + {\bf c}_j^{\rm ref}
            \right\|^2
        }

    where each group :math:`{\bf g}_j` is a set of :math:`n_j` atom indices,
    :math:`n = \sum_{j=1}^m n_j` is the total number of atoms in these groups,
    :math:`{\bf A}(\bf q)` is the rotation matrix corresponding to a unit quaternion
    :math:`{\bf q}`, :math:`{\bf r}_i` and :math:`{\bf r}_i^{\rm ref}` are the
    positions of atom :math:`i` in the current and reference structures, respectively,
    :math:`{\bf c}_j` is the centroid of group :math:`j`, given by

    .. math::

        {\bf c}_j = \frac{1}{n_j} \sum_{i \in {\bf g}_j} {\bf r}_i

    and :math:`{\bf c}_j^{\rm ref}` is the centroid of the reference structure for
    group :math:`j`, defined analogously to :math:`{\bf c}_j`.

    .. note::

        The computation is done by OpenMM's :OpenMM:`RMSDForce`, using reference
        positions shifted so that every group's centroid lies at the origin. With these
        references, the optimal rotation is the same as above, and the composite RMSD
        follows from

        .. math::

            d_{\rm crms}^2 = d_{\rm rms}^2 - \frac{1}{n^2} \sum_{j=1}^{m-1}
                \sum_{k=j+1}^m n_j n_k \left\|{\bf c}_j - {\bf c}_k\right\|^2

        where :math:`d_{\rm rms}` is the RMSD computed by :OpenMM:`RMSDForce` and the
        sum is computed by a :OpenMM:`CustomCentroidBondForce`.

    .. warning::

        Periodic boundary conditions are not supported. Each group of atoms must not be
        split across periodic images.

    Parameters
    ----------
    referencePositions
        The reference coordinates, which can be either a coordinate matrix or a mapping
        from atom indices to coordinate vectors. It must contain all atoms in
        ``groups``, and does not need to contain all atoms in the system. See
        ``numAtoms`` below.
    groups
        A sequence of disjoint atom groups. Each group is a sequence of atom indices.
    numAtoms
        The total number of atoms in the system, including those that are not in
        ``groups``. This argument is necessary only if ``referencePositions`` does not
        contain all atoms in the system.
    name
        The name of the collective variable.

    Raises
    ------
    ValueError
        If ``groups`` is not a sequence of disjoint, nonempty atom groups.
    ValueError
        If referencePositions is a dict and numAtoms is not given, or if numAtoms
        is not positive.

    Example
    -------
    >>> import cvpack
    >>> import openmm as mm
    >>> from openmmtools import testsystems
    >>> from openmm import unit
    >>> model = testsystems.HostGuestVacuum()
    >>> host_atoms, guest_atoms = (
    ...     [a.index for a in r.atoms()]
    ...     for r in model.topology.residues()
    ... )
    >>> composite_rmsd = cvpack.CompositeRMSD(
    ...     model.positions,
    ...     [host_atoms, guest_atoms],
    ... )
    >>> composite_rmsd.addToSystem(model.system)
    >>> context = mm.Context(
    ...     model.system,
    ...     mm.VerletIntegrator(1.0 * unit.femtoseconds),
    ...     mm.Platform.getPlatformByName('Reference'),
    ... )
    >>> context.setPositions(model.positions)
    >>> round(composite_rmsd.getValue(context) / unit.nanometers, 6)
    0.0
    >>> model.positions[guest_atoms] += 1.0 * unit.nanometers
    >>> context.setPositions(model.positions)
    >>> round(composite_rmsd.getValue(context) / unit.nanometers, 6)
    0.0
    """

    def __init__(
        self,
        referencePositions: MatrixQuantity | dict[int, VectorQuantity],
        groups: t.Iterable[t.Iterable[int]],
        numAtoms: int | None = None,
        name: str = "composite_rmsd",
    ) -> None:
        num_atoms = self._getNumAtoms(referencePositions, numAtoms)
        groups = [[int(atom) for atom in group] for group in groups]
        if not all(groups):
            # OpenMM's RMSDForce would interpret an empty group as all atoms
            raise ValueError("The groups of atoms must not be empty")
        atoms = sum(groups, [])
        if len(set(atoms)) != len(atoms):
            raise ValueError("The groups of atoms must be disjoint")
        defined_coords = self._getDefinedCoords(referencePositions, atoms)
        all_coords = self._getAllCoords(defined_coords, num_atoms)
        for group in groups:
            all_coords[group] -= all_coords[group].mean(axis=0)
        sizes = np.array([len(group) for group in groups])
        weights = np.outer(sizes, sizes) / sizes.sum() ** 2
        spread = openmm.CustomCentroidBondForce(2, "w*distance(g1, g2)^2")
        spread.addPerBondParameter("w")
        for group in groups:
            spread.addGroup(group, [1.0] * len(group))
        for j, k in itertools.combinations(range(len(groups)), 2):
            spread.addBond([j, k], [weights[j, k]])
        # Roundoff can make the squared value slightly negative, and the select call
        # avoids an infinite derivative of sqrt(d2) at d2 = 0
        super().__init__("select(d2, sqrt(d2), 0); d2 = max(0, rmsd^2 - spread)")
        self.addCollectiveVariable("rmsd", openmm.RMSDForce(all_coords, atoms))
        self.addCollectiveVariable("spread", spread)
        self._registerCV(
            name,
            mmunit.nanometers,
            referencePositions=defined_coords,
            groups=groups,
            numAtoms=num_atoms,
        )


CompositeRMSD.registerTag("!cvpack.CompositeRMSD")
