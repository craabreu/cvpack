"""
.. class:: TorsionSimilarity
   :platform: Linux, MacOS, Windows
   :synopsis: The degree of similarity between pairs of torsion angles.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import numpy as np
import openmm
from numpy.typing import ArrayLike
from openmm import unit as mmunit

from .collective_variable import CollectiveVariable


class TorsionSimilarity(CollectiveVariable, openmm.CustomCompoundBondForce):
    r"""
    The degree of similarity between `n` pairs of torsion angles:

    .. math::

        s({\bf r}) = \frac{n}{2} + \frac{1}{2} \sum_{i=1}^n \cos\Big(
            \phi^{\rm 1st}_i({\bf r}) - \phi^{\rm 2nd}_i({\bf r})
        \Big)

    where :math:`\phi^{\rm kth}_i` is the torsion angle at position :math:`i` in the
    :math:`k`-th list.

    .. note::

        In `PLUMED <https://www.plumed.org/doc-v2.8/user-doc/html/_colvar.html>`_, this
        collective variable is called ``DIHCOR``.

    Parameters
    ----------
    firstList
        A list of :math:`n` tuples of four atom indices defining the first torsion
        angle in each pair.
    secondList
        A list of :math:`n` tuples of four atom indices defining the second torsion
        angle in each pair.
    pbc
        Whether to use periodic boundary conditions in distance calculations.
    name
        The name of the collective variable.

    Raises
    ------
    ValueError
        If firstList and secondList have different lengths

    Example
    -------
    >>> import cvpack
    >>> import mdtraj
    >>> import openmm
    >>> import openmm.unit as mmunit
    >>> from openmmtools import testsystems
    >>> model = testsystems.LysozymeImplicit()
    >>> traj = mdtraj.Trajectory(
    ...     model.positions, mdtraj.Topology.from_openmm(model.topology)
    ... )
    >>> phi_atoms, _ = mdtraj.compute_phi(traj)
    >>> valid_atoms = traj.top.select("resid 59 to 79 and backbone")
    >>> phi_atoms = [
    ...     phi
    ...     for phi in phi_atoms
    ...     if all(atom in valid_atoms for atom in phi)
    ... ]
    >>> torsion_similarity = cvpack.TorsionSimilarity(
    ...     phi_atoms[1:], phi_atoms[:-1]
    ... )
    >>> torsion_similarity.addToSystem(model.system)
    >>> integrator = openmm.VerletIntegrator(0)
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> torsion_similarity.getValue(context)
    18.659... dimensionless
    """

    def __init__(
        self,
        firstList: ArrayLike,
        secondList: ArrayLike,
        pbc: bool = False,
        name: str = "torsion_similarity",
    ) -> None:
        firstList = [list(map(int, first)) for first in firstList]
        secondList = [list(map(int, second)) for second in secondList]
        if len(firstList) != len(secondList):
            raise ValueError("firstList and secondList must have the same length")
        function = f"0.5*(1 + cos(min(delta, {2 * np.pi} - delta)))"
        definition = "delta = dihedral(p1, p2, p3, p4) - dihedral(p5, p6, p7, p8)"
        super().__init__(8, f"{function}; {definition}")
        for first, second in zip(firstList, secondList):
            self.addBond([*first, *second], [])
        self.setUsesPeriodicBoundaryConditions(pbc)
        self._registerCV(
            name,
            mmunit.dimensionless,
            firstList=firstList,
            secondList=secondList,
            pbc=pbc,
        )


TorsionSimilarity.registerTag("!cvpack.TorsionSimilarity")
