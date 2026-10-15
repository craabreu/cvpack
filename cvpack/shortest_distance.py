"""
.. class:: ShortestDistance
   :platform: Linux, MacOS, Windows
   :synopsis: The number of contacts between two atom groups

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import numpy as np
import openmm
from openmm import unit as mmunit

from .collective_variable import CollectiveVariable
from .units import Quantity


class ShortestDistance(CollectiveVariable, openmm.CustomCVForce):
    r"""
    A smooth approximation for the shortest distance between two atom groups:

    .. math::
        r_{\rm min}({\bf r}) = \frac{S_{rw}({\bf r})}{S_w({\bf r})}

    with

    .. math::

        S_{rw}({\bf r}) = r_c e^{-\beta} +
            \sum_{i \in {\bf g}_1} \sum_{j \in {\bf g}_2 \atop r_{ij} < r_c}
                r_{ij} e^{-\beta \frac{r_{ij}}{r_c}}

    and

    .. math::

        S_w({\bf r}) = e^{-\beta} +
            \sum_{i \in {\bf g}_1} \sum_{j \in {\bf g}_2 \atop r_{ij} < r_c}
                r_{ij} e^{-\beta \frac{r_{ij}}{r_c}}

    where :math:`r_{ij} = \|{\bf r}_j - {\bf r}_i\|` is the distance between atoms
    :math:`i` and :math:`j`, :math:`{\bf g}_1` and :math:`{\bf g}_2` are the sets of
    indices of the atoms in the first and second groups, respectively, :math:`r_c` is
    the cutoff distance, and :math:`\beta` is a parameter that controls the degree of
    approximation.

    The larger the value of :math:`\beta`, the closer the approximation to the true
    shortest distance. However, an excessively large value may lead to numerical
    instability.

    The terms outside the summations guarantee that shortest distance is well-defined
    even when there are no atom pairs within the cutoff distance. In this case, the
    collective variable evaluates to the cutoff distance.

    .. note::

        Atoms are allowed to be in both groups. In this case, terms for which
        :math:`i = j` are ignored.

    Parameters
    ----------
    group1
        The indices of the atoms in the first group.
    group2
        The indices of the atoms in the second group.
    numAtoms
        The total number of atoms in the system, including those that are not in any
        of the groups. This is necessary for the correct initialization of the
        collective variable.
    beta
        The parameter that controls the degree of approximation.
    cutoffDistance
        The cutoff distance for evaluating atom pairs.
    pbc
        Whether to consider periodic boundary conditions in distance calculations.
    name
        The name of the collective variable.

    Example
    -------
    >>> import cvpack
    >>> import numpy as np
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.HostGuestVacuum()
    >>> group1, group2 = [], []
    >>> for residue in model.topology.residues():
    ...     group = group1 if residue.name == "B2" else group2
    ...     group.extend(atom.index for atom in residue.atoms())
    >>> coords = model.positions.value_in_unit(mmunit.nanometers)
    >>> print(np.linalg.norm(
    ...     coords[group1, None, :] - coords[None, group2, :],
    ...     axis=-1,
    ... ).min())
    0.1757...
    >>> num_atoms = model.system.getNumParticles()
    >>> min_dist = cvpack.ShortestDistance(group1, group2, num_atoms)
    >>> min_dist.addToSystem(model.system)
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> integrator = openmm.VerletIntegrator(1.0 * mmunit.femtoseconds)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> min_dist.getValue(context)
    0.1785... nm
    """

    def __init__(
        self,
        group1: t.Sequence[int],
        group2: t.Sequence[int],
        numAtoms: int,
        beta: float = 100,
        cutoffDistance: mmunit.Quantity = Quantity(1 * mmunit.nanometers),
        pbc: bool = True,
        name: str = "shortest_distance",
    ) -> None:
        rc = cutoffDistance
        if mmunit.is_quantity(rc):
            rc = rc.value_in_unit(mmunit.nanometers)
        weight = f"exp(-{beta/rc}*r)"
        forces = {
            "wrsum": openmm.CustomNonbondedForce(f"{weight}*r"),
            "wsum": openmm.CustomNonbondedForce(weight),
        }
        super().__init__(f"({rc*np.exp(-beta)}+wrsum)/({np.exp(-beta)}+wsum)")
        for cv, force in forces.items():
            force.setNonbondedMethod(
                force.CutoffPeriodic if pbc else force.CutoffNonPeriodic
            )
            force.setCutoffDistance(cutoffDistance)
            force.setUseSwitchingFunction(False)
            force.setUseLongRangeCorrection(False)
            for _ in range(numAtoms):
                force.addParticle([])
            force.addInteractionGroup(group1, group2)
            self.addCollectiveVariable(cv, force)
        self._registerCV(
            name,
            mmunit.nanometers,
            group1=group1,
            group2=group2,
            numAtoms=numAtoms,
            beta=beta,
            cutoffDistance=rc,
            pbc=pbc,
        )


ShortestDistance.registerTag("!cvpack.ShortestDistance")
