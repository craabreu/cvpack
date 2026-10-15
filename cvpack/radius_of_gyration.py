"""
.. class:: RadiusOfGyration
   :platform: Linux, MacOS, Windows
   :synopsis: The radius of gyration of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm
from openmm import unit as mmunit

from .collective_variable import CollectiveVariable
from .radius_of_gyration_sq import RadiusOfGyrationSq


class RadiusOfGyration(CollectiveVariable, openmm.CustomCVForce):
    r"""
    The radius of gyration of a group of :math:`n` atoms:

    .. math::

        r_g({\bf r}) = \sqrt{ \frac{1}{n} \sum_{i=1}^n \left\|
            {\bf r}_i - {\bf r}_c({\bf r})
        \right\|^2 }.

    where :math:`{\bf r}_c({\bf r})` is the geometric center of the group:

    .. math::

        {\bf r}_c({\bf r}) = \frac{1}{n} \sum_{j=1}^n {\bf r}_j

    Optionally, the radius of gyration can be computed with respect to the center of
    mass of the group. In this case, the geometric center is replaced by:

    .. math::

        {\bf r}_m({\bf r}) = \frac{1}{M} \sum_{i=1}^n m_i {\bf r}_i

    where :math:`M = \sum_{i=1}^n m_i` is the total mass of the group.

    .. note::

        Without mass weighting and periodic boundary conditions, the computation is
        done by OpenMM's :OpenMM:`RGForce`. Otherwise, this collective variable is the
        square root of a :class:`RadiusOfGyrationSq`.

    Parameters
    ----------
    group
        The indices of the atoms in the group.
    pbc
        Whether to use periodic boundary conditions.
    weighByMass
        Whether to use the center of mass of the group instead of its geometric center.
    name
        The name of the collective variable.

    Example
    -------
    >>> import cvpack
    >>> import openmm
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> num_atoms = model.system.getNumParticles()
    >>> radius_of_gyration = cvpack.RadiusOfGyration(list(range(num_atoms)))
    >>> radius_of_gyration.addToSystem(model.system)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> integrator = openmm.VerletIntegrator(0)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> radius_of_gyration.getValue(context)
    0.2951... nm

    """

    def __init__(
        self,
        group: t.Iterable[int],
        pbc: bool = False,
        weighByMass: bool = False,
        name: str = "radius_of_gyration",
    ) -> None:
        group = list(group)
        if not group:
            # OpenMM's RGForce would interpret an empty group as all atoms
            raise ValueError("The group of atoms must not be empty")
        if pbc or weighByMass:
            super().__init__("sqrt(rgsq)")
            self.addCollectiveVariable(
                "rgsq", RadiusOfGyrationSq(group, pbc=pbc, weighByMass=weighByMass)
            )
        else:
            super().__init__("rg")
            self.addCollectiveVariable("rg", openmm.RGForce(group))
        self._registerCV(
            name, mmunit.nanometers, group=group, pbc=pbc, weighByMass=weighByMass
        )


RadiusOfGyration.registerTag("!cvpack.RadiusOfGyration")
