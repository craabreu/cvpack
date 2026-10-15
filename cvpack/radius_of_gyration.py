"""
.. class:: RadiusOfGyration
   :platform: Linux, MacOS, Windows
   :synopsis: The radius of gyration of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

from openmm import unit as mmunit

from .base_radius_of_gyration import BaseRadiusOfGyration


class RadiusOfGyration(BaseRadiusOfGyration):
    r"""
    The radius of gyration of a group of :math:`n` atoms:

    .. math::

        r_g({\bf r}) = \sqrt{ \frac{1}{n} \sum_{i=1}^n \left\|
            {\bf r}_i - {\bf r}_c({\bf r})
        \right\|^2 }.

    where :math:`{\bf r}_c({\bf r})` is the geometric center of the group:

    .. math::

        {\bf r}_c({\bf r}) = \frac{1}{n} \sum_{j=1}^n {\bf r}_j

    .. note::

        The computation is done by OpenMM's :OpenMM:`RGForce`.

    Parameters
    ----------
    group
        The indices of the atoms in the group.
    pbc
        Deprecated, with no effect. OpenMM does not wrap positions inside a
        :OpenMM:`Context`, so the group of atoms must not be split across periodic
        images, as was already required for its centroid.
    weighByMass
        No longer supported. It must be False, because ``True`` measured distances from
        the center of mass without weighting their average, which is not the usual
        mass-weighted radius of gyration.
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
        super().__init__(False, group, pbc, weighByMass)
        self._registerCV(
            name, mmunit.nanometers, group=group, pbc=pbc, weighByMass=weighByMass
        )


RadiusOfGyration.registerTag("!cvpack.RadiusOfGyration")
