"""
.. class:: Angle
   :platform: Linux, MacOS, Windows
   :synopsis: The angle formed by three atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import openmm
from openmm import unit as mmunit

from .collective_variable import CollectiveVariable


class Angle(CollectiveVariable, openmm.CustomAngleForce):
    r"""
    The angle formed by three atoms:

    .. math::

        \theta({\bf r}) =
            \mathrm{acos}\left(
                \frac{{\bf r}_{2,1} \cdot {\bf r}_{2,3} }
                       {\| {\bf r}_{2,1} \| \| {\bf r}_{2,3} \|}
            \right),

    where :math:`{\bf r}_{i,j} = {\bf r}_j - {\bf r}_i`.

    Parameters
    ----------
    atom1
        The index of the first atom.
    atom2
        The index of the second atom.
    atom3
        The index of the third atom.
    pbc
        Whether to use periodic boundary conditions.
    name
        The name of the collective variable.

    Example
    -------
    >>> import cvpack
    >>> import openmm
    >>> system = openmm.System()
    >>> [system.addParticle(1) for i in range(3)]
    [0, 1, 2]
    >>> angle = cvpack.Angle(0, 1, 2)
    >>> system.addForce(angle)
    0
    >>> integrator = openmm.VerletIntegrator(0)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> context = openmm.Context(system, integrator, platform)
    >>> positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    >>> context.setPositions([openmm.Vec3(*pos) for pos in positions])
    >>> angle.getValue(context)
    1.570796... rad

    """

    def __init__(
        self,
        atom1: int,
        atom2: int,
        atom3: int,
        pbc: bool = False,
        name: str = "angle",
    ) -> None:
        super().__init__("theta")
        self.addAngle(atom1, atom2, atom3, [])
        self.setUsesPeriodicBoundaryConditions(pbc)
        self._registerCV(
            name, mmunit.radians, atom1=atom1, atom2=atom2, atom3=atom3, pbc=pbc
        )


Angle.registerTag("!cvpack.Angle")
