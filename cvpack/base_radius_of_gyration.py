"""
.. class:: BaseRadiusOfGyration
   :platform: Linux, MacOS, Windows
   :synopsis: Base class for the radius of gyration of an atom group

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm

from .collective_variable import CollectiveVariable


class BaseRadiusOfGyration(CollectiveVariable, openmm.CustomCentroidBondForce):
    """
    Abstract class for the radius of gyration of a group of `n` atoms.
    """

    def __init__(
        self,
        num_groups: int,
        expression: str,
        group: t.Sequence[int],
        pbc: bool = False,
        weighByMass: bool = False,
    ) -> None:
        super().__init__(num_groups, expression)
        for atom in group:
            self.addGroup([atom], [1])
        if weighByMass:
            self.addGroup(group)
        else:
            self.addGroup(group, [1] * len(group))
        self.setUsesPeriodicBoundaryConditions(pbc)
