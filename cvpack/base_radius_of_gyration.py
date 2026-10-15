"""
.. class:: BaseRadiusOfGyration
   :platform: Linux, MacOS, Windows
   :synopsis: Base class for the radius of gyration of an atom group

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
import warnings

import openmm

from .collective_variable import CollectiveVariable


class BaseRadiusOfGyration(CollectiveVariable, openmm.CustomCVForce):
    """
    Abstract class for the radius of gyration of a group of `n` atoms, computed by
    OpenMM's RGForce.

    With mass weighting, the mean squared distance to the center of mass follows from
    the parallel-axis theorem: it is the squared radius of gyration about the geometric
    center plus the squared distance between the two centers.

    Parameters
    ----------
    squared
        Whether the collective variable is the squared radius of gyration.
    group
        The indices of the atoms in the group.
    pbc
        Deprecated. It has no effect.
    weighByMass
        Whether to use the center of mass of the group instead of its geometric center.
    """

    def __init__(
        self,
        squared: bool,
        group: t.Sequence[int],
        pbc: bool,
        weighByMass: bool,
    ) -> None:
        if not group:
            # OpenMM's RGForce would interpret an empty group as all atoms
            raise ValueError("The group of atoms must not be empty")
        if len(set(group)) != len(group):
            raise ValueError("The group of atoms must not have repeated indices")
        if pbc:
            warnings.warn(
                "The pbc argument of the radius of gyration is deprecated and has no "
                "effect. The group of atoms must not be split across periodic images.",
                DeprecationWarning,
                stacklevel=3,
            )
        if weighByMass:
            rgsq = "rg^2 + shiftsq"
            super().__init__(rgsq if squared else f"sqrt({rgsq})")
            # A squared distance has no singular derivative when the centers coincide
            shift = openmm.CustomCentroidBondForce(2, "distance(g1, g2)^2")
            shift.addGroup(group, [1.0] * len(group))
            shift.addGroup(group)
            shift.addBond([0, 1])
        else:
            super().__init__("rg^2" if squared else "rg")
        self.addCollectiveVariable("rg", openmm.RGForce(group))
        if weighByMass:
            self.addCollectiveVariable("shiftsq", shift)
