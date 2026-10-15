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

    Parameters
    ----------
    squared
        Whether the collective variable is the squared radius of gyration.
    group
        The indices of the atoms in the group.
    pbc
        Deprecated. It has no effect.
    weighByMass
        No longer supported. It must be False.
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
        if weighByMass:
            raise ValueError(
                "weighByMass=True is no longer supported. It measured distances from "
                "the center of mass but averaged them without mass weights, which is "
                "not the usual mass-weighted radius of gyration."
            )
        if pbc:
            warnings.warn(
                "The pbc argument of the radius of gyration is deprecated and has no "
                "effect. The group of atoms must not be split across periodic images.",
                DeprecationWarning,
                stacklevel=3,
            )
        super().__init__("rg^2" if squared else "rg")
        self.addCollectiveVariable("rg", openmm.RGForce(group))
