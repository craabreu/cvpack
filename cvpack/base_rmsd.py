"""
.. class:: BaseRMSD
   :platform: Linux, MacOS, Windows
   :synopsis: Root-mean-square deviation of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np

from .collective_variable import CollectiveVariable
from .units import MatrixQuantity, VectorQuantity, value_in_md_units


class BaseRMSD(CollectiveVariable):
    r"""
    A base class for root-mean-square deviation (RMSD) collective variables.
    """

    def _getDefinedCoords(
        self,
        referencePositions: MatrixQuantity | dict[int, VectorQuantity],
        group: list[int],
    ) -> None:
        if isinstance(referencePositions, dict):
            positions = {
                atom: value_in_md_units(coords)
                for atom, coords in referencePositions.items()
            }
        else:
            positions = value_in_md_units(referencePositions)
        return {atom: [float(x) for x in positions[atom]] for atom in group}

    def _getAllCoords(
        self, definedCoords: dict[int, list[float]], numAtoms: int
    ) -> np.ndarray:
        all_coords = np.zeros((numAtoms, 3))
        for atom, coords in definedCoords.items():
            all_coords[atom] = coords
        return all_coords
