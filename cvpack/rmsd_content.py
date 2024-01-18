"""
.. class:: RMSDContent
   :platform: Linux, MacOS, Windows
   :synopsis: Secondary-structure RMSD content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from importlib import resources

import numpy as np
import openmm
from openmm import app as mmapp

from .cvpack import AbstractCollectiveVariable
from .rmsd import RMSD


class RMSDContent(openmm.CustomCVForce, AbstractCollectiveVariable):
    """
    Abstract class for secondary-structure RMSD content of a sequence of `n` residues.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        residue_blocks: t.List[int],
        ideal_positions: t.List[openmm.Vec3],
        residues: t.List[mmapp.topology.Residue],
        numAtoms: int,
        thresholdRMSD: float,
        stepFunction: str = "(1+x^4)/(1+x^4+x^8)",
        normalize: bool = False,
    ):
        num_residue_blocks = self._num_residue_blocks = len(residue_blocks)
        if not 1 <= num_residue_blocks <= 1024:
            raise ValueError(
                f"{len(residues)} residues yield {num_residue_blocks} blocks, "
                "which is not between 1 and 1024"
            )
        residue_atoms = list(map(self._getAtomList, residues))
        block_atoms = [
            sum([residue_atoms[index] for index in block], [])
            for block in residue_blocks
        ]

        def expression(start):
            summands = []
            definitions = []
            for i in range(start, min(start + 32, num_residue_blocks)):
                summands.append(stepFunction.replace("x", f"x{i}"))
                definitions.append(f"x{i}=rmsd{i}/{thresholdRMSD}")
            return ";".join(["+".join(summands)] + definitions)

        if num_residue_blocks <= 32:
            summation = expression(0)
            force = self
        else:
            summation = "+".join(
                f"chunk{i}" for i in range((num_residue_blocks + 31) // 32)
            )
        super().__init__(
            f"({summation})/{num_residue_blocks}" if normalize else summation
        )
        for index in range(num_residue_blocks):
            if num_residue_blocks > 32 and index % 32 == 0:
                force = openmm.CustomCVForce(expression(index))
                self.addCollectiveVariable(f"chunk{index//32}", force)
            force.addCollectiveVariable(
                f"rmsd{index}", RMSD(ideal_positions, block_atoms[index], numAtoms)
            )

    @classmethod
    def _loadPositions(cls, filename: str) -> t.List[openmm.Vec3]:
        positions = 0.1 * np.loadtxt(
            str(resources.files("cvpack").joinpath("data").joinpath(filename)),
            delimiter=",",
        )
        return [openmm.Vec3(*position) for position in positions]

    @staticmethod
    def _getAtomList(residue: mmapp.topology.Residue) -> t.List[int]:
        residue_atoms = {atom.name: atom.index for atom in residue.atoms()}
        if residue.name == "GLY":
            residue_atoms["CB"] = residue_atoms["HA2"]
        atom_list = []
        for atom in ("N", "CA", "CB", "C", "O"):
            try:
                atom_list.append(residue_atoms[atom])
            except KeyError as error:
                raise ValueError(
                    f"Atom {atom} not found in residue {residue.name}{residue.id}"
                ) from error
        return atom_list

    def getNumResidueBlocks(self) -> int:
        """
        Get the number of residue blocks.

        Returns
        -------
            The number of residue blocks.
        """
        return self._num_residue_blocks