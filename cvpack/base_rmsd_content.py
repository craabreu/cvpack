"""
.. class:: BaseRMSDContent
   :platform: Linux, MacOS, Windows
   :synopsis: Secondary-structure RMSD content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import re
from importlib import resources

import numpy as np
import openmm
from openmm import app as mmapp

from .collective_variable import CollectiveVariable
from .rmsd import RMSD
from .units import ScalarQuantity, value_in_md_units

# Limits on the number of collective variables per OpenMM CustomCVForce
MAX_NUM_BLOCKS = 1024
CHUNK_SIZE = 32


class BaseRMSDContent(CollectiveVariable, openmm.CustomCVForce):
    """
    Abstract class for secondary-structure RMSD content of a sequence of `n` residues.
    """

    def __init__(
        self,
        residue_blocks: list[int],
        references: list[list[openmm.Vec3]],
        residues: list[mmapp.topology.Residue],
        numAtoms: int,
        thresholdRMSD: ScalarQuantity,
        stepFunction: str = "(1+x^4)/(1+x^4+x^8)",
        normalize: bool = False,
    ):
        # As in PLUMED, the RMSD of a block is the smallest among all references.
        # Each reference takes one collective variable per block.
        num_references = len(references)
        chunk_size = CHUNK_SIZE // num_references
        max_num_blocks = MAX_NUM_BLOCKS // num_references
        num_residue_blocks = self._num_residue_blocks = len(residue_blocks)
        if not 1 <= num_residue_blocks <= max_num_blocks:
            raise ValueError(
                f"{num_residue_blocks} residue blocks were defined, which is not "
                f"between 1 and {max_num_blocks}"
            )
        residue_atoms = list(map(self._getAtomList, residues))
        block_atoms = [
            sum([residue_atoms[index] for index in block], [])
            for block in residue_blocks
        ]

        def get_expression(start):
            summands = []
            definitions = []
            for i in range(start, min(start + chunk_size, num_residue_blocks)):
                summands.append(re.sub(r"\bx\b", f"x{i}", stepFunction))
                rmsd = rmsd_names[i][0]
                for name in rmsd_names[i][1:]:
                    rmsd = f"min({rmsd},{name})"
                definitions.append(f"x{i}={rmsd}/{value_in_md_units(thresholdRMSD)}")
            summation = "+".join(summands)
            if normalize:
                summation = f"({summation})/{num_residue_blocks}"
            return ";".join([summation] + definitions)

        rmsd_names = [
            [f"rmsd{i}"]
            if num_references == 1
            else [f"rmsd{i}_{k}" for k in range(num_references)]
            for i in range(num_residue_blocks)
        ]
        if num_residue_blocks <= chunk_size:
            expression = get_expression(0)
            force = self
        else:
            expression = "+".join(
                f"chunk{i}"
                for i in range((num_residue_blocks + chunk_size - 1) // chunk_size)
            )
        super().__init__(expression)
        for index in range(num_residue_blocks):
            if num_residue_blocks > chunk_size and index % chunk_size == 0:
                force = openmm.CustomCVForce(get_expression(index))
                self.addCollectiveVariable(f"chunk{index // chunk_size}", force)
            for name, reference in zip(rmsd_names[index], references):
                force.addCollectiveVariable(
                    name,
                    RMSD(
                        dict(zip(block_atoms[index], reference)),
                        block_atoms[index],
                        numAtoms,
                    ),
                )

    @classmethod
    def _loadPositions(cls, filename: str) -> list[openmm.Vec3]:
        positions = 0.1 * np.loadtxt(
            str(resources.files("cvpack").joinpath("data").joinpath(filename)),
            delimiter=",",
        )
        return [openmm.Vec3(*position) for position in positions]

    @staticmethod
    def _getAtomList(residue: mmapp.topology.Residue) -> list[int]:
        residue_atoms = {atom.name: atom.index for atom in residue.atoms()}
        if residue.name == "GLY":
            # The hydrogen on the CB side is HA3 in PDB/Amber naming (HA2, HA3) and
            # HA1 in CHARMM naming (HA1, HA2), as in OpenMM's pdbNames.xml
            hydrogen = next((h for h in ("HA3", "HA1") if h in residue_atoms), None)
            if hydrogen is None:
                raise ValueError(
                    f"Atom HA3 or HA1 not found in residue {residue.name}{residue.id}"
                )
            residue_atoms["CB"] = residue_atoms[hydrogen]
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
