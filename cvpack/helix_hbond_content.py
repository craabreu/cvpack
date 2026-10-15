"""
.. class:: HelixHBondContent
   :platform: Linux, MacOS, Windows
   :synopsis: Alpha-helix hydrogen-bond content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import re as regex
import typing as t

import openmm
from openmm import app as mmapp
from openmm import unit as mmunit

from .collective_variable import CollectiveVariable
from .units import Quantity, ScalarQuantity
from .utils import check_contiguous_residues


class HelixHBondContent(CollectiveVariable, openmm.CustomBondForce):
    r"""
    The alpha-helix hydrogen-bond content of a sequence of `n` residues:

    .. math::

        \alpha_{\rm HB}({\bf r}) = \sum_{i=5}^n B_m\left(
            \frac{\| {\bf r}^{\rm H}_i - {\bf r}^{\rm O}_{i-4} \|}{d_{\rm HB}}
        \right)

    where :math:`{\bf r}^{\rm H}_k` and :math:`{\bf r}^{\rm O}_k` are the positions
    of the hydrogen and oxygen atoms bonded, respectively, to the backbone nitrogen and
    carbon atoms of residue :math:`k`. In addition, :math:`d_{\rm HB}` is the threshold
    distance for a hydrogen bond and :math:`B_m(x)` is a smooth step function given by

    .. math::
        B_m(x) = \frac{1}{1 + x^{2m}}

    where :math:`m` is an integer parameter that controls its steepness.

    Optionally, this collective variable can be normalized to the range :math:`[0, 1]`.

    .. note::

        The residues must be a contiguous sequence from a single chain, ordered from the
        N- to the C-terminus. Proline residues, which have no amide hydrogen, are
        skipped as donors, and the normalization uses the number of remaining
        donor-acceptor pairs.

    Parameters
    ----------
    residues
        The residues in the sequence.
    pbc
        Whether to use periodic boundary conditions.
    thresholdDistance
        The threshold distance for a hydrogen bond.
    halfExponent
        The parameter :math:`m` of the step function.
    name
        The name of the collective variable.

    Example
    -------
    >>> import cvpack
    >>> import openmm
    >>> from openmm import app, unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.LysozymeImplicit()
    >>> residues = list(model.topology.residues())[59:80]
    >>> print(*[r.name for r in residues])
    LYS ASP GLU ... ILE LEU ARG
    >>> helix_content = cvpack.HelixHBondContent(residues)
    >>> helix_content.addToSystem(model.system)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> integrator = openmm.VerletIntegrator(0)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> helix_content.getValue(context)
    15.880... dimensionless
    """

    def __init__(
        self,
        residues: t.Sequence[mmapp.topology.Residue],
        pbc: bool = False,
        thresholdDistance: ScalarQuantity = Quantity(0.33 * mmunit.nanometers),
        halfExponent: int = 3,
        normalize: bool = False,
        name: str = "helix_hbond_content",
    ) -> None:
        residues = list(residues)
        check_contiguous_residues(residues)
        if normalize and len(residues) < 5:  # noqa: PLR2004
            raise ValueError("Normalization requires at least 5 residues")

        def find_atom(residue: mmapp.topology.Residue, pattern: t.Pattern) -> int:
            for atom in residue.atoms():
                if regex.match(pattern, atom.name):
                    return atom.index
            raise ValueError(
                f"Could not find atom matching regex "
                f"'{pattern.pattern}'"
                f" in residue {residue.name}{residue.id}"
            )

        pairs = [
            (residues[i - 4], residues[i])
            for i in range(4, len(residues))
            if residues[i].name != "PRO"  # proline has no amide hydrogen
        ]
        if normalize and not pairs:
            raise ValueError("No hydrogen-bond donors found in the residue sequence")
        numerator = 1 / len(pairs) if normalize else 1
        threshold = thresholdDistance
        if mmunit.is_quantity(threshold):
            threshold = threshold.value_in_unit_system(mmunit.md_unit_system)
        super().__init__(f"{numerator}/(1+x^{2 * halfExponent}); x=r/{threshold}")
        hydrogen_pattern = regex.compile(r"\b(H|1H|HN1|HT1|H1|HN)\b")
        oxygen_pattern = regex.compile(r"\b(O|OCT1|OC1|OT1|O1)\b")
        for acceptor, donor in pairs:
            self.addBond(
                find_atom(acceptor, oxygen_pattern),
                find_atom(donor, hydrogen_pattern),
                [],
            )
        self.setUsesPeriodicBoundaryConditions(pbc)
        self._registerCV(
            name,
            mmunit.dimensionless,
            residues=residues,
            pbc=pbc,
            thresholdDistance=thresholdDistance,
            halfExponent=halfExponent,
            normalize=normalize,
        )


HelixHBondContent.registerTag("!cvpack.HelixHBondContent")
