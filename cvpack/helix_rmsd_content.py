"""
.. class:: HelixRMSDContent
   :platform: Linux, MacOS, Windows
   :synopsis: Alpha-helix RMSD content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import typing as t

from openmm import app as mmapp
from openmm import unit as mmunit

from .base_rmsd_content import BaseRMSDContent
from .units import Quantity, ScalarQuantity
from .utils import check_contiguous_residues

ALPHA_POSITIONS = BaseRMSDContent._loadPositions("ideal_alpha_helix.csv")


class HelixRMSDContent(BaseRMSDContent):
    r"""
    The alpha-helix RMSD content of a sequence of `n` residues :cite:`Pietrucci_2009`:

    .. math::

        \alpha_{\rm rmsd}({\bf r}) = \sum_{i=1}^{n-5} S\left(
            \frac{r_{\rm rmsd}({\bf g}_i, {\bf g}_{\rm ref})}{r_0}
        \right)

    where :math:`{\bf g}_i` represents a group of atoms selected from residues
    :math:`i` to :math:`i+5` of the sequence, :math:`{\bf g}_{\rm ref}` represents
    the same atoms in an ideal alpha-helix configuration,
    :math:`r_{\rm rmsd}({\bf g}, {\bf g}_{\rm ref})` is the root-mean-square
    deviation (RMSD) between groups :math:`{\bf g}` and :math:`{\bf g}_{\rm ref}`,
    :math:`r_0` is a threshold RMSD value, and :math:`S(x)` is a smooth step function
    whose default form is

    .. math::
        S(x) = \frac{1 + x^4}{1 + x^4 + x^8}

    The residues must be a contiguous sequence from a single chain, ordered from
    the N-terminus to the C-terminus.

    Every group :math:`{\bf g}_{i,j}` is formed by the N, :math:`{\rm C}_\alpha`,
    :math:`{\rm C}_\beta`, C, and O atoms of the six residues involved, thus
    comprising 30 atoms in total. In the case of glycine, the missing
    :math:`{\rm C}_\beta` atom is replaced by the hydrogen atom in its position (HA3 in
    PDB naming, HA1 in CHARMM naming), whereas PLUMED uses HA2. The RMSD is then
    defined as

    .. math::

        r_{\rm rmsd}({\bf g}, {\bf g}_{\rm ref}) =
            \sqrt{\frac{1}{30} \sum_{j=1}^{30} \left\|
                \hat{\bf r}_j({\bf g}) -
                {\bf A}({\bf g})\hat{\bf r}_j({\bf g}_{\rm ref})
            \right\|^2}

    where :math:`\hat{\bf r}_k({\bf g})` is the position of the :math:`k`-th atom in
    a group :math:`{\bf g}` relative to the group's center of geometry and
    :math:`{\bf A}({\bf g})` is the rotation matrix that minimizes the RMSD between
    :math:`{\bf g}` and :math:`{\bf g}_{\rm ref}`.

    Optionally, the alpha-helix RMSD content can be normalized to the range
    :math:`[0, 1]`. This is done by dividing its value by :math:`N_{\rm groups} =
    n - 5`.

    This collective variable was introduced in Ref. :cite:`Pietrucci_2009`. The default
    step function shown above is identical to the one in the original paper, but written
    in a numerically safe form. The ideal alpha-helix configuration is the same as that
    used for the collective variable `ALPHARMSD`_ in `PLUMED v2.8.1
    <https://github.com/plumed/plumed2>`_.

    .. note::

        The present implementation is limited to :math:`1 \leq N_{\rm groups} \leq
        1024`.

    .. _ALPHARMSD: https://www.plumed.org/doc-v2.8/user-doc/html/_a_l_p_h_a_r_m_s_d.html

    Parameters
    ----------
    residues
        The residue sequence to be used in the calculation.
    numAtoms
        The total number of atoms in the system (required by OpenMM).
    thresholdRMSD
        The threshold RMSD value for considering a group of residues as a close
        match to an ideal alpha-helix.
    stepFunction
        The form of the step function :math:`S(x)`.
    normalize
        Whether to normalize the collective variable to the range :math:`[0, 1]`.
    name
        The name of the collective variable.

    Example
    -------
    >>> import itertools as it
    >>> import cvpack
    >>> import openmm
    >>> from openmm import app, unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.LysozymeImplicit()
    >>> residues = list(it.islice(model.topology.residues(), 59, 80))
    >>> print(*[r.name for r in residues])
    LYS ASP GLU ... ILE LEU ARG
    >>> helix_content = cvpack.HelixRMSDContent(
    ...     residues, model.system.getNumParticles()
    ... )
    >>> helix_content.getNumResidueBlocks()
    16
    >>> helix_content.addToSystem(model.system)
    >>> normalized_helix_content = cvpack.HelixRMSDContent(
    ...     residues, model.system.getNumParticles(), normalize=True
    ... )
    >>> normalized_helix_content.addToSystem(model.system)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> integrator = openmm.VerletIntegrator(0)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> helix_content.getValue(context)
    15.98... dimensionless
    >>> normalized_helix_content.getValue(context)
    0.999... dimensionless
    """

    def __init__(
        self,
        residues: t.Sequence[mmapp.topology.Residue],
        numAtoms: int,
        thresholdRMSD: ScalarQuantity = Quantity(0.08 * mmunit.nanometers),
        stepFunction: str = "(1+x^4)/(1+x^4+x^8)",
        normalize: bool = False,
        name: str = "helix_rmsd_content",
    ) -> None:
        residues = list(residues)
        check_contiguous_residues(residues)
        residue_blocks = [
            list(range(index, index + 6)) for index in range(len(residues) - 5)
        ]

        super().__init__(
            residue_blocks,
            [ALPHA_POSITIONS],
            residues,
            numAtoms,
            thresholdRMSD,
            stepFunction,
            normalize,
        )
        self._registerCV(
            name,
            mmunit.dimensionless,
            residues=residues,
            numAtoms=numAtoms,
            thresholdRMSD=thresholdRMSD,
            stepFunction=stepFunction,
            normalize=normalize,
        )


HelixRMSDContent.registerTag("!cvpack.HelixRMSDContent")
