"""
.. class:: SheetRMSDContent
   :platform: Linux, MacOS, Windows
   :synopsis: Beta-sheet RMSD content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import numpy as np
from openmm import app as mmapp
from openmm import unit as mmunit

from .base_rmsd_content import BaseRMSDContent
from .units import Quantity, ScalarQuantity
from .utils import check_contiguous_residues

PARABETA_POSITIONS = BaseRMSDContent._loadPositions("ideal_parallel_beta_sheet.csv")
ANTIBETA_POSITIONS = BaseRMSDContent._loadPositions("ideal_antiparallel_beta_sheet.csv")


class SheetRMSDContent(BaseRMSDContent):
    r"""
    The beta-sheet RMSD content of `n` residues :cite:`Pietrucci_2009`:

    .. math::

        \beta_{\rm rmsd}({\bf r}) = \sum_{i=1}^{n-2-d_{\rm min}}
            \sum_{j=i+d_{\rm min}}^{n-2} S\left(
                \frac{r_{\rm rmsd}({\bf g}_{i,j}, {\bf g}_{\rm ref})}{r_0}
            \right)

    where :math:`{\bf g}_{i,j}` represents the coordinates of a group of atoms
    selected from residues i to i+2 and residues j to j+2 of the sequence,
    :math:`d_{\rm min}` is the minimum distance between integers i and j,
    :math:`{\bf g}_{\rm ref}` represents the coordinates of the same atoms in an
    ideal beta-sheet configuration, :math:`r_{\rm rmsd}({\bf g}, {\bf g}_{\rm ref})`
    is the root-mean-square deviation (RMSD) between groups :math:`{\bf g}` and
    :math:`{\bf g}_{\rm ref}`, :math:`r_0` is a threshold RMSD value, and
    :math:`S(x)` is a smooth step function whose default form is

    .. math::
        S(x) = \frac{1 + x^4}{1 + x^4 + x^8}

    The sequence of residues must be a contiguous subset of a protein chain, ordered
    from the N-terminus to the C-terminus. The beta-sheet RMSD content is defined for
    both antiparallel and parallel beta sheets, with different ideal configurations
    :math:`{\bf g}_{\rm ref}` and minimum distances :math:`d_{\rm min}` in each case.

    Every group :math:`{\bf g}_{i,j}` is formed by the N, :math:`{\rm C}_\alpha`,
    :math:`{\rm C}_\beta`, C, and O atoms of the six residues involved, thus
    comprising 30 atoms in total. In the case of glycine, the missing
    :math:`{\rm C}_\beta` atom is replaced by the corresponding H atom. The RMSD is
    then defined as

    .. math::

        r_{\rm rmsd}({\bf g}, {\bf g}_{\rm ref}) =
            \sqrt{\frac{1}{30} \sum_{k=1}^{30} \left\|
                \hat{\bf r}_k({\bf g}) -
                {\bf A}({\bf g})\hat{\bf r}_k({\bf g}_{\rm ref})
            \right\|^2}

    where :math:`\hat{\bf r}_k({\bf g})` is the position of the :math:`k`-th atom in
    a group :math:`{\bf g}` relative to the group's center of geometry and
    :math:`{\bf A}({\bf g})` is the rotation matrix that minimizes the RMSD between
    :math:`{\bf g}` and :math:`{\bf g}_{\rm ref}`.

    In addition, one can choose to partition the sequence of residues into :math:`m`
    blocks with :math:`n_1, n_2, \ldots, n_m` residues. In this case, only groups
    :math:`{\bf g}_{i,j}` composed of three consecutive residues from a block and
    three consecutive residues from the next block will be considered. The definition
    of the RMSD content is then modified to

    .. math::

        \beta_{\rm rmsd}({\bf r}) = \sum_{k=1}^{m-1} \sum_{i=l_{k-1}+1}^{l_{k}-2}
            \sum_{j=l_k+1}^{l_{k+1}-2} S\left(
                \frac{r_{\rm rmsd}({\bf g}_{i,j}, {\bf g}_{\rm ref})}{r_0}
            \right)

    where :math:`l_k = \sum_{i=1}^k n_i` is the index of the last residue in block
    :math:`k`. All blocks must be contiguous subsets of the same protein chain, ordered
    from the N-terminus to the C-terminus. However, each block can be separated from the
    next one by any number of residues.

    Optionally, the beta-sheet RMSD content can be normalized to the range
    :math:`[0, 1]`. This is done by dividing its value by the number of
    :math:`{\bf g}_{i,j}` groups. For a single, contiguous sequence of residues, this
    number is

    .. math::

        N_{\rm groups} = \frac{(n - 2 - d_{\rm min})(n - 1 - d_{\rm min})}{2}

    In the case of a partitioned sequence, the number of groups is given by

    .. math::

        N_{\rm groups} = \sum_{k=1}^{m-1} (n_k - 2)(n_{k+1} - 2)

    This collective variable was introduced in Ref. :cite:`Pietrucci_2009`. The default
    step function shown above is identical to the one in the original paper, but written
    in a numerically safe form. The ideal beta-sheet configurations are the same as
    those used for the collective variables `ANTIBETARMSD`_ and `PARABETARMSD`_ in
    `PLUMED v2.8.1 <https://github.com/plumed/plumed2>`_.

    .. _ANTIBETARMSD:
        https://www.plumed.org/doc-v2.8/user-doc/html/_a_n_t_i_b_e_t_a_r_m_s_d.html

    .. _PARABETARMSD:
        https://www.plumed.org/doc-v2.8/user-doc/html/_p_a_r_a_b_e_t_a_r_m_s_d.html

    .. note::

        The present implementation is limited to :math:`1 \leq N_{\rm groups} \leq
        1024`.

    Parameters
    ----------
    residues
        The residue sequence or residue blocks to be used in the calculation.
    numAtoms
        The total number of atoms in the system (required by OpenMM).
    parallel
        Whether to consider a parallel beta sheet instead of an antiparallel one.
    blockSizes
        The number of residues in each block. If ``None``, a single contiguous
        sequence of residues is assumed.
    thresholdRMSD
        The threshold RMSD value for considering a group of residues as a close
        match to an ideal beta sheet.
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
    >>> model = testsystems.SrcImplicit()
    >>> residues = list(it.islice(model.topology.residues(), 68, 82))
    >>> print(*[r.name for r in residues])
    TYR ALA VAL ... VAL THR GLU
    >>> sheet_content = cvpack.SheetRMSDContent(
    ...     residues, model.system.getNumParticles()
    ... )
    >>> sheet_content.getNumResidueBlocks()
    28
    >>> sheet_content.addToSystem(model.system)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> integrator = openmm.VerletIntegrator(0)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> sheet_content.getValue(context)
    1.0465... dimensionless
    >>> blockwise_sheet_content = cvpack.SheetRMSDContent(
    ...     residues[:5] + residues[-5:],
    ...     model.system.getNumParticles(),
    ...     blockSizes=[5, 5],
    ... )
    >>> blockwise_sheet_content.getNumResidueBlocks()
    9
    >>> blockwise_sheet_content.addToSystem(model.system)
    >>> context.reinitialize(preserveState=True)
    >>> blockwise_sheet_content.getValue(context)
    0.9859... dimensionless
    """

    def __init__(
        self,
        residues: t.Sequence[mmapp.topology.Residue],
        numAtoms: int,
        parallel: bool = False,
        blockSizes: t.Sequence[int] | None = None,
        thresholdRMSD: ScalarQuantity = Quantity(0.08 * mmunit.nanometers),
        stepFunction: str = "(1+x^4)/(1+x^4+x^8)",
        normalize: bool = False,
        name: str = "sheet_rmsd_content",
    ) -> None:
        residues = list(residues)
        if blockSizes is None:
            check_contiguous_residues(residues)
            min_distance = 6 if parallel else 5
            residue_groups = [
                [i, i + 1, i + 2, j, j + 1, j + 2]
                for i in range(len(residues) - 2 - min_distance)
                for j in range(i + min_distance, len(residues) - 2)
            ]
        elif sum(blockSizes) == len(residues):
            bounds = np.insert(np.cumsum(blockSizes), 0, 0)
            for k in range(len(blockSizes)):
                check_contiguous_residues(residues[bounds[k] : bounds[k + 1]])
            residue_groups = [
                [i, i + 1, i + 2, j, j + 1, j + 2]
                for k in range(len(blockSizes) - 1)
                for i in range(bounds[k], bounds[k + 1] - 2)
                for j in range(bounds[k + 1], bounds[k + 2] - 2)
            ]
        else:
            raise ValueError(
                f"The sum of block sizes ({sum(blockSizes)}) and the "
                f"number of residues ({len(residues)}) must be equal."
            )

        super().__init__(
            residue_groups,
            PARABETA_POSITIONS if parallel else ANTIBETA_POSITIONS,
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
            parallel=parallel,
            blockSizes=blockSizes,
            thresholdRMSD=thresholdRMSD,
            stepFunction=stepFunction,
            normalize=normalize,
        )


SheetRMSDContent.registerTag("!cvpack.SheetRMSDContent")
