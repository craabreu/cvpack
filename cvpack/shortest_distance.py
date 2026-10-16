"""
.. class:: ShortestDistance
   :platform: Linux, MacOS, Windows
   :synopsis: A smooth approximation for the shortest distance between two atom groups

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import numpy as np
import openmm
from openmm import unit as mmunit

from .collective_variable import CollectiveVariable
from .units import Quantity


class ShortestDistance(CollectiveVariable, openmm.CustomCVForce):
    r"""
    A smooth approximation for the shortest distance between two atom groups:

    .. math::
        r_{\rm min}({\bf r}) = \frac{S_{rw}({\bf r})}{S_w({\bf r})}

    with

    .. math::

        S_{rw}({\bf r}) = r_c e^{-\beta} +
            \sum_{i \in {\bf g}_1} \sum_{j \in {\bf g}_2 \atop r_{ij} < r_c}
                r_{ij} \left(e^{-\beta \frac{r_{ij}}{r_c}} - e^{-\beta}\right)

    and

    .. math::

        S_w({\bf r}) = e^{-\beta} +
            \sum_{i \in {\bf g}_1} \sum_{j \in {\bf g}_2 \atop r_{ij} < r_c}
                \left(e^{-\beta \frac{r_{ij}}{r_c}} - e^{-\beta}\right)

    where :math:`r_{ij} = \|{\bf r}_j - {\bf r}_i\|` is the distance between atoms
    :math:`i` and :math:`j`, :math:`{\bf g}_1` and :math:`{\bf g}_2` are the sets of
    indices of the atoms in the first and second groups, respectively, :math:`r_c` is
    the cutoff distance, and :math:`\beta` is a parameter that controls the degree of
    approximation.

    The larger the value of :math:`\beta`, the closer the approximation to the true
    shortest distance. However, an excessively large value may lead to numerical
    instability.

    The terms outside the summations guarantee that the shortest distance is
    well-defined even when there are no atom pairs within the cutoff distance. In this
    case, the collective variable evaluates to the cutoff distance. Shifting each weight
    by :math:`e^{-\beta}` makes the contribution of a pair vanish at the cutoff
    distance, so the collective variable is continuous when a pair crosses it.

    .. note::

        Atoms are allowed to be in both groups. In this case, terms for which
        :math:`i = j` are ignored, and each pair of distinct atoms contributes a
        single term, even if both atoms are in both groups.

    .. note::

        GPU platforms accumulate forces in 64-bit fixed point, which cannot represent
        the forces of :math:`S_{rw}` and :math:`S_w` when the weights span many orders
        of magnitude. To keep them in range, the collective variable is evaluated in
        several windows. Each window multiplies all weights by a different constant,
        which leaves :math:`r_{\rm min}` unchanged, and the window used is the one
        whose :math:`S_w` is the largest not exceeding :math:`10^6`. The number of
        windows, and thus the cost, grows with :math:`\beta`: one window for
        :math:`\beta \leq 13`, five for the default :math:`\beta = 100`, and at most
        16. For :math:`\beta > 344`, 16 windows are not enough, and forces computed on
        GPU platforms lose precision.

    Parameters
    ----------
    group1
        The indices of the atoms in the first group.
    group2
        The indices of the atoms in the second group.
    numAtoms
        The total number of atoms in the system, including those that are not in any
        of the groups. This is necessary for the correct initialization of the
        collective variable.
    beta
        The parameter that controls the degree of approximation. It must be
        positive.
    cutoffDistance
        The cutoff distance for evaluating atom pairs.
    pbc
        Whether to use periodic boundary conditions in distance calculations. This
        should be ``True`` for periodic systems.
    name
        The name of the collective variable.

    Example
    -------
    >>> import cvpack
    >>> import numpy as np
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.HostGuestVacuum()
    >>> group1, group2 = [], []
    >>> for residue in model.topology.residues():
    ...     group = group1 if residue.name == "B2" else group2
    ...     group.extend(atom.index for atom in residue.atoms())
    >>> coords = model.positions.value_in_unit(mmunit.nanometers)
    >>> print(np.linalg.norm(
    ...     coords[group1, None, :] - coords[None, group2, :],
    ...     axis=-1,
    ... ).min())
    0.1757...
    >>> num_atoms = model.system.getNumParticles()
    >>> min_dist = cvpack.ShortestDistance(group1, group2, num_atoms)
    >>> min_dist.addToSystem(model.system)
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> integrator = openmm.VerletIntegrator(1.0 * mmunit.femtoseconds)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> min_dist.getValue(context)
    0.1785... nm
    """

    WINDOW_EFOLDS = 22
    MAX_WEIGHT_SUM = 1e6
    MAX_WINDOWS = 16

    def __init__(
        self,
        group1: t.Sequence[int],
        group2: t.Sequence[int],
        numAtoms: int,
        beta: float = 100,
        cutoffDistance: mmunit.Quantity = Quantity(1 * mmunit.nanometers),
        pbc: bool = False,
        name: str = "shortest_distance",
    ) -> None:
        if beta <= 0:
            raise ValueError("beta must be positive")
        rc = cutoffDistance
        if mmunit.is_quantity(rc):
            rc = rc.value_in_unit(mmunit.nanometers)
        # Window k multiplies all weights by exp(beta*a_k/rc), which leaves S_rw/S_w
        # unchanged. The offsets a_k decrease from rc in steps of efolds*rc/beta until
        # the weights of the last window are at most MAX_WEIGHT_SUM. A CustomCVForce
        # holds at most 32 inner forces, so large beta values need wider steps.
        log_max = np.log(self.MAX_WEIGHT_SUM)
        num_steps = max(0, int(np.ceil((beta - log_max) / self.WINDOW_EFOLDS)))
        num_windows = min(1 + num_steps, self.MAX_WINDOWS)
        efolds = self.WINDOW_EFOLDS
        if num_windows < 1 + num_steps:
            efolds = (beta - log_max) / (num_windows - 1)
        definitions = []
        expression = None
        super().__init__("")
        for k in reversed(range(num_windows)):
            offset = rc - k * efolds * rc / beta
            shift = np.exp(-k * efolds)
            # The clamp only acts on windows whose weight sum exceeds MAX_WEIGHT_SUM,
            # which are never selected, and prevents overflows in them. Factoring out
            # the shift keeps constants that underflow in single precision out of the
            # GPU kernels.
            exponent = f"min({2 * log_max}, {beta / rc}*({offset}-r))"
            weight = f"exp({exponent})*(1-exp({beta / rc}*(r-{rc})))"
            for cv, function in [(f"wrsum{k}", f"{weight}*r"), (f"wsum{k}", weight)]:
                force = openmm.CustomNonbondedForce(function)
                force.setNonbondedMethod(
                    force.CutoffPeriodic if pbc else force.CutoffNonPeriodic
                )
                force.setCutoffDistance(cutoffDistance)
                force.setUseSwitchingFunction(False)
                force.setUseLongRangeCorrection(False)
                for _ in range(numAtoms):
                    force.addParticle([])
                force.addInteractionGroup(group1, group2)
                self.addCollectiveVariable(cv, force)
            definitions.append(f"value{k}=({rc * shift}+wrsum{k})/sum{k}")
            definitions.append(f"sum{k}={shift}+wsum{k}")
            if expression is None:
                expression = f"value{k}"
            else:
                condition = f"step({self.MAX_WEIGHT_SUM}-sum{k})"
                expression = f"select({condition}, value{k}, {expression})"
        self.setEnergyFunction("; ".join([expression, *definitions]))
        self._registerCV(
            name,
            mmunit.nanometers,
            group1=group1,
            group2=group2,
            numAtoms=numAtoms,
            beta=beta,
            cutoffDistance=rc,
            pbc=pbc,
        )


ShortestDistance.registerTag("!cvpack.ShortestDistance")
