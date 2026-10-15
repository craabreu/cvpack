"""
.. class:: PathInRMSDSpace
   :platform: Linux, MacOS, Windows
   :synopsis: A metric of progress or deviation with respect to a path in RMSD space

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

from openmm import unit as mmunit

from .base_path_cv import BasePathCV
from .path import Metric, progress
from .rmsd import RMSD
from .units.units import VectorQuantity


class PathInRMSDSpace(BasePathCV):
    r"""
    A metric of the system's progress (:math:`s`) or deviation (:math:`z`) with
    respect to a path defined by a sequence of :math:`n` milestones defined as
    reference structures :cite:`Branduardi_2007`:

    .. math::

        s({\bf r}) = \frac{
            \dfrac{\sum_{i=1}^n i w_i({\bf r})}{\sum_{i=1}^n w_i({\bf r})} - 1
        }{n-1}
        \quad \text{or} \quad
        z({\bf r}) = - 2 \sigma ^2 \ln \sum_{i=1}^n w_i({\bf r})

    with :math:`w_i({\bf r})` being a Gaussian kernel centered at the :math:`i`-th
    milestone, i.e.,

    .. math::

        w_i({\bf r}) = \exp\left(\
            -\frac{d^2_{\rm rms}({\bf r},{\bf r}^{\rm ref}_i)}{2 \sigma^2}
        \right)

    where :math:`d_{\rm rms}({\bf r},{\bf r}^{\rm ref}_i)` is the root-mean-square
    distance between the current system state and the :math:`i`-th reference structure
    and :math:`\sigma` sets the width of the kernels.

    .. note::

        The kernel width :math:`\sigma` is related to the parameter :math:`\lambda` of
        Ref. :cite:`Branduardi_2007` by :math:`\sigma = \frac{1}{\sqrt{2\lambda}}`.

    Parameters
    ----------
    metric
        The path-related metric to compute. Use ``cvpack.path.progress`` for
        computing :math:`s({\bf r})` or ``cvpack.path.deviation`` for computing
        :math:`z({\bf r})`.
    milestones
        A sequence of reference structures, each represented as a dictionary mapping
        atom indices to coordinate vectors.
    numAtoms
        The total number of atoms in the system, including those that are not in
        any of the reference structures.
    sigma
        The width of the Gaussian kernels in nanometers
    name
        The name of the collective variable. If not provided, it is set to
        "path_progress_in_rmsd_space" or "path_deviation_in_rmsd_space" depending
        on the metric.

    Raises
    ------
    ValueError
        The number of milestones is less than 2
    ValueError
        If the metric is not `cvpack.path.progress` or `cvpack.path.deviation`

    Examples
    --------
    >>> import cvpack
    >>> import networkx as nx
    >>> import numpy as np
    >>> import openmm
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> from scipy.spatial.transform import Rotation
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> atom1, atom2 = 8, 14
    >>> graph = model.mdtraj_topology.to_bondgraph()
    >>> nodes = list(graph.nodes)
    >>> graph.remove_edge(nodes[atom1], nodes[atom2])
    >>> movable = list(nx.connected_components(graph))[1]
    >>> x = model.positions / model.positions.unit
    >>> x0 = x[atom1, :]
    >>> vector = x[atom2, :] - x0
    >>> vector /= np.linalg.norm(vector)
    >>> rotation = Rotation.from_rotvec((np.pi / 6) * vector)
    >>> atoms = [nodes.index(atom) for atom in movable]
    >>> frames = [x.copy()]
    >>> for _ in range(6):
    ...     x[atoms, :] = x0 + rotation.apply(x[atoms, :] - x0)
    ...     frames.append(x.copy())
    >>> milestones = [
    ...     {i: row for i, row in enumerate(frame)}
    ...     for frame in frames
    ... ]
    >>> s, z = [
    ...    cvpack.PathInRMSDSpace(
    ...        metric, milestones, len(x), 0.5 * unit.angstrom
    ...    )
    ...    for metric in (cvpack.path.progress, cvpack.path.deviation)
    ... ]
    >>> s.addToSystem(model.system)
    >>> z.addToSystem(model.system)
    >>> context = openmm.Context(model.system, openmm.VerletIntegrator(0.001))
    >>> context.setPositions(model.positions)
    >>> s.getValue(context)
    0.172... dimensionless
    >>> z.getValue(context)
    -0.004... nm**2
    """

    def __init__(  # pylint: disable=too-many-branches
        self,
        metric: Metric,
        milestones: t.Sequence[dict[int, VectorQuantity]],
        numAtoms: int,
        sigma: mmunit.Quantity,
        name: str | None = None,
    ) -> None:
        name = self._generateName(metric, name, "rmsd")
        if mmunit.is_quantity(sigma):
            sigma = sigma.value_in_unit(mmunit.nanometers)
        n = len(milestones)
        if n < 2:
            raise ValueError("At least two reference structures are required.")
        collective_variables = {
            f"rmsd{i}": RMSD(reference, reference.keys(), numAtoms, name=f"rmsd{i}")
            for i, reference in enumerate(milestones)
        }
        squared_distances = [f"rmsd{i}^2" for i in range(n)]
        super().__init__(metric, sigma, squared_distances, collective_variables)
        self._registerCV(
            name,
            mmunit.dimensionless if metric == progress else mmunit.nanometers**2,
            metric=metric,
            milestones=[{k: list(v) for k, v in m.items()} for m in milestones],
            numAtoms=numAtoms,
            sigma=sigma,
        )


PathInRMSDSpace.registerTag("!cvpack.PathInRMSDSpace")
