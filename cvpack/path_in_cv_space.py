"""
.. class:: PathInCVSpace
   :platform: Linux, MacOS, Windows
   :synopsis: A metric of progress or deviation with respect to a path in CV space

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from copy import deepcopy

from openmm import unit as mmunit

from .base_path_cv import BasePathCV
from .collective_variable import CollectiveVariable
from .path import Metric
from .units.units import MatrixQuantity, ScalarQuantity, value_in_md_units
from .utils import convert_to_matrix


class PathInCVSpace(BasePathCV):
    r"""
    A metric of the system's progress (:math:`s`) or deviation (:math:`z`) with
    respect to a path defined by a sequence of :math:`n` milestones positioned in a
    collective variable space :cite:`Branduardi_2007`:

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
            -\frac{\|{\bf c}({\bf r}) - \hat{\bf c}_i\|^2}{2 \sigma^2}
        \right)

    where :math:`{\bf c}({\bf r})` is a vector of collective variables,
    :math:`\hat{\bf c}_i` is the location of the :math:`i`-th milestone, and
    :math:`\sigma` sets the width of the kernels. The squared norm of a vector
    :math:`{\bf x}` in the collective variable space is defined as

    .. math::

        \|{\bf x}\|^2 = {\bf x}^T {\bf D}^{-2} {\bf x}

    where :math:`{\bf D}` is a diagonal matrix whose each diagonal element is the
    characteristic scale of the corresponding collective variable, which makes
    :math:`\|{\bf x}\|^2` dimensionless. Appropriate boundary conditions are used for
    periodic CVs.

    .. note::

        The kernel width :math:`\sigma` is related to the parameter :math:`\lambda` of
        Ref. :cite:`Branduardi_2007` by :math:`\sigma = \frac{1}{\sqrt{2\lambda}}`.

    Parameters
    ----------
    metric
        The path-related metric to compute. Use ``cvpack.path.progress`` for
        computing :math:`s({\bf r})` or ``cvpack.path.deviation`` for computing
        :math:`z({\bf r})`.
    variables
        The collective variables that define the space.
    milestones
        The milestones in the collective variable space. The number of rows must be
        equal to the number of milestones and the number of columns must be equal to
        the number of collective variables.
    sigma
        The width of the Gaussian kernels.
    scales
        The characteristic scales for the collective variables. If not provided, the
        scales are assumed to be 1 (in standard MD units) for each collective variable.
    name
        The name of the collective variable. If not provided, it is set to
        "path_progress_in_cv_space" or "path_deviation_in_cv_space" depending on the
        metric.

    Raises
    ------
    ValueError
        If the number of rows in the milestones matrix is less than 2
    ValueError
        If the number of columns in the milestones matrix is different from the number
        of collective variables
    ValueError
        If the metric is not `cvpack.path.progress` or `cvpack.path.deviation`

    Examples
    --------
    >>> import cvpack
    >>> import openmm
    >>> from openmmtools import testsystems
    >>> import numpy as np
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi_atoms = ["ACE-C", "ALA-N", "ALA-CA", "ALA-C"]
    >>> psi_atoms = ["ALA-N", "ALA-CA", "ALA-C", "NME-N"]
    >>> atoms = [f"{a.residue.name}-{a.name}" for a in model.topology.atoms()]
    >>> milestones = np.array(
    ...     [[1.3, -0.2], [1.2, 3.1], [-2.7, 2.9], [-1.3, 2.7]]
    ... )
    >>> phi = cvpack.Torsion(*[atoms.index(atom) for atom in phi_atoms])
    >>> psi = cvpack.Torsion(*[atoms.index(atom) for atom in psi_atoms])
    >>> path_vars = []
    >>> for metric in (cvpack.path.progress, cvpack.path.deviation):
    ...     var = cvpack.PathInCVSpace(metric, [phi, psi], milestones, np.pi / 6)
    ...     var.addToSystem(model.system)
    ...     path_vars.append(var)
    >>> context = openmm.Context(model.system, openmm.VerletIntegrator(1.0))
    >>> context.setPositions(model.positions)
    >>> path_vars[0].getValue(context)
    0.6... dimensionless
    >>> path_vars[1].getValue(context)
    0.2... dimensionless
    """

    def __init__(
        self,
        metric: Metric,
        variables: t.Iterable[CollectiveVariable],
        milestones: MatrixQuantity,
        sigma: float,
        scales: t.Iterable[ScalarQuantity] | None = None,
        name: str | None = None,
    ) -> None:
        name = self._generateName(metric, name, "cv")
        variables = list(variables)
        cv_scales = [1.0] * len(variables) if scales is None else list(scales)
        milestones, n, numvars = convert_to_matrix(milestones)
        if numvars != len(variables):
            raise ValueError("Wrong number of columns in the milestones matrix.")
        if n < 2:  # noqa: PLR2004
            raise ValueError("At least two rows are required in the milestones matrix.")
        squared_distances = []
        periods = {}
        for i, variable in enumerate(variables):
            values = variable.getPeriodicBounds()
            if values is not None:
                periods[i] = value_in_md_units(values[1] - values[0])
        for values in milestones:
            deltas = [f"({value}-cv{j})" for j, value in enumerate(values)]
            for j, period in periods.items():
                deltas[j] = f"min(abs{deltas[j]},{period}-abs{deltas[j]})"
            squared_distances.append(
                "+".join(
                    f"{delta}^2" if scale == 1.0 else f"({delta}/{scale})^2"
                    for delta, scale in zip(deltas, cv_scales)
                )
            )
        collective_variables = {
            f"cv{i}": deepcopy(variable) for i, variable in enumerate(variables)
        }
        super().__init__(metric, sigma, squared_distances, collective_variables)
        self._registerCV(
            name,
            mmunit.dimensionless,
            metric=metric,
            variables=variables,
            milestones=milestones.tolist(),
            sigma=sigma,
            scales=scales,
        )


PathInCVSpace.registerTag("!cvpack.PathInCVSpace")
