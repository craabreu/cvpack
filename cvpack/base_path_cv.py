"""
.. class:: BasePathCV
   :platform: Linux, MacOS, Windows
   :synopsis: A base class for path-related collective variables

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import re
import typing as t
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import openmm

from .collective_variable import CollectiveVariable
from .path import Metric, deviation, progress

# OpenMM's limit on the number of collective variables of a CustomCVForce
MAX_CVS = 32


def _kernel_sum_expression(
    squared_distances: t.Sequence[str],
    lam: float,
    progress_terms: t.Sequence[str] | None = None,
    divisor: float = 1,
) -> str:
    """
    Return an expression for the soft minimum of the squared distances, or, if
    ``progress_terms`` is given, for the average of these terms weighted by the
    Gaussian kernels and divided by ``divisor``.
    """
    n = len(squared_distances)
    definitions = OrderedDict(
        {f"x{i}": sqdist for i, sqdist in enumerate(squared_distances)}
    )
    definitions["lambda"] = lam
    definitions["xmin0"] = "x0"
    for i in range(1, n):
        definitions[f"xmin{i}"] = f"min(xmin{i - 1},x{i})"
    # Shifting all exponents by about the smallest squared distance keeps the
    # weights between exp(-1) and 1 for the nearest milestone. The metrics do not
    # depend on the shift, and floor() has a zero derivative, so the derivatives
    # of the expression do not contain those of the chain of minima, which made
    # Context creation scale steeply with the number of milestones.
    definitions["shift"] = f"floor(lambda*xmin{n - 1})/lambda"
    for i in range(n):
        definitions[f"w{i}"] = f"exp(lambda*(shift-x{i}))"
    definitions["wsum"] = "+".join(f"w{i}" for i in range(n))
    expressions = [f"{key}={value}" for key, value in definitions.items()]
    if progress_terms is None:
        expressions.append("shift-log(wsum)/lambda")
    else:
        numerator = "+".join(
            f"{term}*w{i}" for i, term in enumerate(progress_terms) if term != "0"
        )
        expressions.append(f"({numerator or 0})/({divisor}*wsum)")
    return "; ".join(reversed(expressions))


def _make_cv_force(
    expression: str, variables: dict[str, CollectiveVariable]
) -> openmm.CustomCVForce:
    force = openmm.CustomCVForce(expression)
    for name, variable in variables.items():
        # Each force takes ownership of its variables, which cannot be shared
        force.addCollectiveVariable(name, deepcopy(variable))
    return force


class BasePathCV(CollectiveVariable, openmm.CustomCVForce):
    """
    A base class for path-related collective variables

    Parameters
    ----------
    metric
        A measure of progress or deviation with respect to a path in CV space
    sigma
        The width of the Gaussian kernels
    squared_distances
        Expressions for the squared distance to each milestone
    variables
        A dictionary of collective variables used in the expressions for the squared
        distances
    """

    def __init__(
        self,
        metric: Metric,
        sigma: float,
        squared_distances: t.Sequence[str],
        variables: dict[str, CollectiveVariable],
    ) -> None:
        n = len(squared_distances)
        lam = 1 / (2 * sigma**2)
        progress_terms = [str(i) for i in range(n)] if metric == progress else None
        if len(variables) <= MAX_CVS:
            super().__init__(
                _kernel_sum_expression(squared_distances, lam, progress_terms, n - 1)
            )
            for name, variable in variables.items():
                self.addCollectiveVariable(name, variable)
            return
        # A CustomCVForce cannot have more than MAX_CVS variables, so each chunk of
        # milestones becomes an inner CustomCVForce. The deviation is a soft minimum,
        # which can be taken over the soft minima of the chunks. The progress also
        # needs the progress within each chunk, weighted by the chunk's kernel sum.
        forces_per_chunk = 1 if metric == deviation else 2
        num_chunks = -(-n // MAX_CVS)
        if forces_per_chunk * num_chunks > MAX_CVS:
            raise ValueError(
                f"A path with {metric.name} metric supports at most "
                f"{MAX_CVS * MAX_CVS // forces_per_chunk} milestones."
            )
        outer_variables = {}
        for c, chunk in enumerate(np.array_split(np.arange(n), num_chunks)):
            chunk_distances = [squared_distances[i] for i in chunk]
            chunk_variables = {
                name: variable
                for name, variable in variables.items()
                if any(re.search(rf"\b{name}\b", x) for x in chunk_distances)
            }
            outer_variables[f"z{c}"] = _make_cv_force(
                _kernel_sum_expression(chunk_distances, lam), chunk_variables
            )
            if metric == progress:
                outer_variables[f"s{c}"] = _make_cv_force(
                    _kernel_sum_expression(
                        chunk_distances, lam, [str(i) for i in chunk], n - 1
                    ),
                    chunk_variables,
                )
        super().__init__(
            _kernel_sum_expression(
                [f"z{c}" for c in range(num_chunks)],
                lam,
                [f"s{c}" for c in range(num_chunks)] if metric == progress else None,
            )
        )
        for name, variable in outer_variables.items():
            self.addCollectiveVariable(name, variable)

    def _generateName(self, metric: Metric, name: str, kind: str) -> str:
        if metric not in (progress, deviation):
            raise ValueError(
                "Invalid metric. Use 'cvpack.path.progress' or 'cvpack.path.deviation'."
            )
        if name is None:
            return f"path_{metric.name}_in_{kind}_space"
        return name
