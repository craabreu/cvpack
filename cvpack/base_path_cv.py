"""
.. class:: BasePathCV
   :platform: Linux, MacOS, Windows
   :synopsis: A base class for path-related collective variables

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from collections import OrderedDict

import openmm

from .collective_variable import CollectiveVariable
from .path import Metric, deviation, progress


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
        variables: t.Dict[str, CollectiveVariable],
    ) -> None:
        n = len(squared_distances)
        definitions = OrderedDict(
            {f"x{i}": sqdist for i, sqdist in enumerate(squared_distances)}
        )
        definitions["lambda"] = 1 / (2 * sigma**2)
        definitions["xmin0"] = "min(x0,x1)"
        for i in range(n - 2):
            definitions[f"xmin{i + 1}"] = f"min(xmin{i},x{i + 2})"
        for i in range(n):
            definitions[f"w{i}"] = f"exp(lambda*(xmin{n - 2}-x{i}))"
        definitions["wsum"] = "+".join(f"w{i}" for i in range(n))
        expressions = [f"{key}={value}" for key, value in definitions.items()]
        if metric == progress:
            numerator = "+".join(f"{i}*w{i}" for i in range(1, n))
            expressions.append(f"({numerator})/({n - 1}*wsum)")
        else:
            expressions.append(f"xmin{n - 2}-log(wsum)/lambda")
        super().__init__("; ".join(reversed(expressions)))
        for name, variable in variables.items():
            self.addCollectiveVariable(name, variable)

    def _generateName(self, metric: Metric, name: str, kind: str) -> str:
        if metric not in (progress, deviation):
            raise ValueError(
                "Invalid metric. Use 'cvpack.path.progress' or 'cvpack.path.deviation'."
            )
        if name is None:
            return f"path_{metric.name}_in_{kind}_space"
        return name
