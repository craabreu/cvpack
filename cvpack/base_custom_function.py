"""
.. class:: BaseCustomFunction
   :platform: Linux, MacOS, Windows
   :synopsis: Abstract class for collective variables defined by a custom function

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import typing as t

from .collective_variable import CollectiveVariable
from .units import ScalarQuantity, VectorQuantity, value_in_md_units


class BaseCustomFunction(CollectiveVariable):
    """
    Abstract class for collective variables defined by a custom function.
    """

    def _extractParameters(
        self,
        size: int,
        **parameters: ScalarQuantity | VectorQuantity,
    ) -> tuple[dict[str, ScalarQuantity], dict[str, list[ScalarQuantity]]]:
        global_parameters = {}
        perbond_parameters = {}
        for name, value in parameters.items():
            data = value_in_md_units(value)
            if isinstance(data, t.Iterable):
                data = list(data)
                if len(data) != size:
                    raise ValueError(
                        f"Parameter {name} has {len(data)} values, but there are "
                        f"{size} groups"
                    )
                perbond_parameters[name] = data
            else:
                global_parameters[name] = data
        return global_parameters, perbond_parameters

    def _addParameters(
        self,
        overalls: dict[str, ScalarQuantity],
        perbonds: dict[str, list[ScalarQuantity]],
        groups: list[tuple[int, ...]],
        pbc: bool,
    ) -> None:

        for name, value in overalls.items():
            self.addGlobalParameter(name, value)
        for name in perbonds:
            self.addPerBondParameter(name)
        for group, *values in zip(groups, *perbonds.values()):
            self.addBond(group, values)
        self.setUsesPeriodicBoundaryConditions(pbc)
