"""
.. class:: BaseCustomFunction
   :platform: Linux, MacOS, Windows
   :synopsis: Abstract class for collective variables defined by a custom function

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

from openmm import unit as mmunit

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
        for name, data in parameters.items():
            data = value_in_md_units(data)
            if isinstance(data, t.Iterable):
                perbond_parameters[name] = [data[i] for i in range(size)]
            else:
                global_parameters[name] = data
        return global_parameters, perbond_parameters

    def _addParameters(
        self,
        overalls: dict[str, ScalarQuantity],
        perbonds: dict[str, list[ScalarQuantity]],
        groups: list[tuple[int, ...]],
        pbc: bool,
        unit: mmunit.Unit,
    ) -> None:
        # pylint: disable=no-member
        for name, value in overalls.items():
            self.addGlobalParameter(name, value)
        for name in perbonds:
            self.addPerBondParameter(name)
        for group, *values in zip(groups, *perbonds.values()):
            self.addBond(group, values)
        self.setUsesPeriodicBoundaryConditions(pbc)
        if (1 * unit).value_in_unit_system(mmunit.md_unit_system) != 1:
            raise ValueError(f"Unit {unit} is not compatible with the MD unit system.")
