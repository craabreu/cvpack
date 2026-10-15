"""
.. module:: units
   :platform: Linux, MacOS, Windows
   :synopsis: Units of measurement for CVPack.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import ast
import typing as t
from numbers import Real

import numpy as np
import openmm
from openmm import unit as mmunit

from ..serialization import Serializable

ScalarQuantity = t.Union[mmunit.Quantity, Real]
VectorQuantity = t.Union[mmunit.Quantity, np.ndarray, openmm.Vec3]
MatrixQuantity = t.Union[
    mmunit.Quantity, np.ndarray, t.Sequence[openmm.Vec3], t.Sequence[np.ndarray]
]


class Unit(mmunit.Unit, Serializable, ast.NodeTransformer):
    """
    Extension of the OpenMM Unit class to allow serialization and deserialization.

    Parameters
    ----------
    data
        The data to be used to create the unit.
    """

    def __init__(self, data: str | mmunit.Unit | dict) -> None:
        if isinstance(data, str):
            expression = self.visit(ast.parse(data, mode="eval"))
            code = compile(ast.fix_missing_locations(expression), "", mode="eval")
            data = eval(code)  # pylint: disable=eval-used
        if isinstance(data, mmunit.Unit):
            data = dict(data.iter_base_or_scaled_units())
        super().__init__(data)

    def __repr__(self) -> str:
        return self.get_symbol()

    def __getstate__(self) -> dict[str, str]:
        return {"data": str(self)}

    def __setstate__(self, keywords: dict[str, str]) -> None:
        self.__init__(keywords["data"])

    def visit_Name(  # pylint: disable=invalid-name
        self, node: ast.Name
    ) -> ast.Attribute:
        """
        Visit a Name node and transform it into an Attribute node.

        Parameters
        ----------
        node
            The node to be visited and transformed.
        """
        return ast.Attribute(
            value=ast.Name(id="mmunit", ctx=ast.Load()), attr=node.id, ctx=ast.Load()
        )


Unit.registerTag("!cvpack.Unit")


class Quantity(mmunit.Quantity, Serializable):
    """
    Extension of the OpenMM Quantity class to allow serialization and deserialization.
    """

    def __init__(self, *args: t.Any) -> None:
        if len(args) == 1 and mmunit.is_quantity(args[0]):
            super().__init__(args[0].value_in_unit(args[0].unit), Unit(args[0].unit))
        else:
            super().__init__(*args)

    def __repr__(self):
        return str(self)

    def __getstate__(self) -> dict[str, t.Any]:
        return {"value": self.value, "unit": str(self.unit)}

    def __setstate__(self, keywords: dict[str, t.Any]) -> None:
        self.__init__(keywords["value"], Unit(keywords["unit"]))

    @property
    def value(self) -> t.Any:
        """The value of the quantity."""
        return self._value

    def in_md_units(self) -> t.Any:  # pylint: disable=invalid-name
        """The value of the quantity in MD units."""
        return self.value_in_unit_system(mmunit.md_unit_system)


Quantity.registerTag("!cvpack.Quantity")


def in_md_units(
    quantity: ScalarQuantity | VectorQuantity | MatrixQuantity,
) -> Quantity:
    """
    Return a quantity in the MD unit system (e.g. mass in Da, distance in
    nm, time in ps, temperature in K, energy in kJ/mol, angle in rad).

    Parameters
    ----------
    quantity
        The quantity to be converted.

    Returns
    -------
    Quantity
        The quantity in the MD unit system.
    """
    if mmunit.is_quantity(quantity):
        unit_in_md_system = quantity.unit.in_unit_system(mmunit.md_unit_system)
        if 1 * quantity.unit / unit_in_md_system == 1:
            return Quantity(quantity)
        return Quantity(quantity.in_units_of(unit_in_md_system))
    return quantity * Unit("dimensionless")


def value_in_md_units(
    quantity: ScalarQuantity | VectorQuantity | MatrixQuantity,
) -> t.Any:
    """
    Return the value of a quantity in the MD unit system (e.g. mass in Da, distance in
    nm, time in ps, temperature in K, energy in kJ/mol, angle in rad).

    Parameters
    ----------
    quantity
        The quantity to be converted.

    Returns
    -------
    Any
        The value of the quantity in the MD unit system.

    """
    if mmunit.is_quantity(quantity):
        return quantity.value_in_unit_system(mmunit.md_unit_system)
    return quantity
