"""
.. module:: units
   :platform: Linux, MacOS, Windows
   :synopsis: Units of measurement for CVPack.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import ast
import functools
import operator
import typing as t
from numbers import Real

import numpy as np
import openmm
from openmm import unit as mmunit

from ..serialization import Serializable

ScalarQuantity = mmunit.Quantity | Real
VectorQuantity = mmunit.Quantity | np.ndarray | openmm.Vec3
MatrixQuantity = (
    mmunit.Quantity | np.ndarray | t.Sequence[openmm.Vec3] | t.Sequence[np.ndarray]
)


def _to_builtin(value: t.Any) -> t.Any:
    """Convert numpy and OpenMM containers to plain Python lists and scalars."""
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    if isinstance(value, openmm.Vec3):
        return [_to_builtin(item) for item in value]
    if type(value) in (list, tuple):
        return type(value)(_to_builtin(item) for item in value)
    return value


_UNIT_OPERATORS = {
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


@functools.cache
def _unit_names() -> dict[t.Any, str]:
    """Map each OpenMM base or scaled unit to an ``openmm.unit`` attribute name."""
    names = {}
    for attribute in dir(mmunit):
        value = getattr(mmunit, attribute)
        if isinstance(value, mmunit.Unit):
            components = list(value.iter_base_or_scaled_units())
            if len(components) == 1 and components[0][1] == 1:
                component = components[0][0]
                # Prefer the attribute named like the unit itself (e.g. kilojoule)
                if component not in names or attribute == component.name:
                    names[component] = attribute
    return names


def _evaluate_unit_node(node: ast.AST) -> t.Any:  # noqa: PLR0911
    """Evaluate a parsed unit expression, allowing only unit names and arithmetic."""
    if isinstance(node, ast.Expression):
        return _evaluate_unit_node(node.body)
    if isinstance(node, ast.Name):
        value = getattr(mmunit, node.id, None)
        if isinstance(value, mmunit.Unit):
            return value
        raise ValueError(f"Unknown unit: {node.id}")
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        operand = _evaluate_unit_node(node.operand)
        if isinstance(operand, int | float):
            return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp) and type(node.op) in _UNIT_OPERATORS:
        left = _evaluate_unit_node(node.left)
        right = _evaluate_unit_node(node.right)
        if isinstance(node.op, ast.Pow) and not (
            isinstance(left, mmunit.Unit) and isinstance(right, int | float)
        ):
            raise ValueError("Only units can be raised to numeric powers")
        return _UNIT_OPERATORS[type(node.op)](left, right)
    raise ValueError(f"Invalid unit expression: {ast.unparse(node)}")


def _parse_unit(expression: str) -> mmunit.Unit:
    """Safely parse a unit expression such as ``kilojoule/(mole*radian**2)``."""
    text = f"1{expression}" if expression.startswith("/") else expression
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Invalid unit expression: {expression}") from error
    result = _evaluate_unit_node(tree)
    if mmunit.is_quantity(result) and result.value_in_unit(result.unit) == 1:
        result = result.unit
    if not isinstance(result, mmunit.Unit):
        raise ValueError(f"Invalid unit expression: {expression}")
    return result


class Unit(mmunit.Unit, Serializable):
    """
    Extension of the OpenMM Unit class to allow serialization and deserialization.

    Parameters
    ----------
    data
        The data to be used to create the unit.
    """

    def __init__(self, data: str | mmunit.Unit | dict) -> None:
        if isinstance(data, str):
            data = _parse_unit(data)
        if isinstance(data, mmunit.Unit):
            data = dict(data.iter_base_or_scaled_units())
        super().__init__(data)

    def __repr__(self) -> str:
        return self.get_symbol()

    def __getstate__(self) -> dict[str, str]:
        names = _unit_names()
        factors = []
        for component, exponent in self.iter_base_or_scaled_units():
            if component not in names:
                return {"data": str(self)}
            power = int(exponent) if float(exponent).is_integer() else exponent
            name = names[component]
            factors.append(name if power == 1 else f"{name}**{power}")
        return {"data": "*".join(factors) or "dimensionless"}

    def __setstate__(self, keywords: dict[str, str]) -> None:
        self.__init__(keywords["data"])


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
        return {
            "value": _to_builtin(self._value),
            "unit": Unit(self.unit).__getstate__()["data"],
        }

    def __setstate__(self, keywords: dict[str, t.Any]) -> None:
        self.__init__(keywords["value"], Unit(keywords["unit"]))

    @property
    def value(self) -> t.Any:
        """The value of the quantity."""
        return self._value

    def in_md_units(self) -> t.Any:
        """The value of the quantity in MD units."""
        return self.value_in_unit_system(mmunit.md_unit_system)


Quantity.registerTag("!cvpack.Quantity")


_MD_BASE_UNITS = {
    "mass": "dalton",
    "length": "nanometer",
    "time": "picosecond",
    "charge": "elementary_charge",
    "temperature": "kelvin",
    "amount": "mole",
    "angle": "radian",
}


def _md_exponents(unit: mmunit.Unit) -> dict[str, float]:
    """Exponents of a unit in the base units of OpenMM's MD unit system."""
    exponents = {}
    for base, exponent in unit.iter_all_base_units():
        name = _MD_BASE_UNITS[base.dimension.name]
        exponents[name] = exponents.get(name, 0) + exponent
    if "dalton" in exponents:
        # openmm.unit defines the dalton as gram/mole
        exponents["mole"] = exponents.get("mole", 0) + exponents["dalton"]
    return {name: value for name, value in exponents.items() if value != 0}


def _md_unit(exponents: dict[str, float]) -> mmunit.Unit:
    """Build a unit from MD exponents, writing Da nm^2/ps^2 as kJ/mol if possible."""
    exponents = dict(exponents)
    result = mmunit.dimensionless
    energy = exponents.get("dalton", 0)
    if energy != 0 and exponents.get("picosecond", 0) == -2 * energy:
        # The time dimension comes only from energies, so express them in kJ/mol
        result = result * mmunit.kilojoule_per_mole**energy
        exponents.pop("dalton")
        exponents.pop("picosecond")
        exponents["nanometer"] = exponents.get("nanometer", 0) - 2 * energy
    for name, exponent in exponents.items():
        if exponent != 0:
            result = result * getattr(mmunit, name) ** exponent
    return result


def in_md_units(
    quantity: ScalarQuantity | VectorQuantity | MatrixQuantity,
) -> Quantity:
    """
    Return a quantity in the MD unit system (e.g. mass in Da, distance in
    nm, time in ps, temperature in K, energy in kJ/mol, angle in rad).

    As in kJ/mol, amounts of substance follow OpenMM's per-mole convention: a quantity
    defined per particle, such as a pressure, is multiplied by Avogadro's number
    (e.g. 1 bar becomes 0.0602 kJ/(mol nm^3)). Energies are written in kJ/mol.

    Parameters
    ----------
    quantity
        The quantity to be converted.

    Returns
    -------
    Quantity
        The quantity in the MD unit system.
    """
    if not mmunit.is_quantity(quantity):
        return quantity * Unit("dimensionless")
    moles = _md_exponents(quantity.unit).get("mole", 0)
    if moles != 0:
        quantity = quantity * mmunit.AVOGADRO_CONSTANT_NA**moles
    exponents = _md_exponents(quantity.unit)
    exponents.pop("mole", None)
    unit_in_md_system = _md_unit(exponents)
    if 1 * quantity.unit / unit_in_md_system == 1:
        # Keep the original spelling of a unit that is already in the MD system
        return Quantity(quantity)
    return Quantity(quantity.in_units_of(unit_in_md_system))


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
        return in_md_units(quantity).value
    return quantity
