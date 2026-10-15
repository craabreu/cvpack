"""
Tests for the conversion of quantities to the MD unit system.
"""

import numpy as np
import pytest
from openmm import unit

import cvpack
from cvpack.units import in_md_units, value_in_md_units

AVOGADRO = 6.02214076e23


@pytest.mark.parametrize(
    "quantity, value, symbol",
    [
        (10 * unit.kilocalorie_per_mole, 41.84, "kJ/mol"),
        (2494.0 * unit.joule / unit.mole, 2.494, "kJ/mol"),
        (1 * unit.kilocalorie_per_mole / unit.angstrom**2, 418.4, "kJ/(nm**2 mol)"),
        (1 * unit.kilojoule_per_mole, 1.0, "kJ/mol"),
        (1 * unit.kilojoule_per_mole / unit.radian**2, 1.0, "kJ/(mol rad**2)"),
        (
            1 * unit.nanometer**2 * unit.dalton / unit.radian**2,
            1.0,
            "nm**2 Da/(rad**2)",
        ),
        (180 * unit.degree, np.pi, "rad"),
        (5 * unit.angstrom, 0.5, "nm"),
        # Leftover moles become Avogadro's number, as in kJ/mol
        (1 * unit.bar, 1e5 * AVOGADRO * 1e-30, "kJ/(nm**3 mol)"),
        (1 * unit.molar, AVOGADRO * 1e-24, "/(nm**3)"),
    ],
)
def test_in_md_units(quantity, value, symbol):
    converted = in_md_units(quantity)
    assert converted.value == pytest.approx(value)
    assert converted.unit.get_symbol() == symbol
    assert value_in_md_units(quantity) == pytest.approx(value)


def test_meta_cv_parameter_in_kcal_per_mol():
    phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
    meta = cvpack.MetaCollectiveVariable(
        "0.5*kappa*phi^2",
        [phi],
        unit.kilojoule_per_mole,
        kappa=1 * unit.kilocalorie_per_mole / unit.radian**2,
    )
    kappa = meta.getParameterDefaultValues()["kappa"]
    assert kappa.value == pytest.approx(4.184)
    assert kappa.unit.get_symbol() == "kJ/(mol rad**2)"


@pytest.mark.parametrize(
    "quantity, value, symbol",
    [
        # A per-particle energy becomes a molar energy
        (1 * unit.joule, AVOGADRO * 1e-3, "kJ/mol"),
        # A leftover inverse mole becomes a per-particle value
        (1 * unit.kilojoule_per_mole / unit.mole, 1 / AVOGADRO, "kJ/mol"),
    ],
)
def test_in_md_units_with_leftover_moles(quantity, value, symbol):
    converted = in_md_units(quantity)
    assert converted.value == pytest.approx(value)
    assert converted.unit.get_symbol() == symbol
