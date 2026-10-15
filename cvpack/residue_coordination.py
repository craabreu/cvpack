"""
.. class:: ResidueCoordination
   :platform: Linux, MacOS, Windows
   :synopsis: The number of contacts between two groups of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm
from openmm import unit as mmunit
from openmm.app.element import hydrogen
from openmm.app.topology import Residue

from .collective_variable import CollectiveVariable
from .units import Quantity, ScalarQuantity, value_in_md_units


class ResidueCoordination(CollectiveVariable, openmm.CustomCentroidBondForce):
    r"""
    The number of contacts between two disjoint groups of residues:

    .. math::
        N({\bf r}) = \sum_{i \in {\bf G}_1} \sum_{j \in {\bf G}_2} S\left(
            \frac{\|{\bf R}_j({\bf r}) - {\bf R}_i({\bf r})\|}{r_0}
        \right)

    where :math:`{\bf G}_1` and :math:`{\bf G}_2` are the two groups of residues,
    :math:`{\bf R}_i` is the centroid of the residue :math:`i`, :math:`r_0` is the
    threshold distance for defining a contact and :math:`S(x)` is a step function equal
    to :math:`1` if a contact is made or equal to :math:`0` otherwise. For trajectory
    analysis, it is fine to make :math:`S(x) = H(1-x)`, where `H` is the `Heaviside step
    function <https://en.wikipedia.org/wiki/Heaviside_step_function>`_. For molecular
    dynamics, however, :math:`S(x)` should be a continuous approximation of
    :math:`H(1-x)` for :math:`x \geq 0`. By default :cite:`Iannuzzi_2003`, the
    following function is used:

    .. math::

        S(x) = \frac{1-x^6}{1-x^{12}} = \frac{1}{1+x^6}

    Parameters
    ----------
    residueGroup1
        The residues in the first group.
    residueGroup2
        The residues in the second group.
    pbc
        Whether to use periodic boundary conditions in distance calculations. This
        should be ``True`` for periodic systems.
    stepFunction
        The function "step(1-x)" (for analysis only) or a continuous approximation
        thereof.
    thresholdDistance
        The threshold distance (:math:`r_0`) for considering two residues as being
        in contact.
    normalize
        Whether the number of contacts should be normalized by the total number of
        possible contacts.
    weighByMass
        Whether the centroid of each residue should be weighted by the mass of the
        atoms in the residue.
    includeHydrogens
        Whether hydrogen atoms should be included in the centroid calculations.
    name
        The name of the collective variable.

    Raises
    ------
    ValueError
        If the two groups of residues are not disjoint

    Example
    -------
    >>> import cvpack
    >>> import itertools as it
    >>> import openmm
    >>> from openmm import app, unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.LysozymeImplicit()
    >>> group1 = list(it.islice(model.topology.residues(), 125, 142))
    >>> print(*[r.name for r in group1])  # doctest: +ELLIPSIS
    TRP ASP GLU ... ASN GLN THR
    >>> group2 = list(it.islice(model.topology.residues(), 142, 156))
    >>> print(*[r.name for r in group2])  # doctest: +ELLIPSIS
    PRO ASN ARG ... ARG THR GLY
    >>> residue_coordination = cvpack.ResidueCoordination(
    ...     group1,
    ...     group2,
    ...     stepFunction="step(1-x)",
    ...     thresholdDistance=0.6*unit.nanometers,
    ... )
    >>> residue_coordination.addToSystem(model.system)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> context = openmm.Context(
    ...     model.system, openmm.CustomIntegrator(0), platform
    ... )
    >>> context.setPositions(model.positions)
    >>> residue_coordination.getValue(context)
    7.0 dimensionless
    >>> residue_coordination.setReferenceValue(7 * unit.dimensionless)
    >>> context.reinitialize(preserveState=True)
    >>> residue_coordination.getValue(context)
    0.99999... dimensionless
    """

    def __init__(
        self,
        residueGroup1: t.Iterable[Residue],
        residueGroup2: t.Iterable[Residue],
        pbc: bool = False,
        stepFunction: str = "1/(1+x^6)",
        thresholdDistance: ScalarQuantity = Quantity(1.0 * mmunit.nanometers),
        normalize: bool = False,
        weighByMass: bool = True,
        includeHydrogens: bool = True,
        name: str = "residue_coordination",
    ) -> None:
        residueGroup1 = list(residueGroup1)
        residueGroup2 = list(residueGroup2)
        nr1 = len(residueGroup1)
        nr2 = len(residueGroup2)
        self._ref_val = nr1 * nr2 if normalize else 1.0
        threshold = thresholdDistance
        if mmunit.is_quantity(threshold):
            threshold = threshold.value_in_unit(mmunit.nanometers)
        expression = (
            f"({stepFunction})/refval"
            f"; x=distance(g1,g2)/{threshold}"
            f"; refval={self._ref_val}"
        )
        super().__init__(2, expression)
        self.setUsesPeriodicBoundaryConditions(pbc)
        for res in residueGroup1 + residueGroup2:
            atoms = [
                atom.index
                for atom in res.atoms()
                if includeHydrogens or atom.element is not hydrogen
            ]
            self.addGroup(atoms, *([] if weighByMass else [[1] * len(atoms)]))
        for idx1 in range(nr1):
            for idx2 in range(nr1, nr1 + nr2):
                self.addBond([idx1, idx2], [])
        self._registerCV(
            name,
            mmunit.dimensionless,
            residueGroup1=residueGroup1,
            residueGroup2=residueGroup2,
            pbc=pbc,
            stepFunction=stepFunction,
            thresholdDistance=thresholdDistance,
            normalize=normalize,
            weighByMass=weighByMass,
            includeHydrogens=includeHydrogens,
        )

    def getReferenceValue(self) -> mmunit.Quantity:
        """
        Get the reference value used for normalizing the residue coordination.

        Returns
        -------
        mmunit.Quantity
            The reference value.
        """
        return self._ref_val * self.getUnit()

    def setReferenceValue(self, value: ScalarQuantity) -> None:
        """
        Set the reference value used for normalizing the residue coordination.

        Parameters
        ----------
            value
                The reference value.
        """
        expression = self.getEnergyFunction()
        value = value_in_md_units(value)
        self.setEnergyFunction(
            expression.replace(f"refval={self._ref_val}", f"refval={value}")
        )
        self._ref_val = value


ResidueCoordination.registerTag("!cvpack.ResidueCoordination")
