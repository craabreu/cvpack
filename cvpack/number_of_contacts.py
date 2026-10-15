"""
.. class:: NumberOfContacts
   :platform: Linux, MacOS, Windows
   :synopsis: The number of contacts between two atom groups

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from numbers import Real

import openmm
from openmm import unit as mmunit

from .collective_variable import CollectiveVariable
from .units import Quantity, ScalarQuantity
from .utils import evaluate_in_context


class NumberOfContacts(CollectiveVariable, openmm.CustomNonbondedForce):
    r"""
    The number of contacts between two atom groups:

    .. math::
        N({\bf r}) = \sum_{i \in {\bf g}_1} \sum_{j \in {\bf g}_2}
                        S\left(\frac{\|{\bf r}_j - {\bf r}_i\|}{r_0}\right)

    where :math:`r_0` is the threshold distance for defining a contact and :math:`S(x)`
    is a step function equal to :math:`1` if a contact is made or equal to :math:`0`
    otherwise. For trajectory analysis, it is fine to make :math:`S(x) = H(1-x)`, where
    `H` is the `Heaviside step function
    <https://en.wikipedia.org/wiki/Heaviside_step_function>`_. For molecular dynamics,
    however, :math:`S(x)` should be a continuous approximation of :math:`H(1-x)` for
    :math:`x \geq 0`. By default :cite:`Iannuzzi_2003`, the following function is used:

    .. math::

        S(x) = \frac{1-x^6}{1-x^{12}} = \frac{1}{1+x^6}

    In fact, a cutoff distance :math:`r_c = x_c r_0` (typically, :math:`x_c = 2`) is
    applied so that :math:`S(x) = 0` for :math:`x \geq x_c`. To avoid discontinuities,
    there is also the option to smoothly switch off :math:`S(x)` starting from
    :math:`r_s = x_s r_0` (typically, :math:`x_s = 1.5`) instead of doing it abruptly at
    :math:`r_c`.

    .. note::

        Atoms are allowed to be in both groups. In this case, self-contacts
        (:math:`i = j`) are ignored and each pair of distinct atoms (:math:`i \neq j`)
        is counted only once.

    .. note::
        Any non-exclusion exceptions involving atoms in :math:`{\bf g}_1` and
        :math:`{\bf g}_2` in the provided :class:`openmm.NonbondedForce` are turned
        into exclusions in this collective variable.

    Parameters
    ----------
    group1
        The indices of the atoms in the first group.
    group2
        The indices of the atoms in the second group.
    nonbondedForce
        The :class:`openmm.NonbondedForce` object from which the total number of
        atoms, the exclusions, and whether to use periodic boundary conditions are
        taken.
    reference
        A dimensionless reference value to which the collective variable should be
        normalized. One can also provide an :OpenMM:`Context` object from which to
        obtain the reference number of contacts.
    stepFunction
        The function "step(1-x)" (for analysis only) or a continuous approximation
        thereof.
    thresholdDistance
        The threshold distance (:math:`r_0`) for considering two atoms as being in
        contact.
    cutoffFactor
        The factor :math:`x_c` that multiplies the threshold distance to define
        the cutoff distance.
    switchFactor
        The factor :math:`x_s` that multiplies the threshold distance to define
        the distance at which the step function starts switching off smoothly.
        If None, it switches off abruptly at the cutoff distance.
    name
        The name of the collective variable.

    Example
    -------
    >>> import cvpack
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.HostGuestExplicit()
    >>> group1, group2 = [], []
    >>> for residue in model.topology.residues():
    ...     if residue.name != "HOH":
    ...         group = group1 if residue.name == "B2" else group2
    ...         group.extend(atom.index for atom in residue.atoms())
    >>> forces = {f.getName(): f for f in model.system.getForces()}
    >>> nc = cvpack.NumberOfContacts(
    ...     group1,
    ...     group2,
    ...     forces["NonbondedForce"],
    ...     stepFunction="step(1-x)",
    ... )
    >>> nc.addToSystem(model.system)
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> integrator = openmm.VerletIntegrator(1.0 * mmunit.femtoseconds)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> nc.getValue(context)
    30.0... dimensionless
    >>> nc_normalized = cvpack.NumberOfContacts(
    ...     group1,
    ...     group2,
    ...     forces["NonbondedForce"],
    ...     stepFunction="step(1-x)",
    ...     reference=context,
    ... )
    >>> nc_normalized.addToSystem(model.system)
    >>> context.reinitialize(preserveState=True)
    >>> nc_normalized.getValue(context)
    0.99999... dimensionless
    """

    def __init__(
        self,
        group1: t.Sequence[int],
        group2: t.Sequence[int],
        nonbondedForce: openmm.NonbondedForce,
        reference: Real | openmm.Context = 1.0,
        stepFunction: str = "1/(1+x^6)",
        thresholdDistance: ScalarQuantity = Quantity(0.3 * mmunit.nanometers),
        cutoffFactor: float = 2.0,
        switchFactor: float | None = 1.5,
        name: str = "number_of_contacts",
    ) -> None:
        num_atoms = nonbondedForce.getNumParticles()
        pbc = nonbondedForce.usesPeriodicBoundaryConditions()
        threshold = thresholdDistance
        if mmunit.is_quantity(threshold):
            threshold = threshold.value_in_unit(mmunit.nanometers)
        expression = f"({stepFunction})/1; x=r/{threshold}"
        super().__init__(expression)
        nonbonded_method = self.CutoffPeriodic if pbc else self.CutoffNonPeriodic
        self.setNonbondedMethod(nonbonded_method)
        for _ in range(num_atoms):
            self.addParticle([])
        for index in range(nonbondedForce.getNumExceptions()):
            i, j, *_ = nonbondedForce.getExceptionParameters(index)
            self.addExclusion(i, j)
        self.setCutoffDistance(cutoffFactor * thresholdDistance)
        use_switching_function = switchFactor is not None
        self.setUseSwitchingFunction(use_switching_function)
        if use_switching_function:
            self.setSwitchingDistance(switchFactor * thresholdDistance)
        self.setUseLongRangeCorrection(False)
        self.addInteractionGroup(group1, group2)
        if isinstance(reference, openmm.Context):
            reference = evaluate_in_context(self, reference)
        self.setEnergyFunction(expression.replace("/1;", f"/{reference};"))
        self._registerCV(
            name,
            mmunit.dimensionless,
            group1=group1,
            group2=group2,
            nonbondedForce=nonbondedForce,
            reference=reference,
            stepFunction=stepFunction,
            thresholdDistance=thresholdDistance,
            cutoffFactor=cutoffFactor,
            switchFactor=switchFactor,
        )


NumberOfContacts.registerTag("!cvpack.NumberOfContacts")
