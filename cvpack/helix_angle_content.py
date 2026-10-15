"""
.. class:: HelixAngleContent
   :platform: Linux, MacOS, Windows
   :synopsis: Alpha-helix angle content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm
from openmm import app as mmapp
from openmm import unit as mmunit

from .collective_variable import CollectiveVariable
from .units import Quantity, ScalarQuantity


class HelixAngleContent(CollectiveVariable, openmm.CustomAngleForce):
    r"""
    The alpha-helix angle content of a sequence of `n` residues:

    .. math::

        \alpha_{\theta}({\bf r}) = \sum_{i=2}^{n-1} B_m\left(
            \frac{\theta^\alpha_i({\bf r}) - \theta_{\rm ref}}{\theta_{\rm tol}}
        \right)

    where :math:`\theta^\alpha_i` is the angle formed by the alpha-carbon atoms of
    residues :math:`i-1`, :math:`i`, and :math:`i+1`, :math:`\theta_{\rm ref}` is its
    reference value in an alpha helix, and :math:`\theta_{\rm tol}` is the threshold
    tolerance around this reference. :math:`B_m(x)` is a smooth boxcar function given by

    .. math::
        B_m(x) = \frac{1}{1 + x^{2m}}

    where :math:`m` is an integer parameter that controls its steepness. Note that
    :math:`x` needs to be elevated to an even power for :math:`B_m(x)` to be an even
    function.

    Optionally, this collective variable can be normalized to the range :math:`[0, 1]`.

    .. note::

        The residues must be a contiguous sequence from a single chain, ordered from the
        N- to the C-terminus. Due to an OpenMM limitation, the maximum supported number
        of residues is 37.

    Parameters
    ----------
    residues
        The residues in the sequence.
    pbc
        Whether to use periodic boundary conditions.
    thetaReference
        The reference value of the
        :math:`{\rm C}_\alpha{\rm -C}_\alpha{\rm -C}_\alpha` angle in an alpha
        helix.
    tolerance
        The threshold tolerance around the reference values.
    halfExponent
        The parameter :math:`m` of the boxcar function.
    normalize
        Whether to normalize the collective variable to the range :math:`[0, 1]`.
    name
        The name of the collective variable.

    Raises
    ------
    ValueError
        If some residue does not contain a :math:`{\rm C}_\alpha` atom

    Example
    -------
    >>> import cvpack
    >>> import openmm
    >>> from openmm import app, unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.LysozymeImplicit()
    >>> residues = list(model.topology.residues())[59:80]
    >>> print(*[r.name for r in residues])  # doctest: +ELLIPSIS
    LYS ASP GLU ... ILE LEU ARG
    >>> helix_content = cvpack.HelixAngleContent(residues)
    >>> helix_content.addToSystem(model.system)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> integrator = openmm.VerletIntegrator(0)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> helix_content.getValue(context)
    18.7605... dimensionless
    """

    def __init__(
        self,
        residues: t.Sequence[mmapp.topology.Residue],
        pbc: bool = False,
        thetaReference: ScalarQuantity = Quantity(88 * mmunit.degrees),
        tolerance: ScalarQuantity = Quantity(15 * mmunit.degrees),
        halfExponent: int = 3,
        normalize: bool = False,
        name: str = "helix_angle_content",
    ) -> None:
        def find_alpha_carbon(residue: mmapp.topology.Residue) -> int:
            for atom in residue.atoms():
                if atom.name == "CA":
                    return atom.index
            raise ValueError(
                f"Could not find atom CA in residue {residue.name}{residue.id}"
            )

        num_angles = len(residues) - 2
        numerator = 1 / num_angles if normalize else 1
        theta0, tol = thetaReference, tolerance
        if mmunit.is_quantity(theta0):
            theta0 = theta0.value_in_unit(mmunit.radians)
        if mmunit.is_quantity(tol):
            tol = tol.value_in_unit(mmunit.radians)
        super().__init__(
            f"{numerator}/(1+x^{2 * halfExponent}); x=(theta-{theta0})/{tol}"
        )
        for i in range(1, len(residues) - 1):
            self.addAngle(
                find_alpha_carbon(residues[i - 1]),
                find_alpha_carbon(residues[i]),
                find_alpha_carbon(residues[i + 1]),
                [],
            )
        self.setUsesPeriodicBoundaryConditions(pbc)
        self._registerCV(
            name,
            mmunit.dimensionless,
            residues=residues,
            pbc=pbc,
            thetaReference=thetaReference,
            tolerance=tolerance,
            halfExponent=halfExponent,
            normalize=normalize,
        )


HelixAngleContent.registerTag("!cvpack.HelixAngleContent")
