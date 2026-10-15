"""
.. class:: CompositeRMSD
   :platform: Linux, MacOS, Windows
   :synopsis: Deviation of multiple corotating bodies from their reference structures

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import typing as t

from openmm import unit as mmunit

from .base_rmsd import BaseRMSD
from .units import MatrixQuantity, VectorQuantity

# Earlier versions compute wrong forces for groups of different sizes
MIN_OPENMM_CPP_FORCES_VERSION = (0, 2, 0)


def _stub(message: str) -> type:
    class Stub:
        def __init__(self, *_):
            raise ImportError(message)

    return Stub


def _get_force_class() -> type:
    """Return CompositeRMSDForce, or a stub that raises an informative ImportError."""
    requirement = "openmm-cpp-forces 0.2.0 or later (mdtools conda channel or PyPI)"
    try:
        import openmmcppforces  # noqa: PLC0415
    except ImportError:
        return _stub(f"CompositeRMSD requires {requirement}")
    version = getattr(openmmcppforces, "__version__", None)
    numbers = tuple(
        int(part) for part in (version or "").split(".")[:3] if part.isdigit()
    )
    if numbers < MIN_OPENMM_CPP_FORCES_VERSION:
        return _stub(
            f"CompositeRMSD requires {requirement}, because earlier versions compute "
            f"wrong forces, but version {version or 'unknown'} is installed"
        )
    return openmmcppforces.CompositeRMSDForce


CompositeRMSDForce = _get_force_class()


class CompositeRMSD(BaseRMSD, CompositeRMSDForce):
    r"""
    The composite root-mean-square deviation (RMSD) between the current and reference
    coordinates of :math:`m` groups of atoms:

    .. math::

        d_{\rm crms}({\bf r}) = \sqrt{
            \frac{1}{n} \min_{
                \bf q \in \mathbb{R}^4 \atop \|{\bf q}\| = 1
            } \sum_{j=1}^m \sum_{i \in {\bf g}_j} \left\|
                {\bf A}({\bf q})\left({\bf r}_i - {\bf c}_j\right) -
                    {\bf r}_i^{\rm ref} + {\bf c}_j^{\rm ref}
            \right\|^2
        }

    where each group :math:`{\bf g}_j` is a set of :math:`n_j` atom indices,
    :math:`n = \sum_{j=1}^m n_j` is the total number of atoms in these groups,
    :math:`{\bf A}(\bf q)` is the rotation matrix corresponding to a unit quaternion
    :math:`{\bf q}`, :math:`{\bf r}_i` and :math:`{\bf r}_i^{\rm ref}` are the
    positions of atom :math:`i` in the current and reference structures, respectively,
    :math:`{\bf c}_j` is the centroid of group :math:`j`, given by

    .. math::

        {\bf c}_j = \frac{1}{n_j} \sum_{i \in {\bf g}_j} {\bf r}_i

    and :math:`{\bf c}_j^{\rm ref}` is the centroid of the reference structure for
    group :math:`j`, defined analogously to :math:`{\bf c}_j`.

    .. warning::

        To use this class, you must install the `openmm-cpp-forces`_ package,
        version 0.2.0 or later.

    .. _openmm-cpp-forces: https://anaconda.org/mdtools/openmm-cpp-forces

    Parameters
    ----------
    referencePositions
        The reference coordinates, which can be either a coordinate matrix or a mapping
        from atom indices to coordinate vectors. It must contain all atoms in
        ``groups``, and does not need to contain all atoms in the system. See
        ``numAtoms`` below.
    groups
        A sequence of disjoint atom groups. Each group is a sequence of atom indices.
    numAtoms
        The total number of atoms in the system, including those that are not in
        ``groups``. This argument is necessary only if ``referencePositions`` does not
        contain all atoms in the system.
    name
        The name of the collective variable.

    Raises
    ------
    ImportError
        If the `openmm-cpp-forces`_ package (version 0.2.0 or later) is not
        installed.
    ValueError
        If ``groups`` is not a sequence of disjoint atom groups.
    ValueError
        If referencePositions is a dict and numAtoms is not given

    Example
    -------
    >>> import cvpack
    >>> import openmm as mm
    >>> import pytest
    >>> from openmmtools import testsystems
    >>> from openmm import unit
    >>> model = testsystems.HostGuestVacuum()
    >>> host_atoms, guest_atoms = (
    ...     [a.index for a in r.atoms()]
    ...     for r in model.topology.residues()
    ... )
    >>> try:
    ...     composite_rmsd = cvpack.CompositeRMSD(
    ...         model.positions,
    ...         [host_atoms, guest_atoms],
    ...     )
    ... except ImportError:
    ...     pytest.skip("openmm-cpp-forces is not installed")
    >>> composite_rmsd.addToSystem(model.system)
    >>> context = mm.Context(
    ...     model.system,
    ...     mm.VerletIntegrator(1.0 * unit.femtoseconds),
    ...     mm.Platform.getPlatformByName('Reference'),
    ... )
    >>> context.setPositions(model.positions)
    >>> composite_rmsd.getValue(context)
    0.0 nm
    >>> model.positions[guest_atoms] += 1.0 * unit.nanometers
    >>> context.setPositions(model.positions)
    >>> composite_rmsd.getValue(context)
    0.0 nm
    """

    def __init__(
        self,
        referencePositions: MatrixQuantity | dict[int, VectorQuantity],
        groups: t.Iterable[t.Iterable[int]],
        numAtoms: int | None = None,
        name: str = "composite_rmsd",
    ) -> None:
        if numAtoms is None and isinstance(referencePositions, dict):
            raise ValueError("numAtoms is required when referencePositions is a dict")
        num_atoms = numAtoms or len(referencePositions)
        groups = [[int(atom) for atom in group] for group in groups]
        if not all(groups):
            # CompositeRMSDForce would interpret an empty group as all atoms
            raise ValueError("The groups of atoms must not be empty")
        defined_coords = self._getDefinedCoords(referencePositions, sum(groups, []))
        all_coords = self._getAllCoords(defined_coords, num_atoms)
        super().__init__(all_coords)
        for group in groups:
            self.addGroup(group)
        self._registerCV(
            name,
            mmunit.nanometers,
            referencePositions=defined_coords,
            groups=groups,
            numAtoms=num_atoms,
        )


CompositeRMSD.registerTag("!cvpack.CompositeRMSD")
