"""
.. class:: Utils
   :platform: Linux, MacOS, Windows
   :synopsis: Utility functions and classes for CVpack

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import functools
import inspect
import typing as t

import numpy as np
import openmm
from numpy import typing as npt
from openmm import XmlSerializer
from openmm import app as mmapp
from openmm import unit as mmunit

from .serialization import (
    Serializable,
    SerializableAtom,
    SerializableForce,
    SerializableResidue,
)
from .units import Quantity, Unit, value_in_md_units


def evaluate_in_context(
    forces: openmm.Force | t.Iterable[openmm.Force], context: openmm.Context
) -> float | list[float]:
    """Evaluate the potential energies of OpenMM Forces in a given context.

    Parameters
    ----------
        forces
            The forces to be evaluated.
        context
            The context in which the force will be evaluated.

    Returns
    -------
        float
            The potential energy of the force in the given context.
    """
    is_single = isinstance(forces, openmm.Force)
    if is_single:
        forces = [forces]
    system = openmm.System()
    for _ in range(context.getSystem().getNumParticles()):
        system.addParticle(1.0)
    for i, force in enumerate(forces):
        force_copy = XmlSerializer.deserialize(XmlSerializer.serialize(force))
        force_copy.setForceGroup(i)
        system.addForce(force_copy)
    state = context.getState(getPositions=True)
    context = openmm.Context(system, openmm.VerletIntegrator(1.0))
    context.setPositions(state.getPositions())
    context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    energies = []
    for i in range(len(forces)):
        state = context.getState(getEnergy=True, groups=1 << i)
        energies.append(value_in_md_units(state.getPotentialEnergy()))
    return energies[0] if is_single else tuple(energies)


def convert_to_matrix(array: npt.ArrayLike) -> tuple[np.ndarray, int, int]:
    """Convert a 1D or 2D array-like object to a 2D numpy array.

    Parameters
    ----------
        array : array_like
            The array to be converted.

    Returns
    -------
        numpy.ndarray
            The 2D numpy array.
        int
            The number of rows in the array.
        int
            The number of columns in the array.
    """
    array = np.atleast_2d(array)
    numrows, numcols, *other_dimensions = array.shape
    if other_dimensions:
        raise ValueError("Array-like object cannot have more than two dimensions.")
    return array, numrows, numcols


def get_single_force_state(
    force: openmm.Force,
    context: openmm.Context,
    allowReinitialization: bool = False,
    **kwargs: bool,
) -> openmm.State:
    """
    Get an OpenMM State containing the potential energy and/or force values computed
    from a single force object.

    Parameters
    ----------
    force
        The force object from which the state should be extracted.
    context
        The context from which the state should be extracted.
    allowReinitialization
        If True, the force group of the given force will be temporarily changed to a
        group that is not used by any other force in the system, if necessary.

    Keyword Args
    ------------
    **kwargs
        Additional keyword arguments to be passed to the `getState` method, except for
        the `groups` argument.

    Returns
    -------
    openmm.State
        The state containing the requested values.

    Raises
    ------
    ValueError
        If this force is not present in the given context.
    """
    forces_and_groups = [
        (f, f.getForceGroup()) for f in context.getSystem().getForces()
    ]
    if not any(f.this == force.this for f, _ in forces_and_groups):
        raise RuntimeError("This force is not present in the given context.")
    self_group = force.getForceGroup()
    other_groups = {g for f, g in forces_and_groups if f.this != force.this}
    if self_group not in other_groups:
        return context.getState(groups=1 << self_group, **kwargs)
    if not allowReinitialization:
        raise ValueError("Context reinitialization required, but not allowed.")
    new_group = force._setUnusedForceGroup(context.getSystem())
    context.reinitialize(preserveState=True)
    state = context.getState(groups=1 << new_group, **kwargs)
    force.setForceGroup(self_group)
    context.reinitialize(preserveState=True)
    return state


def compute_effective_mass(
    force: openmm.Force, context: openmm.Context, allowReinitialization: bool = False
) -> float:
    r"""
    Compute the effective mass of an :OpenMM:`Force` at a given :OpenMM:`Context`.

    Parameters
    ----------
    force
        The force object from which the effective mass should be computed
    context
        The context at which the force's effective mass should be evaluated
    allowReinitialization
        If True, the force group of the given force will be temporarily changed to a
        group that is not used by any other force in the system, if necessary.

    Returns
    -------
    float
        The effective mass of the force at the given context
    """
    state = get_single_force_state(
        force, context, allowReinitialization, getForces=True
    )
    get_mass = functools.partial(
        openmm._openmm.System_getParticleMass, context.getSystem()
    )
    force_vectors = state.getForces(asNumpy=True)._value
    squared_forces = np.sum(np.square(force_vectors), axis=1)
    nonzeros = np.nonzero(squared_forces)[0]
    if nonzeros.size == 0:
        return np.inf
    mass_values = np.fromiter(map(get_mass, nonzeros), dtype=np.float64)
    return 1.0 / np.sum(squared_forces[nonzeros] / mass_values)


def preprocess_args(func: t.Callable) -> t.Callable:
    """
    A decorator that converts instances of unserializable classes to their
    serializable counterparts.

    Parameters
    ----------
        func
            The function to be decorated.

    Returns
    -------
        The decorated function.

    Example
    -------
    >>> from cvpack import units, utils
    >>> from openmm import unit as mmunit
    >>> @utils.preprocess_args
    ... def function(data):
    ...     return data
    >>> assert isinstance(function(mmunit.angstrom), units.Unit)
    >>> assert isinstance(function(5 * mmunit.angstrom), units.Quantity)
    >>> seq = [mmunit.angstrom, mmunit.nanometer]
    >>> assert isinstance(function(seq), list)
    >>> assert all(isinstance(item, units.Unit) for item in function(seq))
    >>> dct = {"length": 3 * mmunit.angstrom, "time": 2 * mmunit.picosecond}
    >>> assert isinstance(function(dct), dict)
    >>> assert all(isinstance(item, units.Quantity) for item in function(dct).values())
    """
    signature = inspect.signature(func)

    def convert(data: t.Any) -> t.Any:  # noqa: PLR0911
        if isinstance(data, np.integer):
            return int(data)
        if isinstance(data, np.floating):
            return float(data)
        if isinstance(data, mmunit.Quantity):
            return Quantity(data)
        if isinstance(data, mmunit.Unit):
            return Unit(data)
        if isinstance(data, mmapp.Atom):
            return SerializableAtom(data)
        if isinstance(data, mmapp.Residue):
            return SerializableResidue(data)
        if isinstance(data, openmm.Force) and not isinstance(data, Serializable):
            return SerializableForce(data)
        if isinstance(data, t.Sequence) and not isinstance(data, str):
            return type(data)(map(convert, data))
        if isinstance(data, dict):
            return type(data)((key, convert(value)) for key, value in data.items())
        return data

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        for name, data in bound.arguments.items():
            bound.arguments[name] = convert(data)
        return func(*bound.args, **bound.kwargs)

    return wrapper
