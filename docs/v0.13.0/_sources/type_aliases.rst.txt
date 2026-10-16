Type Aliases
============

Arguments with physical dimensions accept either OpenMM quantities or plain numbers.
Plain numbers are interpreted in the MD unit system (nm, ps, kJ/mol, rad, Da, K).

.. currentmodule:: cvpack.units

.. py:data:: ScalarQuantity

    A scalar: an :class:`openmm.unit.Quantity` or a real number.

.. py:data:: VectorQuantity

    A vector: an :class:`openmm.unit.Quantity`, a NumPy array, or an :class:`openmm.Vec3`.

.. py:data:: MatrixQuantity

    A list of vectors, such as particle positions: an :class:`openmm.unit.Quantity`,
    a two-dimensional NumPy array, or a sequence of :class:`openmm.Vec3` objects or
    NumPy arrays.

Arrays of indices accept any :data:`numpy.typing.ArrayLike`, such as a list, a tuple,
or a NumPy array.
