Interaction
===========

The :class:`mars.spin_system.Interaction` class is the general-purpose container for symmetric second-rank tensor interactions in EPR spectroscopy. It supports isotropic, axial, and orthorhombic symmetry and can be rotated into an arbitrary molecular frame via Euler angles or a rotation matrix.

Mathematical Form
-----------------

An interaction tensor **T** is defined by its principal values :math:`(T_x, T_y, T_z)` in its intrinsic frame. In the laboratory frame, it becomes:

.. math::

   \mathbf{T}^{\text{lab}} = \mathbf{R} \cdot \text{diag}(T_x, T_y, T_z) \cdot \mathbf{R}^\top,

where :math:`\mathbf{R}` is the rotation matrix derived from the specified Euler angles (ZYZ' convention).


Construction
------------

.. code-block:: python

   # Isotropic (scalar)
   J = Interaction(1e9)  # 1 GHz isotropic exchange

   # Axial: [A_perp, A_par]
   A_axial = Interaction([20e6, 70e6])  # MHz

   # Orthorhombic: [A_x, A_y, A_z]
   A_ortho = Interaction([20e6, 25e6, 70e6])

   # With frame rotation (Euler angles in radians)
   g_rot = Interaction((2.02, 2.04, 2.06), frame=[0.0, 0.7, 0.0])

   # With strain (distribution of principal values)
   g_strained = Interaction((2.02, 2.04, 2.06), strain=[0.01, 0.01, 0.02])

Note: All values must be provided in SI-compatible units (Hz for couplings, dimensionless for g-tensors).

Addition
--------

Two :class:`Interaction` instances can be added. If their frames match, components are summed directly. Otherwise, tensors are rotated to the lab frame, summed, and diagonalized to extract new principal values and frame.