DEInteraction
=============

The :class:`mars.spin_system.DEInteraction` is a specialized subclass of :class:`Interaction` designed explicitly for interactions determined not as 3 principal values but D and E paramers: zero-field splitting or dipolar interaction. It parameterizes the traceless tensor using the conventional spectroscopic parameters :math:`D` (axial) and :math:`E` (rhombic).

Difference from General Interaction
-----------------------------------

While a general :class:`mars.spin_system.Interaction` treats :math:`(D_x, D_y, D_z)` as independent, :class:`mars.spin_system.DEInteraction` enforces the physical constraint:

.. math::

   D_x + D_y + D_z = 0,

and uses the standard transformation:

.. math::

   D_x &= -\frac{D}{3} + E, \\
   D_y &= -\frac{D}{3} - E, \\
   D_z &= \frac{2D}{3}.

For the cases when distribution of D and C parameters are zeros ('strain' is absente), DEInteraction can be written via simple interaction (:class:`mars.spin system.Interaction`)
This distinction is critical when modeling **strain**: in :class:`DEInteraction`, strain is applied to :math:`D` and :math:`E` (not to :math:`D_x, D_y, D_z`), preserving the traceless nature and physical meaning of the distribution.


Construction
------------

.. code-block:: python

   # Only D (E = 0)
   zfs1 = DEInteraction(500e6)  # D = 500 MHz

   # D and E
   zfs2 = DEInteraction([500e6, 100e6])  # D = 500 MHz, E = 100 MHz

   # With strain on D and E
   zfs_strained = DEInteraction([500e6, 100e6], strain=[50e6, 10e6])

   # With orientation
   zfs_oriented = DEInteraction([500e6, 100e6], frame=[0.1, 0.2, 0.3])

Internally, strain is correlated via the transformation matrix:

.. math::

   \begin{bmatrix}
   \delta D_x \\ \delta D_y \\ \delta D_z
   \end{bmatrix}
   =
   \begin{bmatrix}
   -1/3 & 1 \\
   -1/3 & -1 \\
   2/3 & 0
   \end{bmatrix}
   \begin{bmatrix}
   \delta D \\ \delta E
   \end{bmatrix},

which is stored in the ``strain_correlation`` attribute.