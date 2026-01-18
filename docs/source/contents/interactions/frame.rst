Frame and Euler Angles
======================

In MarS, the orientation of any interaction tensor relative to the laboratory frame is defined by a *rotation*, specified either as Euler angles or a full rotation matrix.

Euler Angle Convention
----------------------

MarS uses the *ZYZ' (proper Euler) convention* with angles :math:`(\alpha, \beta, \gamma)` in *radians*, applied in the following order:

1. Rotation by :math:`\alpha` around the **lab Z-axis**,
2. Rotation by :math:`\beta` around the **new Y'-axis**,
3. Rotation by :math:`\gamma` around the **new Z''-axis**.

The total rotation matrix is:

.. math::

   \mathbf{R}(\alpha, \beta, \gamma) = \mathbf{R}_z(\gamma) \mathbf{R}_y(\beta) \mathbf{R}_z(\alpha).

This is the standard convention in magnetic resonance for describing molecular orientations in powders or single crystals.

Usage
-----

When constructing an :class:`mars.spin_system.Interaction` or :class:`mars.spin_system.DEInteraction`, the ``frame`` argument accepts:

- ``None`` → identity (tensor aligned with lab frame),
- A sequence ``[α, β, γ]`` (in radians),
- A 3×3 rotation matrix (torch.Tensor).

Example:

.. code-block:: python

   # Rotate g-tensor by 40° around y, then 30° around new z
   import math
   g = Interaction((2.0, 2.0, 2.1), frame=[0.0, math.radians(40), math.radians(30)])


Note: The same convention is used in EasySpin and many quantum chemistry packages.