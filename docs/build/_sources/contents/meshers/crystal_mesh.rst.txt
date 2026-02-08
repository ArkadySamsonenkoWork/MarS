Crystal Mesh: CrystalMesh
=========================

The :class:`mars.mesher.general_mesh.CrystalMesh` class represents oriented samples.
Instead of averaging over orientations, it evaluates the spectrum along one or more fixed
crystallographic directions specified by Euler angles.

Key Features
------------

- Accepts a tensor of Euler angles (in radians) of shape ``(..., 3)``.
- Converts Euler angles to rotation matrices using a specified convention (default: ZYZ).

Usage
-----

.. code-block:: python

   euler = torch.tensor([[0.0, 0.0, 0.0],    # z-axis
                         [np.pi/2, 0.0, 0.0], # x-axis
                         [np.pi/2, np.pi/2, 0.0]]) # y-axis
   mesh = mesher.CrystalMesh(euler_angles=euler, device=device)

Each row defines a lab-frame orientation of the crystal. The total signal is the average signal from 3 crystals.
