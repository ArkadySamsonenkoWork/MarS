Powder Mesh: DelaunayMeshNeighbour
==================================

The :class:`mars.mesher.delanay_neigbour.DelaunayMeshNeighbour` class implements a spherical triangulation
based on an adaptive grid in spherical coordinates (θ, φ). It is designed for disordered
(powder) samples where all orientations are equally probable.

Key Features
------------

- Supports full-sphere or restricted φ ranges via ``phi_limits``.
- Optional interpolation from a coarse initial grid to a finer evaluation grid.
- Returns rotation matrices that map the laboratory z-axis to each sampled direction.
- Provides spherical triangle areas for proper integration weighting.

Usage
-----

Typical initialization:

.. code-block:: python

   mesh = mesher.DelaunayMeshNeighbour(
       initial_grid_frequency=30,
       interpolate=True,
       interpolation_grid_frequency=60,
       phi_limits=(0, np.pi),  # half-sphere due to symmetry
       device=device,
       dtype=torch.float64
   )

The mesh is used internally by :class:`mars.spin_model.MultiOrientedSample` to average
the spin Hamiltonian over orientations.