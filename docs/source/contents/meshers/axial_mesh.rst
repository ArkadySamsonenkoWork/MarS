Axial Mesh: AxialMeshNeighbour
==============================

The :class:`mars.mesher.AxialMeshNeighbour` class is optimized for systems with axial (cylindrical)
symmetry, where physical observables depend only on the polar angle θ and are invariant under
azimuthal rotation (φ).

Key Features
------------

- Samples only the polar angle θ ∈ [0, π/2] (exploiting symmetry).
- Constructs line segments for integration.
- No interpolation support (currently).
- Generates rotation matrices about the y-axis only: R = R_y(θ).

Usage
-----

.. code-block:: python

   mesh = mesher.AxialMeshNeighbour(
       initial_grid_frequency=50,
       device=device,
       dtype=torch.float64
   )

This mesh is automatically selected when the spin system has axial symmetry and no φ-dependent
interactions (e.g., axial g-tensors and D-tensors with aligned frames).

Mathematical Notes
------------------

Integration weight for a segment [θ_i, θ_{i+1}] is proportional to  
2π(cos θ_i − cos θ_{i+1}), corresponding to the surface area of a spherical zone.