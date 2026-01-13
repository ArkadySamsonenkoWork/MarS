Interpolation on Spherical Meshes
=================================

MarS supports interpolation of the full resonance triplet—**resonance fields**, **intensities**, and **linewidths**—from a coarse initial orientation grid to a finer evaluation grid. This decouples the expensive Hamiltonian diagonalization (performed on the coarse grid) from the high-resolution spherical integration required for smooth spectra.

Two interpolation strategies are provided:

1. **Nearest Neighbors**
2. **Radial Basis Function (RBF) Interpolation** 

These are implemented in :class:`mars.mesher.NearestNeighborsInterpolator` and :class:`mars.mesher.RBFInterpolator`.

When ``interpolate=True`` in :class:`DelaunayMeshNeighbour`, all three resonance quantities are interpolated onto the extended mesh before triangulation and integration.


Recommendation
--------------

For routine powder simulations, use nearest-neighbor interpolation with  
``initial_grid_frequency=20–30`` and ``interpolation_grid_frequency=60–100``.  


Technical Details
-----------------

**What Is Interpolated?**
~~~~~~~~~~~~~~~~~~~~~~~~~

Given resonance data on the base mesh:
- **Fields** :math:`B_{\text{res}}(\Omega_i)` — resonance magnetic field at orientation :math:`\Omega_i`
- **Intensities** :math:`I(\Omega_i)` — transition intensity
- **Widths** :math:`\Delta B(\Omega_i)` — linewidth

All three tensors are interpolated **independently** to the extended mesh vertices :math:`\{\Omega'_j\}`:

.. math::

   B'_{\text{res}}(\Omega'_j) = \mathcal{I}[B_{\text{res}}](\Omega'_j), \quad
   I'(\Omega'_j) = \mathcal{I}[I](\Omega'_j), \quad
   \Delta B'(\Omega'_j) = \mathcal{I}[\Delta B](\Omega'_j),

where :math:`\mathcal{I}` denotes the chosen interpolator.

After interpolation, values are assigned to triangle vertices via :meth:`mesh.to_delaunay`,
and per-triangle averages of width and intensity are used during integration (see :class:`PowderStationaryProcessing`).

**NearestNeighborsInterpolator**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses a BallTree (scikit-learn) with haversine distance on the sphere. For each extended vertex,
the *k*=4 nearest base vertices are found. Weights are computed as:

.. math::

   w_i = \frac{1 / d_i}{\sum_{j=1}^k 1 / d_j},

with :math:`d_i` the angular distance. The implementation is fully vectorized and GPU-compatible.

**RBFInterpolator**
~~~~~~~~~~~~~~~~~~~

Solves :math:`\mathbf{K} \boldsymbol{\alpha} = \mathbf{f}` for each resonance quantity separately,
where :math:`\mathbf{K}_{mn} = \phi(\| \Omega_m - \Omega_n \|)` is the RBF kernel matrix on base vertices.
Interpolation at extended points uses :math:`\mathbf{f}' = \mathbf{K}_{\text{ext}} \boldsymbol{\alpha}`.

A jitter (1e⁻⁴) stabilizes inversion, and truncated SVD discards singular values <1e⁻¹⁰.
Supported kernels: ``gaussian``, ``multiquadric``, ``inverse_multiquadric``, ``linear``, ``cubic``, ``thin_plate``.

Usage in Spectral Processing
----------------------------

In :class:`PowderStationaryProcessing`, the method :meth:`_transform_data_to_mesh_format` calls
:meth:`mesh(...)` on a stacked tensor of fields, intensities, and widths. The mesh’s internal
interpolator processes all channels simultaneously.

Note: Interpolation is applied **before** triangle averaging. Thus, even with interpolation,
the integrator receives consistent triplets (field, intensity, width) per triangle vertex.

Limitations
-----------

- RBF interpolation is **not differentiable** due to SVD-based pseudo-inversion.
- Interpolation assumes smooth variation; may blur sharp features near level anti-crossings.