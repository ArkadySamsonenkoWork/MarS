Interpolation on Spherical Meshes
=================================

MarS supports interpolation of the three resonnce data: **resonance fields**, **intensities**, and **linewidths** - from a coarse initial orientation grid to a finer evaluation grid.
This decouples the expensive Hamiltonian diagonalization (performed on the coarse grid) from the high-resolution spherical integration required for smooth spectra.

Two interpolation strategies are provided:

1. **Nearest Neighbors**
2. **Radial Basis Function (RBF) Interpolation** 

These are implemented in :class:`mars.mesher.delanay_neigbour.NearestNeighborsInterpolator` and :class:`mars.mesher.delanay_neigbour.RBFInterpolator`.

When ``interpolate=True`` in :class:`DelaunayMeshNeighbour`, all three resonance quantities are interpolated onto the extended mesh before triangulation and integration.

Technical Details
-----------------

Interpolation Procedures
~~~~~~~~~~~~~~~~~~~~~~~~

Given resonance data on the base mesh:

- **Fields** :math:`B_{\text{res}}(\omega_i)` - resonance magnetic field at orientation :math:`\omega_i`

- **Intensities** :math:`I(\omega_i)` - transition intensity

- **Widths** :math:`\Delta B(\omega_i)` - linewidth

All three tensors are interpolated independently to the extended mesh vertices :math:`\{\omega'_j\}`:

.. math::

   B'_{\text{res}}(\omega'_j) = \mathcal{I}[B_{\text{res}}](\omega'_j), \quad
   I'(\omega'_j) = \mathcal{I}[I](\omega'_j), \quad
   \Delta B'(\omega'_j) = \mathcal{I}[\Delta B](\omega'_j),

where :math:`\mathcal{I}` denotes the chosen interpolator.

**NearestNeighborsInterpolator**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses a BallTree with haversine distance on the sphere. For each extended vertex,
the *k=4* nearest base vertices are found. Weights are computed as:

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
rossings.