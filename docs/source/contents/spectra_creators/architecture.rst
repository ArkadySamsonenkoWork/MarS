Internal Architecture of Spectra Creators
=========================================

All spectra creators derive from :class:`BaseSpectra`, which orchestrates five modular stages:

1. **Resonance Solver**: Computes eigenvalues/eigenvectors of :math:`\hat{H}(B)` or :math:`\hat{H}(\omega)`.
2. **Intensity Calculator**: Evaluates transition matrix elements and weights by populations or density matrix.
3. **Linewidth Model**: Computes inhomogeneous broadening from strain tensors and mesh geometry.
4. **Spectral Integrator**: Performs orientation averaging (powder) or single-crystal projection.
5. **Post-Processor**: Applies final Gaussian and Lorentzian broadening

Each stage is replaceable via dependency injection (e.g., custom `intensity_calculator`), enabling advanced extensions without subclassing.

Performance Notes
-----------------

- Use `inference_mode=True` (default) to disable gradient tracking for speed.
- For time-resolved simulations, caching (`recompute_spin_parameters=False`) avoids redundant diagonalization.

Extensibility
-------------

To implement a new creator:
- Subclass :class:`BaseSpectra`
- Override `_init_spectra_processor` to select integration strategy
- Optionally override `_postcompute_batch_data` for custom dynamics