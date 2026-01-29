.. _propagator-computations:

Propagator-Based Density Matrix Evolution
===========================================

Overview
--------

The class:`mars.population.stationary.`PropagatorDensityPopulator` class computes time-resolved EPR signals by explicitly calculating the full time-evolution propagator U(t, 0) of the density matrix.
This method imposes no approximations on the g-tensor, zero-field splitting, or relaxation superoperator, making it the most general approach for time-dependent density matrix evolution.

Theory
------

Evolution Propagator
~~~~~~~~~~~~~~~~~~~~

The density matrix at time t is related to its initial value through the propagator:

.. math::

   \hat{\rho}(t) = \hat{G}(t, 0) \hat{\rho}(0)

where the propagator satisfies:

.. math::

   \frac{d\hat{G}(t, 0)}{dt} = (-i\hat{H}(t) + \hat{R})\hat{G}(t, 0)

with initial condition G(0, 0) = I (identity superoperator).

In Liouville space, the Hamiltonian superoperator is:

.. math::

   \hat{H} = H \otimes I - I \otimes H

where H is the spin Hamiltonian in Hilbert space and I is the identity matrix.

Floquet Theory for Periodic Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The time-dependent Hamiltonian H(t) is periodic with period T = 2π/w, where w is the microwave frequency (9-10 GHz in X-band, giving T ~ 0.1 ns). This periodicity enables efficient long-time propagation using Floquet theory.

For any time t = kT + τ where k is an integer and 0 ≤ τ < T:

.. math::

   \hat{G}(t, 0) = [\hat{G}(T, 0)]^k \hat{G}(\tau, 0)

Thus the propagator need only be computed over one microwave period [0, T], then raised to the k-th power for longer times.

Notification
~~~~~~~~~~~~
Populator uses Runge–Kutta method to fine :math:\hat{G}(T, 0). It requires the parameter of populator n_steps (see :meth:`mars.population.stationary.PropagatorDensityPopulator.__init__`)
This parameter is crucial and for some applications it should be increased to avoid numerical instability.

Signal Detection
~~~~~~~~~~~~~~~~

The observable EPR signal is proportional to the transverse magnetization:

.. math::

   S(t) \propto \text{Tr}[\hat{G}_x \hat{\rho}(t)] = \text{Tr}[\hat{G}_x \hat{G}(t,0) \hat{\rho}(0)]

where G_x is the x-component of the total spin angular momentum operator (proportional to the transverse magnetization).

For phase-sensitive detection at the microwave frequency:

.. math::

   I(t) = \int_0^t \text{Tr}[\hat{G}(\tau) \hat{\rho}(\tau)] \sin(\omega\tau) d\tau

Computational Implementation
----------------------------

Efficient Propagator Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than storing the full propagator G(τ, 0) for all τ ∈ [0, T], only two quantities are computed during integration over one period:

1. **Full-period propagator**: G(T, 0)
2. **Phase-weighted integral**:

   .. math::

      \hat{I} = \int_0^T \hat{G}(\tau, 0) \sin(\omega\tau) d\tau

These are sufficient to compute the detected signal at any time t.

Matrix Power via Diagonalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compute [G(T, 0)]^k efficiently, the propagator is diagonalized once:

.. math::

   \hat{G}(T, 0) = S \Lambda S^{-1}

Then:

.. math::

   [\hat{G}(T, 0)]^k = S \Lambda^k S^{-1}

where Λ^k is a diagonal matrix easily obtained by raising eigenvalues to the k-th power.

This avoids repeated matrix multiplications and scales linearly with k rather than as k times the cost of matrix multiplication.

Time Discretization
~~~~~~~~~~~~~~~~~~~

Since the microwave period T ~ 0.1 ns is much shorter than typical detection timescales (hundreds of nanoseconds or longer), time points are rounded to the nearest integer multiple of T. This introduces negligible error while greatly simplifying calculations.

Relaxation Parameter Constraints
---------------------------------

For the propagator method, all relaxation parameters in the Context *must be time-independent*.
The Floquet approach relies on the periodicity of the evolution operator. If the relaxation superoperator R depends on time (e.g., due to temperature changes), the propagator loses its periodic structure and cannot be efficiently computed via [G(T, 0)]^k.

If relaxation parameters vary with time, use the kinetic approach or RWA with adaptive ODE integration instead.

Powder Averaging
----------------

For disordered samples (powders, frozen solutions), the spectrum must be averaged over all molecular orientations. Unlike the RWA, the propagator method supports fully anisotropic g-tensors, so the γ Euler angle affects both resonance frequencies and intensities.

The averaging over γ is performed by computing two orthogonal polarizations of the microwave field:

.. math::

   \langle I \rangle_\gamma = \frac{1}{2}[\text{Tr}(\hat{G}_x \hat{\rho}) + \text{Tr}(\hat{G}_y \hat{\rho})]

This accounts for the rotation of the molecular frame around the applied field direction.

The spatial integration over (α, β) uses the same triangular discretization and interpolation as other methods.

Advantages
----------

The propagator method:

* Supports arbitrary g-tensor anisotropy
* Handles any zero-field splitting tensor
* Allows general relaxation superoperators (no secular approximation)
* Includes coherence-population coupling if present
* Provides numerically exact evolution

Computational Cost
------------------

This method is more demanding than RWA because:

* Propagator dimension is N^2 × N^2 (versus N^2 for density vector evolution)
* Integration must be performed over one microwave period


Applicability
-------------

Use the propagator method when:

* g-tensor anisotropy is significant (transition metals, high-field EPR)
* Zero-field splitting has arbitrary orientation
* Non-secular relaxation is important
* Coherence-population coupling cannot be neglected
* RWA assumptions are violated

The propagator method is essential for:

* Single-molecule magnets
* High-spin metal complexes
* Strongly coupled radical pairs with anisotropic interactions

For simpler systems where RWA is valid, use class:`mars.population.stationary.StationaryPopulator` for faster computation.