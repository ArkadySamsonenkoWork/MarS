Rotating Wave Approximation
============================

Overview
--------

The class:`mars.population.stationary.StationaryPopulator` computes time-dependent EPR signals using the full density matrix formalism under the Rotating Wave Approximation (RWA).
This approach efficiently models coherent evolution and relaxation when specific physical constraints are satisfied. This is default method for class:`mars.spectra_manager.spectra_manager.DensityTimeSpectra'

Theory
------

Liouville-von Neumann Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The density matrix ρ(t) evolves according to:

.. math::

   \frac{d\rho}{dt} = -i[H, \rho] + \hat{R}[\rho]

where:

* **H**: Spin Hamiltonian
* **R**: Relaxation superoperator describing decoherence and population transfer

To solve this numerically, the equation is transformed into Liouville space where the density matrix becomes a vector of dimension N² and operators become N²×N² superoperators:

.. math::

   \frac{d\hat{\rho}}{dt} = (-i\hat{H} + \hat{R})\hat{\rho}

where:

.. math::

   \hat{H} = H \otimes I - I \otimes H

Rotating Frame Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For time-dependent Hamiltonians oscillating at the spectrometer frequency ω, the RWA eliminates rapidly oscillating terms by transforming to a rotating reference frame:

.. math::

   \tilde{\rho} = e^{i\omega S_z t} \rho e^{-i\omega S_z t}

In the rotating frame, the effective Hamiltonian becomes time-independent:

.. math::

   H_{\text{eff}} = F + g\mu_B(B_0 - \omega/g\mu_B)S_z + g\mu_B B_1 S_x

where:

* **F**: Zero-field Hamiltonian (exchange, dipolar, zero-field splitting)
* **B₀**: Static magnetic field
* **B₁**: Microwave field amplitude
* **ω**: Spectrometer frequency

Constraints and Limitations
----------------------------

In MarS, the RWA imposes strict requirements on the spin system:

Isotropic g-Tensor
~~~~~~~~~~~~~~~~~~

The Zeeman operators must be proportional to spin operators:

.. math::

   G_x = g\mu_B S_x, \quad G_y = g\mu_B S_y, \quad G_z = g\mu_B S_z

This requires an isotropic g-factor. Systems with anisotropic g-tensors violate the commutation relations:

.. math::

   [G_x, G_y] = ig\mu_B G_z

that are essential for the rotating frame transformation.

**Why this occurs**: With anisotropic g-tensors, the Zeeman interaction becomes direction-dependent, and the angular momentum algebra that underlies the RWA breaks down.

Circular Polarization
~~~~~~~~~~~~~~~~~~~~~

The oscillating magnetic field must be circularly polarized:

.. math::

   B_1(t) = B_1[\cos(\omega t)\hat{x} + \sin(\omega t)\hat{y}]

Linear polarization introduces counter-rotating components that are not eliminated by RWA.

**Why this occurs**: Linear polarization contains both co-rotating and counter-rotating components. The RWA discards the counter-rotating terms, which is valid only when they oscillate at ~2ω and can be neglected.

Commutation with Zero-Field Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The zero-field part of the Hamiltonian must commute with S_z:

.. math::

   [F, S_z] = 0

**Why this occurs**: The rotating frame transformation exp(iωS_z t) must leave F invariant. If F does not commute with S_z, additional time-dependent terms appear that cannot be eliminated.

This is automatically satisfied for:

* Isotropic exchange interactions
* Axial zero-field splitting with quantization axis along z
* Small spin-spin interactions in the secular approximation

This is violated for:

* Non-axial zero-field splitting with arbitrary orientation
* Anisotropic exchange or dipolar interactions not aligned with the field

Relaxation Superoperator Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The relaxation superoperator R_{ijkl} (coupling density matrix elements ρ_{ij} and ρ_{kl}) must satisfy:

.. math::

   R_{ijkl} \neq 0 \quad \text{only if} \quad i - j = k - l

This constraint arises because the rotating frame transformation requires R to commute with S_z in Liouville space.

**Allowed processes**:

* Population transfer: i = j, k = l (including pure decay i = j = k = l)
* Dephasing of coherences: i = k, j = l

**Forbidden processes**:

* Coherence-population coupling: mixing off-diagonal and diagonal density matrix elements with i - j ≠ k - l

**Why this occurs**: The RWA assumes that coherences oscillating at different frequencies do not mix with populations. Relaxation mechanisms that couple states with different energy differences violate this secular approximation.

Powder Averaging
----------------

For disordered samples, spectra are averaged over molecular orientations (α, β).
Since the RWA assumes an isotropic g-factor, the γ Euler angle does not affect resonance frequencies


Applicability
-------------

The RWA is suitable for:

* Organic radicals with isotropic g-factors
* Triplet states with axial zero-field splitting aligned with the field
* Systems where coherence-population coupling is negligible

The RWA should **not** be used for:

* Transition metal complexes with anisotropic g-tensors
* Systems with strong non-secular relaxation
* Single-molecule magnets with large zero-field splittings
* High-field EPR where g-anisotropy is resolved

For such systems, use the propagator-based approach which imposes no similar approximations.