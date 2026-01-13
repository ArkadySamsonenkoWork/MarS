.. _magnetization_computation:

Magnetization Computation in MarS
==================================

Overview
--------

The magnetization term quantifies how strongly an EPR transition couples to the oscillating magnetic field **B**₁ of the microwave radiation. It is computed from matrix elements of the Zeeman operator components between the eigenstates involved in the transition.

Physical Basis
--------------
In MarS magnitization matrix elements are computated always for stationary EPR spectroscopy and for time-resolved EPR spectroscopy when relaxation of populations are considered

The interaction Hamiltonian between spins and radiation is:

.. math::
   \hat{H}_{\text{int}} = -\boldsymbol{\hat{\mu}} \cdot {\bf B}_1(t)

where **μ̂** is the total magnetic dipole moment operator:

.. math::
   \boldsymbol{\hat{\mu}} = -\mu_B \sum_e {\bf g}^{(e)} \cdot {\bf\hat{S}}^{(e)} + \sum_n g_n \mu_N {\bf\hat{I}}^{(n)}

The first sum runs over electrons with Bohr magneton μ_B, g-tensors **g**^(e), and spin operators **Ŝ**^(e). The second sum covers nuclei with nuclear magneton μ_N, nuclear g-factors g_n, and nuclear spin operators **Î**^(n).

In the laboratory frame where **B**₀ defines the z-axis, the Zeeman operators **Ĝ**ₓ, **Ĝ**ᵧ, **Ĝ**_z encode the coupling to magnetic field components along x, y, z directions (see :ref:`epr_spectrum_construction`).

Magnetic Transition Dipole Moment Vector
-----------------------------------------

The **magnetic transition dipole moment vector** for a transition |i⟩ → |j⟩ is:

.. math::
   \boldsymbol{\mu}_{ij} = \langle j | \boldsymbol{\hat{\mu}} | i \rangle = \langle j | (-\hat{G}_x, -\hat{G}_y, -\hat{G}_z) | i \rangle

This complex vector **μ**_ij characterizes both the strength and spatial orientation of the transition. Its properties:

- **Magnitude** |**μ**_ij| determines transition probability
- **Direction** indicates the polarization of radiation that couples most strongly
- **Phase** (for complex **μ**_ij) describes rotational sense around **B**₀

The magnetic transition dipole moment is unique up to an arbitrary complex phase factor.

Powder-Averaged Magnetization
------------------------------

For **disordered (powder) samples**, molecules adopt all possible orientations relative to **B**₀. In standard EPR with **B**₁ ⊥ **B**₀, the magnetization for a transition is:

.. math::
   M_{ij}^{\text{powder}} = \left( |\langle j | \hat{G}_x | i \rangle|^2 + |\langle j | \hat{G}_y | i \rangle|^2 \right) \left(\frac{\hbar}{\mu_B}\right)^2

This sums the squared matrix elements of the transverse Zeeman components **Ĝ**ₓ and **Ĝ**ᵧ. The conversion factor (ℏ/μ_B)² ensures proper units.

**Implementation:**

The method ``_compute_magnetization_powder`` in :class:`mars.spectra_manager.spectra_manager.BaseIntensityCalculator` computes:

Crystal Magnetization
---------------------

For **single-crystal or oriented samples**, the crystal axes have fixed orientation relative to **B**₀. In conventional resonator EPR with **B**₁ ⊥ **B**₀ and **B**₁ aligned along laboratory x:

.. math::
   M_{ij}^{\text{crystal}} = |\langle j | \hat{G}_x | i \rangle|^2 \left(\frac{\hbar}{\mu_B}\right)^2

Only the **Ĝ**ₓ component contributes, as the radiation magnetic field oscillates purely along x.

Generalized Magnetization for Polarized Radiation
--------------------------------------------------

When using **circular, linear, or unpolarized radiation** with arbitrary propagation direction **k** relative to **B**₀, all three Zeeman components contribute with polarization-dependent weights:

.. math::
   M_{ij} = \left[ M_{xy} \cdot w_{xy} + M_z \cdot w_z + M_{\text{mixed}} \cdot w_{\text{mixed}} \right] \left(\frac{\hbar}{\mu_B}\right)^2

where:

.. math::
   M_{xy} &= |\langle j | \hat{G}_x | i \rangle|^2 + |\langle j | \hat{G}_y | i \rangle|^2 \\
   M_z &= |\langle j | \hat{G}_z | i \rangle|^2 \\
   M_{\text{mixed}} &= \text{Im}[\langle j | \hat{G}_x | i \rangle \cdot \langle j | \hat{G}_y | i \rangle^*]

The weight factors w_{xy}, w_z, w_{mixed} depend on polarization and geometry (see :ref:`polarized_radiation`).

**Physical interpretation:**

- **M_{xy}**: Transverse magnetization for standard allowed transitions
- **M_z**: Longitudinal magnetization for forbidden transitions in non-standard geometries
- **M_{mixed}**: Term which differs circular polarization handedness

The mixed term is another notation of cross product Im(**μ**ₓ × **μ**ᵧ*)_z, which determines the rotation axis of the magnetic transition dipole moment vector. For circular polarization, constructive/destructive interference between left and right components enhances or suppresses transitions based on their rotational sense.

**Implementation:**

:class:`mars.spectra_manager.spectra_manager.WaveIntensityCalculator` implements ``_compute_magnetization_powder`` and ``_compute_magnetization_crystal`` to compute all three components:

.. code-block:: python

   μ_x = ⟨ψ_down|(-Ĝ_x)|ψ_up⟩
   μ_y = ⟨ψ_down|(-Ĝ_y)|ψ_up⟩
   μ_z = ⟨ψ_down|(-Ĝ_z)|ψ_up⟩
   
   M_xy = |μ_x|² + |μ_y|²
   M_z = |μ_z|²
   M_mixed = Im(μ_x · μ_y*)
   
   M = M_xy·w_xy + M_z·w_z + M_mixed·w_mixed

The negative signs ensure correct phase conventions for the dipole moment.

Wigner d-Matrix Elements
-------------------------

For **powder samples with polarized radiation**, the weight factors involve Wigner d-matrix elements d^(1)_{m,m'}(θ) that describe rotation of spin-1 states. For radiation with helicity h (±1 for circular, 0 effectively for linear) and angle θ between **k** and **B**₀:

.. math::
   w_{xy}^{\text{powder}} &= \frac{1}{2}\left[ d^2_{h,+1}(\theta) + d^2_{h,-1}(\theta) \right] \\
   w_z^{\text{powder}} &= d^2_{h,0}(\theta) \\
   w_{\text{mixed}}^{\text{powder}} &= d^2_{h,+1}(\theta) - d^2_{h,-1}(\theta)

These functions are computed by ``wigner_term_square`` in the code. Specific expressions:

.. math::
   d^2_{h,h}(\theta) &= \cos^4(\theta/2) \\
   d^2_{h,-h}(\theta) &= \sin^4(\theta/2) \\
   d^2_{h,0}(\theta) &= \sin^2(\theta)/2 \quad \text{for } h \ne 0
