.. _intensity_calculation:

Transition Intensity Calculation in MarS
========================================

Overview
--------

The intensity of a transition between two spin eigenstates |a> and |b> is determined by:

1. **Transition matrix elements** (magnetization) — describing how strongly the microwave magnetic field couples to the transition
2. **Level populations** — determining the occupancy difference between initial and final states

For stationary EPR spectroscopy and time-resolved EPR spectroscopy with kinetic relaxation calculation:

.. math::
   I_{ab} = \chi \frac{B_1^2}{4} |{\bf n}_1^T \boldsymbol{\mu}|^2

where χ contains the population difference, B₁ is the oscillating field amplitude, **n**₁ describes the radiation polarization, and **μ** is the magnetic transition dipole moment (MTDM) vector.

Intensity Calculator Classes
-----------------------------

MarS provides several intensity calculator implementations through :class:`mars.spectra_manager.spectra_manager.BaseIntensityCalculator` and its subclasses:

**StationaryIntensityCalculator**
   For continuous-wave (CW) EPR experiments with thermal equilibrium populations. Uses Boltzmann distributions or context-defined populations.
   Reference: :class:`mars.spectra_manager.spectra_manager.StationaryIntensityCalculator`

**TimeIntensityCalculator**
   For time-resolved EPR based on level population relaxation. Populations evolve according to kinetic equations with configurable relaxation rates.
   Reference: :class:`mars.spectra_manager.spectra_manager.TimeIntensityCalculator`

**TimeDensityCalculator**
   For time-resolved EPR using density matrix formalism. In this approach the transition matrix elements and level populations are not separable.
   Reference: :class:`mars.spectra_manager.spectra_manager.TimeDensityCalculator`

**WaveIntensityCalculator**
   For EPR experiments with non-standard excitation geometries and polarizations (circular, linear, unpolarized). Accounts for arbitrary orientation of radiation relative to the magnetic field.
   Reference: :class:`mars.spectra_manager.wave_calculator.WaveIntensityCalculator`

Computational Workflow
----------------------

All intensity calculators share a common computational pattern:

1. **Receive transition parameters**
   
   - Zeeman operator components **G**ₓ, **G**ᵧ, **G**_z
   - Eigenvectors of lower (|ψ_down⟩) and upper (|ψ_up⟩) states
   - Energy level indices and eigenvalues
   - Resonance manifold values (fields or frequencies)

2. **Compute magnetization term**
   
   Calculate the squared matrix elements of Zeeman operators between transition states (see :ref:`magnetization_computation`).

3. **Evaluate population contribution**
   
   Determine the population difference (p_down - p_up) or time-dependent populations (see :ref:`population_calculation`).

4. **Combine components**
   
   Multiply magnetization and population terms to obtain final intensity.

Standard CW-EPR Intensity
--------------------------

For continuous-wave EPR under thermal equilibrium, :class:`mars.spectra_manager.spectra_manager.StationaryIntensityCalculator` computes:

.. math::
   I_{ij} = (p_j - p_i) \cdot M_{ij}

where the population difference follows Boltzmann statistics:

.. math::
   p_k = \frac{e^{-E_k / k_B T}}{Z}, \quad Z = \sum_k e^{-E_k / k_B T}

or is defined by Contex (see :class:`mars.population.contexts.Context`)

and M_{ij} is the magnetization term (see :ref:`magnetization_computation`).


Time-Resolved EPR Intensity
----------------------------

For time-resolved EPR experiments, :class:`mars.spectra_manager.spectra_manager.TimeIntensityCalculator` handles non-equilibrium time dependant populations:

.. math::
   I_{ij}(t) = p_i(t) \cdot M_{ij}

Here populations p_i(t) evolve according to rate equations defined in the Context. The magnetization M_{ij} remains constant but populations change with time.


Density Matrix Time-Domain Calculation
---------------------------------------

:class:`mars.spectra_manager.spectra_manager.TimeDensityCalculator` extends ``TimeIntensityCalculator`` with density matrix formalism. This approach:

- Computes detected signal as Tr(**G⊥,+,-,x,y** **ρ**(t)) depending on detection method or computation method


Polarized Radiation and Excitation Geometry
--------------------------------------------

:class:`mars.spectra_manager.spectra_manager.WaveIntensityCalculator` handles experiments with:

- **Circular polarization** (left/right-handed)
- **Linear polarization**
- **Unpolarized radiation**

The intensity includes polarization-dependent weight factors (see :ref:`polarized_radiation`):

.. math::
   I_{ij} = (p_j - p_i) \cdot \left[ M_{xy} \cdot w_{xy} + M_z \cdot w_z + M_{mixed} \cdot w_{mixed} \right]

where w_{xy}, w_z, w_{mixed} are computed by ``terms_computer`` classes based on Wigner d-matrix elements
