.. _population_calculation:

Population Calculation in MarS
===============================

Overview
--------

The population term in EPR intensity calculations determines the populations difference between energy levels involved in a transition. While magnetization (see :ref:`magnetization_computation`) describes the coupling strength to radiation, populations describe how many spins participate in the transition.

MarS supports three population models:

1. **Thermal equilibrium** — Boltzmann distributions at specified temperature
2. **Non-equilibrium kinetics** — Time-dependent populations with relaxation
3. **Density matrix** — Full computation of density matrix evolution for time-resolved case

Physical Foundation
-------------------

In continuous-wave EPR, the intensity of a transition |i⟩ → |j⟩ depends on the population difference:

.. math::
   I_{ij} \propto (p_j - p_i) \cdot M_{ij}

where p_k is the fractional population of state |k⟩ (Σ_k p_k = 1), and M_{ij} is the magnetization term.

**Physical interpretation:**

The population difference (p_j - p_i) represents the net number of spins available for absorption. Non-equilibrium conditions can produce:

- **Enhanced absorption**: p_j - p_i > thermal value
- **Emission**: p_j - p_i < 0 (population inversion)
- **Absorption-emission patterns**: Mixed positive/negative intensities

Such anomalous populations arise in chemically induced dynamic electron polarization (CIDEP), optically detected EPR (ODEPR), and time-resolved experiments following photochemical excitation.

Thermal Equilibrium Populations
--------------------------------

**Boltzmann distribution**

At temperature T, energy level k with energy E_k has thermal population:

.. math::
   p_k = \frac{e^{-E_k / k_B T}}{Z}, \quad Z = \sum_{\ell} e^{-E_{\ell} / k_B T}

where k_B is Boltzmann's constant and Z is the partition function ensuring normalization.


Non-Equilibrium Level Populations
----------------------------------
MarS allows to compute stationary (CW) EPR spectrum of non-eqalibrium level populations

Time-resolved EPR following some excitation (laser flash, radiation pulse, electric discharge) creates non-equilibrium populations that evolve according to kinetic equations:

.. math::
   \frac{dp_i}{dt} = -\sum_{j \ne i} k_{ij} p_i + \sum_{j \ne i} k_{ji} p_j

where k_{ij} is the rate constant for relaxation from |i⟩ to |j⟩.


Density Matrix Populations
---------------------------

For pulsed EPR, spin echoes, ESEEM, and other coherent experiments, populations alone are insufficient. The full density matrix **ρ** includes:

- **Diagonal elements**: Populations p_ii = ⟨i|**ρ**|i⟩
- **Off-diagonal elements**: Coherences p_ij = ⟨i|**ρ**|j⟩ (i ≠ j)

The density matrix evolves under the Liouville-von Neumann equation:

.. math::
   \frac{d\boldsymbol{\rho}}{dt} = -\frac{i}{\hbar}[\hat{H}, \boldsymbol{\rho}] + \hat{\mathcal{L}}_{\text{relax}}(\boldsymbol{\rho})

where **ℒ**_relax is the relaxation superoperator (Lindblad or Redfield form).

**Rotating wave approximation (RWA)**

For microwave-driven systems, the RWA transforms to a rotating frame at the microwave frequency ω_mw. The secular Hamiltonian **Ĥ**_sec commutes with **Ĝ**_z, simplifying the equations. Off-resonant terms average to zero over a microwave period.

**Implementation: RWADensityPopulator**

The class :class:`mars.population.populators.RWADensityPopulator` propagates the density matrix under RWA:

1. **Initialize** **ρ**(0) from Context (e.g., thermal **ρ**_eq or custom state)
2. **Secular Hamiltonian** **Ĥ**_sec constructed by :class:`mars.secular_approximation.ResSecular`
3. **Liouvillian** **ℒ** = -i[**Ĥ**_sec, ·] + **ℒ**_relax built from Context relaxation parameters
4. **Propagate** **ρ**(t) = exp(**ℒ** t) **ρ**(0) using matrix exponential or ODE solver
5. **Detect** signal as Tr(**G**_⊥ **ρ**(t)) where **G**_⊥ is the detected spin component

This populator is used by :class:`mars.spectra_manager.spectra_manager.TimeDensityCalculator`.


The density matrix approach correctly captures the coherence evolution, whereas level-based kinetics would miss the echo.

Separation of Intensity and Population
---------------------------------------

In :class:`mars.spectra_manager.spectra_manager.TimeIntensityCalculator` and derived classes, intensity calculation is split:

**compute_intensity()** returns M_{ij}
   The geometric/magnetization factor, independent of time

**calculate_population()** returns p_i(t)
   The time-dependent populations or coherences

**Final intensity** I_{ij}(t) = M_{ij} × population_factor(t)

This separation is computationally efficient: magnetization is calculated once per orientation/transition, while populations are recomputed for each time point. For experiments scanning 100 time delays, this avoids redundant magnetization calculations.

**Example workflow:**

.. code-block:: python

   # One-time calculation
   M_ij = calculator.compute_intensity(Gx, Gy, Gz, ψ_down, ψ_up, ...)
   
   # Loop over time delays
   for t in time_points:
       pop_factor = calculator.calculate_population(t, ...)
       I_ij[t] = M_ij * pop_factor

The method ``calculate_population`` queries the populator for p_i(t) or density matrix elements at each t.

Field-Dependent Populations
----------------------------

In most cases, populations depend on energy eigenvalues, which are field-dependent. For each orientation in a powder:

1. **Diagonalize** **Ĥ**(**B**) to get {E_i(**B**), |ψ_i(**B**)⟩}
2. **Compute** thermal or non-equilibrium p_i(**B**)
3. **Calculate** transition intensities I_ij(**B**)

This field dependence is handled transparently by MarS: as resonance algorithms find **B**_res for each transition, the corresponding eigenvalues are passed to the populator.

**Example: Level crossing**

At certain fields, two levels may become nearly degenerate (E_i ≈ E_j). Thermal population differences (p_j - p_i) vanish, suppressing intensity. This creates an "avoided crossing" feature in the spectrum where intensity drops dramatically.

Orientation-Dependent Populations
----------------------------------

For anisotropic photoexcitation (e.g., linearly polarized laser), the initial populations p_i(0) depend on molecular orientation relative to the light polarization. The Context can specify:

.. math::
   p_i^{\alpha}(0) = f(\boldsymbol{\Omega}, \text{excitation geometry})

where **Ω** is the Euler angle orientation and α is the orientation index.

This introduces an additional orientation-weighting beyond the usual Zeeman anisotropy, resulting in spectra with unusual angular patterns.

**Example: Triplet photoselection**

A triplet state excited with light polarized along z preferentially populates molecules with their photoactive axis near z. The initial population for orientation **Ω** is weighted by cos²θ, where θ is the angle between molecular axis and z. This creates a "photoselection spectrum" distinct from isotropic excitation.

Relaxation and Saturation Effects
----------------------------------

**Spin-lattice relaxation (T₁)**

Energy exchange with the lattice drives populations toward thermal equilibrium:

.. math::
   \frac{dp_i}{dt} = -\frac{p_i - p_i^{\text{eq}}}{T_1^{(i)}}

Typical T₁ values range from nanoseconds (Kramers doublets at high field) to seconds (isolated radicals in frozen solution).

**Spin-spin relaxation (T₂)**

Dephasing of coherences from magnetic field inhomogeneity and fluctuations:

.. math::
   \frac{d\rho_{ij}}{dt} = -\frac{\rho_{ij}}{T_2^{(ij)}}

T₂ ≤ 2T₁ always, with equality for homogeneous broadening.

**Saturation**

Strong microwave power can equalize populations (p_i ≈ p_j), reducing intensity. The saturation parameter is:

.. math::
   s = \frac{\gamma^2 B_1^2 T_1 T_2}{1 + (\omega - \omega_{\text{res}})^2 T_2^2}

At high power (s ≫ 1), populations saturate and intensity decreases.

The Context can include saturation effects by modifying relaxation rates or adding driven transition terms.

Powder Averaging of Populations
--------------------------------

For disordered samples, populations generally do not require explicit orientation averaging—they are computed per-orientation and combined with orientation-specific magnetizations:

.. math::
   I(\omega) = \sum_{\alpha} (p_j^{\alpha} - p_i^{\alpha}) \cdot M_{ij}^{\alpha}

where α indexes orientations. Thermal populations at each orientation depend on local energy eigenvalues E_i^α.

However, if the Context specifies global (orientation-averaged) relaxation or external reservoirs, populations may involve orientation integrals. This is handled by the Context providing orientation-dependent or orientation-averaged rates as appropriate.

Computational Considerations
-----------------------------

**Memory scaling**

- **Level-based**: Memory ~ N_levels × N_orientations × N_times
- **Density matrix**: Memory ~ N_levels² × N_orientations × N_times

Density matrix methods require significantly more memory for large spin systems.

**Numerical stability**

Exponentiating the Liouvillian **ℒ** for long times can suffer from numerical errors. Use adaptive ODE solvers or Krylov subspace methods for stiff relaxation problems.

**Parallelization**

Population calculations parallelize efficiently across orientations and time points. GPU tensor operations in PyTorch handle batches of populations simultaneously.

**Relaxation superoperator construction**

Building **ℒ**_relax from T₁, T₂ rates involves tensor products and Kronecker sums. Precompute and cache **ℒ** when possible, especially if relaxation rates are field-independent.

Examples
--------

**Example 1: Cu(II) complex at 4 K**

At liquid helium temperature, only the ground Kramers doublet is populated. Transitions within this doublet dominate the spectrum. Higher-lying states have negligible population, so transitions involving them are invisible. This "spectroscopic isolation" simplifies analysis.

**Example 2: Photoexcited quintet (S=2)**

A photogenerated quintet state has initial populations set by the intersystem crossing mechanism. If ISC populates mS = 0 selectively:

.. math::
   p_0(0) = 1, \quad p_{\pm 1}(0) = 0, \quad p_{\pm 2}(0) = 0

Transitions from mS = 0 show enhanced absorption, others show emission or weak absorption. As the system relaxes, the pattern evolves, revealing relaxation pathways.

**Example 3: ESEEM modulation**

In electron spin echo envelope modulation, nuclear coherences between |mI⟩ states create oscillations in the echo amplitude. The density matrix formalism captures this:

.. math::
   \rho_{mI, mI'}(t) \propto \cos(\omega_{\text{nuclear}} t)

Level-based kinetics cannot describe these coherences—only the density matrix approach reveals the modulation pattern.

**Example 4: Dynamic nuclear polarization (DNP)**

In DNP, microwave irradiation saturates electron transitions while cross-relaxation transfers polarization to nuclei. The Context includes:

- Electron saturation (driven transitions)
- Electron-nuclear cross-relaxation (off-diagonal **ℒ**_relax terms)
- Nuclear T₁ (slow return to equilibrium)

Populations of nuclear states become enhanced, increasing NMR sensitivity.

Summary of Populators
----------------------

**StationaryPopulator**
   - Computes Boltzmann populations at specified temperature
   - Used in :class:`mars.spectra_manager.spectra_manager.StationaryIntensityCalculator`
   - Default for CW-EPR simulations

**LevelBasedPopulator**
   - Handles time-dependent populations with kinetic relaxation
   - Used in :class:`mars.spectra_manager.spectra_manager.TimeIntensityCalculator`
   - Suitable for TREPR with simple relaxation (no coherences)

**RWADensityPopulator**
   - Full density matrix under rotating wave approximation
   - Used in :class:`mars.spectra_manager.spectra_manager.TimeDensityCalculator`
   - Required for pulsed EPR, echoes, ESEEM, coherent experiments

**Custom populators**
   Users can implement custom populators by subclassing :class:`mars.population.populators.BasePopulator` to handle exotic mechanisms (e.g., chemically induced polarization, triplet-triplet annihilation, radical pair effects with specific recombination dynamics).

Best Practices
--------------

**Choosing a populator:**

- Use ``StationaryPopulator`` unless there's a specific reason for non-equilibrium
- Use ``LevelBasedPopulator`` for TREPR with clear initial populations and simple T₁ relaxation
- Use ``RWADensityPopulator`` for any experiment involving coherences, pulses, or driven evolution

**Defining Contexts:**

Provide realistic relaxation rates based on literature or experiment. Unrealistic rates (e.g., T₁ = 1 ps for organic radicals) produce unphysical spectra.

**Validating population models:**

Compare simulated time-dependence against experimental decay curves to validate T₁, T₂, and initial population choices.

**Temperature accuracy:**

Ensure the specified temperature matches experimental conditions. Even 10 K errors significantly affect population differences at cryogenic temperatures.

See Also
--------

- :ref:`intensity_calculation` — Overall intensity framework
- :ref:`magnetization_computation` — Magnetization term calculation
- :class:`mars.population.contexts.Context` — Context system for relaxation and dynamics
- :class:`mars.secular_approximation.ResSecular` — Secular approximation for RWA

