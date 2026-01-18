.. _epr_spectrum_construction:

EPR Spectrum Construction in MarS
=================================

The construction of an EPR spectrum in MarS follows a sequence of well-defined computational steps:

1. **Define spin particles**  
   The spin system consists of quantum particles-electrons and magnetic nuclei-each specified by its spin quantum number and intrinsic magnetic parameters.

2. **Specify interactions**  
   Interactions (e.g., hyperfine, zero-field splitting, nuclear-nuclear couplings) are defined using the :class:`mars.spin_system.Interaction` class.  
   See :ref:`interaction_in_mars` for details.

3. **Assemble the spin system and sample**  
   Particles and interactions are combined into a :class:`mars.spin_system.SpinSystem`.  
   Disordered (powder) samples are represented by :class:`mars.spin_system.MultiOrientedSample`.  
   The next workflow is managed by the abstract class :class:`mars.spectra_manager.BaseSpectra`.

4. **Construct Hamiltonian matrices**  
   MarS computes four matrices that define the spin Hamiltonian under an external magnetic field **B** = (Bx, By, Bz):

   .. math::
      \hat{H} = \hat{F} + B_x \hat{G}_x + B_y \hat{G}_y + B_z \hat{G}_z

   - **F**: Field-independent term (zero-field splitting, hyperfine, nuclear couplings, etc.)
   - **Gx, Gy, Gz**: Zeeman coupling operators, defined explicitly as:

     .. math::
        \hat{G}_\alpha = \mu_B \sum_{e} \sum_{\beta} g^{(e)}_{\alpha\beta} \hat{S}^{(e)}_\beta
                        + \sum_{n} \gamma_n \hat{I}^{(n)}_\alpha, \quad \alpha \in \{x, y, z\}

     where:
       - The first sum runs over all electrons (*e*), with :math:`g^{(e)}_{\alpha\beta}` the components of the electron g-tensor,
       - :math:`\hat{S}^{(e)}_\beta` is the β-component of the electron spin operator,
       - The second sum runs over all nuclei (*n*),
       - :math:`\gamma_n = g_n \mu_N / \hbar` is the gyromagnetic ratio of nucleus *n*, expressed via its g-factor :math:`g_n` and the nuclear magneton :math:`\mu_N`,
       - :math:`\hat{I}^{(n)}_\alpha` is the α-component of the nuclear spin operator.

5. **Determine resonance conditions**  
   MarS offers three algorithms:

   5.1. **Resonance-field search** (:class:`mars.res_field_algorithm.ResField`)  
        Solves :math:`\hbar \omega = E_i(B_{ij}) - E_j(B_{ij})` for resonance fields :math:`B_{ij}` at fixed frequency.  
        This is a batched, GPU-accelerated implementation of the method used in EasySpin [doi:10.1016/j.jmr.2005.08.013], default for solid-state simulations.

   5.2. **Secular approximation** (:class:`mars.secular_approximation.ResSecular`)  
        Modifies **F** to commute with **Gz**, enabling efficient simulation under the rotating wave approximation.  
        Used by default in :class:`mars.spectra_manager.DensityTimeSpectra` for time-resolved EPR.

   5.3. **Resonance-frequency search** (:class:`mars.res_freq_algorithm.ResFreq`)  
        Finds resonant frequencies in a given interval at fixed magnetic field.

6. **Compute transition intensities**  
   Intensities are computed by :class:`mars.spectra_manager.BaseIntensityCalculator`.

   Assuming the quantization axis aligns with the magnetic field direction (e.g., **B** ∥ *z*), the intensity of a transition between eigenstates |i⟩ and |j⟩ is:

   .. math::
      I_{ij} \propto \left( |\langle i | \hat{G}_x | j \rangle|^2 + |\langle i | \hat{G}_y | j \rangle|^2 \right) \cdot (p_j - p_i)

   where the population of state *k* for the equilibrium case is:

   .. math::
      p_k = \frac{e^{-E_k / k_B T}}{Z}, \quad Z = \sum_k e^{-E_k / k_B T} 

   In the crystal case the I_{ij} \propto \left( |\langle i | \hat{G}_x | j \rangle|^2 \cdot (p_j - p_i)

   In **time-resolved** or **non-equilibrium** simulations (e.g., photoexcited states), populations :math:`p_i(t)` are not thermal and are managed via Context tool: see :class:`mars.population.contexts.Context`.

   For density matrix-based time dependant methods, the signal is computed directly as ~:math:`\mathrm{Tr}(\hat{G}_{\perp} \hat{\rho}(t))`,
   where :math:`\hat{G}_{\perp}` is the detected transverse spin component (e.g., :math:`\hat{G}_x`, :math:`\hat{G}_y`, or circular combinations depending on detection method and computation method)

7. **Account for line broadening**  
   Linewidths arise from unresolved interactions or distributions in Hamiltonian parameters (e.g., g-strain, ham_strain).

8. **Field-sweep Jacobian correction**  
   When converting from frequency-domain to field-swept spectra, intensities are scaled by :math:`|d\nu/dB|` to preserve spectral weight, consistent with standard EPR theory and EasySpin.

9. **Powder averaging with operation parallelism**  
   For powder samples, all matrix operations act on a batched tensor that combines orientation and sample dimensions. This enables full GPU parallelism across orientations without explicit loops.

10. **Interpolation to refined orientation grid**  
    Resonance positions, intensities, and linewidths are interpolated from a coarse to a fine spatial orientation grid to improve spectral resolution.

11. **Assemble stick spectrum**  
    A discrete spectrum is built from resonance lines with computed positions, intensities, and intrinsic widths.

12. **Apply lineshape convolution**  
    The final spectrum is obtained by convolving the stick spectrum with a Gaussian, Lorentzian, or Voigt profile, depending on the dominant broadening mechanism.