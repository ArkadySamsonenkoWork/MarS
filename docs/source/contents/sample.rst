Sample Representation
=====================

In EPR modeling, a *sample* extends a base spin system with broadening mechanisms (Gaussian/Lorentzian), unresolved strain, and-critically for powders-orientational averaging.


Examples
--------

**1. Powder spectrum of a triplet with axial ZFS**

Simulate a frozen solution of a diradical with D = 0.35 GHz, E = 0, using a default orientation mesh:

.. code-block:: python

   from mars import spin_system, particles

   g_tensor = spin_system.Interaction(2.002)  # g = 2.002
   D_tensor = spin_system.DEInteraction(components=0.35 * 1e9)  # D = 0.35 GHz
   triplet = spin_system.SpinSystem(electrons=[1.0], g_tensors=[g_tensor], electron_electron=[(0, 0, D_tensor)])

   sample = spin_system.MultiOrientedSample(spin_system=triplet, gauss="0.01") # It is better to specify some broadening

The total broadening of each spectral line is constructed from four components:

1. **Homogeneous Lorentzian broadening**, which is applied after the spectrum is constructed.
2. **Inhomogeneous Gaussian broadening**, which is applied after the spectrum is constructed.  
   More information about final spectrum postprocessing can be found in :class:`mars.spectra_manager.spectra_manager.PostSpectraProcessing`.
3. **Residual spectrum broadening** caused by unresolved interactions. This is specified by a parameter ``ham_strain`` and can be anisotropic.
4. **Broadening caused by the presence of a Hamiltonian parameter distribution**. This is specified during :class:`mars.spin_system.Interaction` creation.

All widths are specified as full width at half maximum (FWHM).

**2. Anisotropic Gaussian broadening**

Add orientation-dependent inhomogeneous broadening due to unresolved hyperfine structure:

.. code-block:: python

   # Axial unresolved broadening: σ⊥ = 5 MHz, σ∥ = 15 MHz
   ham_strain = [5e-3 * 1e9, 5e-3 * 1e9, 15e-3 * 1e9]

   sample = spin_system.MultiOrientedSample(
       spin_system=triplet,
       ham_strain=ham_strain,
       lorentz=0.001  # 1 mT homogeneous width
   )


**3. Custom orientation mesh**

.. code-block:: python

   # Use only 80 orientations instead of default ~ 200
   coarse_mesh = (10, 20)  # (initial_grid_frequency, interpolation_grid_frequency)

   fast_sample = spin_system.MultiOrientedSample(
       spin_system=triplet,
       mesh=coarse_mesh
   )

**4. Sample spin system orientation**

In some cases, it's convenient to rotate not just one interaction, but all interactions at once (the entire spin system).
This doesn't change the final spectrum of the sample, but it does change the eigenvectors in each individual powder orientation.

.. code-block:: python

   rotated_sample = spin_system.MultiOrientedSample(
       spin_system=triplet,
       spin_system_frame=[0.0, 0.2, 0.3],
   )


Useful Features
---------------

The sample provides access to the underlying Hamiltonian terms, which are essential for advanced simulations, custom line shape models, or secular approximations.

Hamiltonian Terms
~~~~~~~~~~~~~~~~~

The total spin Hamiltonian in the presence of a magnetic field **B** = (Bₓ, Bᵧ, B_z) is expressed as:

.. math::

   \mathcal{H}(\mathbf{B}) = F + B_x G_x + B_y G_y + B_z G_z

where:

- **F** is the field-independent (zero-field) part of the Hamiltonian,
- **Gₓ**, **Gᵧ**, **G_z** are the Zeeman coupling operators that encode the system’s response to the external magnetic field via electron and nuclear g‑tensors.

Explicitly, these operators are defined as:

.. math::

   \begin{aligned}
   G_x &= \frac{\mu_\mathrm{B}}{h} \sum_{i} \left( g_{i,xx} \hat{S}_i^{(x)} + g_{i,xy} \hat{S}_i^{(y)} + g_{i,xz} \hat{S}_i^{(z)} \right)
         + \frac{\mu_\mathrm{N}}{h} \sum_{j} g_{n,j} \hat{I}_j^{(x)}, \\
   G_y &= \frac{\mu_\mathrm{B}}{h} \sum_{i} \left( g_{i,yx} \hat{S}_i^{(x)} + g_{i,yy} \hat{S}_i^{(y)} + g_{i,yz} \hat{S}_i^{(z)} \right)
         + \frac{\mu_\mathrm{N}}{h} \sum_{j} g_{n,j} \hat{I}_j^{(y)}, \\
   G_z &= \frac{\mu_\mathrm{B}}{h} \sum_{i} \left( g_{i,zx} \hat{S}_i^{(x)} + g_{i,zy} \hat{S}_i^{(y)} + g_{i,zz} \hat{S}_i^{(z)} \right)
         + \frac{\mu_\mathrm{N}}{h} \sum_{j} g_{n,j} \hat{I}_j^{(z)},
   \end{aligned}

where:

- :math:`\hat{S}_i^{(x,y,z)}` are the electron spin operators for electron :math:`i`,
- :math:`\hat{I}_j^{(x,y,z)}` are the nuclear spin operators for nucleus :math:`j`,
- :math:`\mathbf{g}_i` is the (possibly anisotropic) electron g‑tensor for electron :math:`i`,
- :math:`g_{n,j}` is the (currently isotropic) nuclear g‑factor for nucleus :math:`j`,
- :math:`\mu_\mathrm{B}` is the Bohr magneton,
- :math:`\mu_\mathrm{N}` is the nuclear magneton,
- :math:`h` is Planck’s constant.

These operators act on the full product-state Hilbert space and are returned by the sample methods as complex-valued tensors.

You can retrieve them directly:

.. code-block:: python

   # Full Hamiltonian decomposition
   F, Gx, Gy, Gz = sample.get_hamiltonian_terms()

   # Secular approximation: only retain elements of F that commute with Gz.
   # This function is valid in MarS when Gz is giagonal in the basis of individual spin projections
   F_sec, Gx, Gy, Gz = sample.get_hamiltonian_terms_secular()

In the secular form, non-commuting matrix elements of **F** with respect to **G_z** are zeroed out, enforcing the high-field selection rule:

.. math::

   F_{ij}^{\text{sec}} = 
   \begin{cases}
     F_{ij}, & \text{if } |(G_z)_{ii} - (G_z)_{jj}| < \varepsilon \\
     0,      & \text{otherwise}
   \end{cases}

with a small threshold :math:`\varepsilon` (default: :math:`10^{-9}`).

