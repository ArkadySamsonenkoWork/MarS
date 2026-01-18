.. _complex_context:

Complex Context Construction
=============================

Real spin systems often involve multiple simultaneous relaxation mechanisms or coupled subsystems.
MarS provides powerful algebraic operations to construct complex Context objects from simpler building blocks: **addition** (``+``) and **multiplication** (``@``).

Context Algebra Overview
-------------------------

MarS supports two fundamental operations:

1. **Addition** (``context_1 + context_2``): Combines independent relaxation processes acting on the **same** spin system

2. **Multiplication** (``context_1 @ context_2``): Creates a composite system from **independent** subsystems

Both operations automatically handle:

- Basis transformations to a common eigenbasis
- Detailed balance enforcement
- Proper composition of kinetic matrices or relaxation superoperators

Addition: Combining Relaxation Mechanisms
------------------------------------------

The addition operator combines multiple relaxation pathways that act simultaneously on the same set of quantum states. This is the most common operation for building realistic relaxation models.

Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^

For **population dynamics**, the total kinetic matrix is the sum of individual contributions:

.. math::

   K_{\text{total}} = K^{(1)} + K^{(2)} + \cdots + K^{(n)}

For **density matrix dynamics**, the total relaxation superoperator becomes:

.. math::

   \hat{\mathcal{R}}_{\text{total}} = \hat{\mathcal{R}}^{(1)} + \hat{\mathcal{R}}^{(2)} + \cdots + \hat{\mathcal{R}}^{(n)}

Each component :math:`K^{(i)}` or :math:`\hat{\mathcal{R}}^{(i)}` is first calculated in its own specified basis, then all are transformed to the common eigenbasis of the full Hamiltonian before summation.

Physical Scenarios
^^^^^^^^^^^^^^^^^^

Common situations requiring addition:

1. **Multiple relaxation pathways:** Phosphorescence decay + relaxation between triplet sublevels
2. **Different bases:** Initial populations in ZFS basis + transitions in eigen basis
3. **Competing processes:** Fast equilibration + slow population loss


Example 1: Triplet State with Multiple Mechanisms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A triplet state formed by intersystem crossing typically exhibits:

- Selective initial population in the molecular frame (ZFS basis)
- Phosphorescence (radiative decay) in the molecular frame
- Spin-lattice relaxation between Zeeman levels (eigen basis)

.. code-block:: python

   import torch
   from mars import spin_system, population, spectra_manager
   
   # Define triplet system
   g_tensor = spin_system.Interaction(2.0032, dtype=torch.float64)
   zfs = spin_system.DEInteraction([540e6, -78e6], dtype=torch.float64)
   
   triplet_system = spin_system.SpinSystem(
       electrons=[1.0],
       g_tensors=[g_tensor],
       electron_electron=[(0, 0, zfs)]
   )
   
   sample = spin_system.MultiOrientedSample(
       spin_system=triplet_system,
       ham_strain=2.2e7,
       gauss=0.0011,
       lorentz=0.0011
   )
   
   # Context 1: Selective population and phosphorescence (ZFS basis)

   initial_pops = [0.72, 0.08, 0.20]  # [TX, TY, TZ]
   
   # Different phosphorescence rates for each sublevel
   phosphorescence = torch.tensor([105.0, 48.0, 82.0])  # s^-1
   
   context_molecular = population.Context(
       sample=sample,
       basis="zfs",
       init_populations=initial_pops,
       out_probs=phosphorescence
   )
   
   # Context 2: Spin-lattice relaxation (eigen basis)
   # Fast equilibration between adjacent levels
   # Slower direct transitions between extreme levels
   spin_lattice = torch.tensor([
       [0.0,    1100.0,  85.0],
       [1100.0, 0.0,     1100.0],
       [85.0,   1100.0,  0.0]
   ])  # s^-1
   
   context_zeeman = population.Context(
       sample=sample,
       basis="eigen",
       free_probs=spin_lattice,
       temperature=140.0  # K
   )
   
   # Combine both mechanisms
   context_total = context_molecular + context_zeeman
   
   # Calculate time-resolved spectrum
   tr_spectra = spectra_manager.CoupledTimeSpectra(
       freq=9.6e9,
       sample=sample,
       harmonic=0,
       context=context_total,
       temperature=300.0
   )
   
   fields = torch.linspace(0.30, 0.40, 1000, dtype=torch.float64)
   times = torch.linspace(0, 8e-3, 600, dtype=torch.float64)  # 8 ms
   
   spectrum_2d = tr_spectra(sample, fields, times)

**Physical behavior:**

- **t < 1 ms:** Fast redistribution of population between Zeeman levels
- **1 ms < t < 10 ms:** Slow decay of total triplet population via phosphorescence

Example 2: Adding Dephasing to Population Dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For density matrix calculations, coherence decay is crucial. Without some dephasing, the system may saturate due to an oscillating magnetic field, which is not observed experimentally.
Therefore, even if the dephasing is unknown, it is best to specify it to avoid unexpected spectral saturation effects.:

.. code-block:: python

   import torch
   from mars import population, spectra_manager
   
   # Context 1: Population dynamics (populations + transitions)
   context_pop = population.Context(
       sample=sample,
       basis="zfs",
       init_populations=[0.68, 0.12, 0.20],
       out_probs=torch.tensor([92.0, 45.0, 75.0]),  # s^-1
       free_probs=torch.tensor([
           [0.0,    950.0,   70.0],
           [950.0,  0.0,     950.0],
           [70.0,   950.0,   0.0]
       ])
   )
   
   # Context 2: Pure dephasing (only affects coherences)
   # Faster dephasing for higher energy levels
   T2_probs = torch.tensor([6e3, 9.5e3, 14e3])  # s^-1
   
   context_dephasing = population.Context(
       sample=sample,
       basis="eigen",
       dephasing=T2_probs
   )
   
   # Combine for full density matrix dynamics
   context_density = context_pop + context_dephasing
   
   # Use density matrix solver
   tr_spectra_density = spectra_manager.DensityTimeSpectra(
       freq=9.6e9,
       sample=sample,
       harmonic=0,
       context=context_density,
       temperature=140.0,
       populator="rwa"  # Rotating wave approximation
   )
   
   spectrum_with_dephasing = tr_spectra_density(sample, fields, times)

Example 3: Multiple Independent Processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Radical pair systems may have multiple competing decay channels:

.. code-block:: python

   # Context 1: Singlet-triplet interconversion
   st_mixing = torch.tensor([
       [0.0,    2500.0,  0.0,     0.0],
       [2500.0, 0.0,     0.0,     0.0],
       [0.0,    0.0,     0.0,     0.0],
       [0.0,    0.0,     0.0,     0.0]
   ])  # s^-1
   
   context_st = population.Context(
       sample=radical_pair_sample,
       basis="multiplet",
       free_probs=st_mixing
   )
   
   # Context 2: Radical recombination (singlet channel)
   recombination = torch.tensor([5000.0, 0.0, 0.0, 0.0])  # s^-1
   
   context_recomb = population.Context(
       sample=radical_pair_sample,
       basis="multiplet",
       out_probs=recombination
   )
   
   # Context 3: Spin relaxation in product basis
   relaxation = torch.tensor([
       [0.0,  500.0,  500.0,  0.0],
       [500.0, 0.0,   0.0,    500.0],
       [500.0, 0.0,   0.0,    500.0],
       [0.0,   500.0, 500.0,  0.0]
   ])  # s^-1
   
   context_relax = population.Context(
       sample=radical_pair_sample,
       basis="product",
       free_probs=relaxation
   )
   
   # Combine all three mechanisms
   context_rp = context_st + context_recomb + context_relax

Properties of Addition
^^^^^^^^^^^^^^^^^^^^^^

1. **Commutative:** ``context_1 + context_2 == context_2 + context_1``

2. **Associative:** ``(context_1 + context_2) + context_3 == context_1 + (context_2 + context_3)``

3. **Basis-independent:** Contexts can be defined in different bases; transformation is automatic

4. **Initial state:** All initial populations are summed up. 

5. **Detailed balance:** Applied separately to each context free_probs before summation

Multiplication: Composite Quantum Systems
------------------------------------------

The multiplication operator (``@``) constructs a composite quantum system from two (or more) independent subsystems. This is essential for modeling weakly coupled or non-interacting spin systems.

Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^

For **population dynamics**, the composite system is formed via tensor products:

.. math::

   \mathbf{n}_{\text{total}} = \mathbf{n}^{(1)} \otimes \mathbf{n}^{(2)}

.. math::

   K_{\text{total}} = K^{(1)} \otimes \mathbb{I}^{(2)} + \mathbb{I}^{(1)} \otimes K^{(2)}

For **density matrix dynamics**:

.. math::

   \hat{\rho}_{\text{total}} = \hat{\rho}^{(1)} \otimes \hat{\rho}^{(2)}

.. math::

   \hat{\mathcal{R}}_{\text{total}} = \hat{\mathcal{R}}^{(1)} \otimes \hat{\mathbb{I}}^{(2)} + \hat{\mathbb{I}}^{(1)} \otimes \hat{\mathcal{R}}^{(2)}

where :math:`\mathbb{I}` and :math:`\hat{\mathbb{I}}` are identity operators in Hilbert and Liouville space, respectively.

**Implementation:** MarS uses Clebsch-Gordan-like coefficients to handle these transformations. This is quite more efficiently than the direct mathematical expressions. Key functions:

- :func:`mars.population.transform.compute_clebsch_gordan_probabilities` - Compute transformation coefficients
- :func:`mars.population.transform.transform_kronecker_populations` - Transform populations
- :func:`mars.population.transform.transform_kronecker_matrix` - Transform rate matrices  
- :func:`mars.population.transform.transform_kronecker_density` - Transform density matrices
- :func:`mars.population.transform.transform_kronecker_superoperator` - Transform superoperators

.. code-block:: python

   from mars.population.transform import (
       compute_clebsch_gordan_probabilities,
       transform_kronecker_populations,
       transform_kronecker_matrix
   )
   
   # Example: Two subsystems with bases basis_1 and basis_2
   # Target is the coupled basis (eigen basis of full system)
   
   # Compute Clebsch-Gordan coefficients
   coeffs = compute_clebsch_gordan_probabilities(
       target_basis=coupled_basis,
       basis_list=[basis_1, basis_2]
   )
   # Shape: [..., k1, k2, K] where K = k1*k2
   
   # Transform populations
   pops_1 = torch.tensor([0.7, 0.2, 0.1])  # 3 levels
   pops_2 = torch.tensor([0.5, 0.5])        # 2 levels  
   pops_total = transform_kronecker_populations([pops_1, pops_2], coeffs)
   # Shape: [..., 6] for combined 3×2 system
   
   # Transform rate matrices
   K1 = torch.zeros(3, 3)  # Rates for system 1
   K2 = torch.zeros(2, 2)  # Rates for system 2
   K_total = transform_kronecker_matrix([K1, K2], coeffs)
   # Shape: [..., 6, 6]


Combining Multiplication and Addition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For complex systems, both operations can be combined:

.. code-block:: python

   # Let's consider two triplet systems with defined relaxation to initial singlet state
   context_triplet_1 = population.Context(
       sample=single_sample,
       basis="zfs",
       init_populations=[0.68, 0.12, 0.20],
       out_probs=torch.tensor([95.0, 50.0, 75.0])
   )
   
   context_triplet_2 = population.Context(
       sample=single_sample,
       basis="zfs",
       init_populations=[0.22, 0.65, 0.13],
       out_probs=torch.tensor([95.0, 50.0, 75.0])
   )
   
   # Composite initial state and decay
   context_product = context_triplet_1 @ context_triplet_2
   
   # Add collective relaxation processes in eigenbasis


   collective_relax = torch.zeros(9, 9, dtype=torch.float64)
   # Define selected transitions between composite levels
   collective_relax[1, 3] = 200.0  # s^-1
   collective_relax[3, 1] = 200.0
   collective_relax[2, 5] = 150.0
   collective_relax[5, 2] = 150.0
   
   context_collective = population.Context(
       sample=double_sample,
       basis="eigen",
       free_probs=collective_relax,
       temperature=100.0
   )
   
   # Total context: product state + collective relaxation
   context_full = context_product + context_collective

Properties of Multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Commutative:** ``context_1 @ context_2 == context_2 @ context_1``

2. **Associative:** ``(context_1 @ context_2) @ context_3 == context_1 @ (context_2 @ context_3)``

3. **Dimensionality:** :math:`N_{\text{total}} = N_1 \times N_2`

4. **Independent evolution:** Each subsystem evolves according to its own dynamics

5. **Product basis:** Natural for specifying independent initial conditions

Practical Considerations
-------------------------

Choosing Between Addition and Multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Use addition (+) when:**

- Multiple processes affect the same spin system
- Different physical mechanisms
- Processes defined in different bases
- Adding coherence effects to population dynamics

**Use multiplication (@) when:**

- System consists of coupled subsystems
- Independent initial conditions for each subsystem


Order of Operations
^^^^^^^^^^^^^^^^^^^

When combining both operations:

.. code-block:: python

   # These are equivalent:
   result1 = (context_1 @ context_2) + context_3  # Multiply first
   result2 = context_3 + (context_1 @ context_2)  # Addition is commutative