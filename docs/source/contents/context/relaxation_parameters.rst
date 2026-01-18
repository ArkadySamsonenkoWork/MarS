.. _relaxation_parameters:

Relaxation Parameters
======================

The Context class in MarS supports four types of relaxation parameters that define how spin populations and coherences evolve over time. All probabilities are expressed in **s⁻¹** (inverse seconds).

Overview
--------

MarS defines relaxation through four distinct mechanisms:

1. **out_probs** - Population loss rates
2. **free_probs** - Spontaneous transition probabilities  
3. **driven_probs** - Stimulated transition probabilities
4. **dephasing** - Pure dephasing rates used only for density-based relaxation

Each mechanism serves a specific physical purpose and transforms differently under basis changes.

Out Probabilities (out_probs)
------------------------------

.. list-table::
   :widths: 60 40
   
   * - .. image:: _static/context/out_probs.png
          :width: 100%
          :alt: Out probabilities diagram
     - **Physical Meaning**
       
       Out probabilities describe irreversible population loss from energy levels.
       **Mathematical Form**
       A diagonal vector where element :math:`o_i` represents the loss rate from level :math:`|i\rangle`.

**Example: Triplet State Phosphorescence**

.. code-block:: python

   import torch
   from mars import population
   
   # Define different phosphorescence rates for each sublevel
   # TX sublevel decays fastest, TZ slowest
   out_probs = torch.tensor([120.0, 50.0, 80.0])  # s^-1
   
   # In ZFS basis: [TX, TY, TZ]
   # TX: 120 s^-1 → lifetime ~8.3 ms
   # TY: 50 s^-1  → lifetime ~20 ms  
   # TZ: 80 s^-1  → lifetime ~12.5 ms
   
   context = population.Context(
       sample=triplet_sample,
       basis="zfs",
       init_populations=[0.4, 0.2, 0.4],
       out_probs=out_probs
   )

**Transformation Rule**

Under basis transformation with matrix :math:`U`:

.. math::

   \mathbf{o}' = |U|^2 \cdot \mathbf{o}

where :math:`|U|^2` denotes element-wise squaring of the transformation matrix.

Free Probabilities (free_probs)
--------------------------------

.. list-table::
   :widths: 60 40
   
   * - .. image:: _static/context/free_probs.png
          :width: 100%
          :alt: Free probabilities diagram
     - **Physical Meaning**
       
       Free probabilities represent spontaneous transitions between energy levels that obey detailed balance

       **Mathematical Form**
       
       A matrix :math:`W` where element :math:`w_{ij}` is the transition rate from :math:`|j\rangle` to :math:`|i\rangle`.
       
       Diagonal elements are zero (no self-transitions).

**Example: Triplet Spin-Lattice Relaxation**

.. code-block:: python

   import torch
   from mars import population
   
   # Define transition matrix in eigen basis
   # Strong relaxation between adjacent levels
   # Weak direct relaxation between extreme levels
   free_probs = torch.tensor([
       [0.0,    800.0,   50.0],   # to level 0
       [800.0,  0.0,     800.0],  # to level 1
       [50.0,   800.0,   0.0]     # to level 2
   ])  # s^-1
   
   # Adjacent transitions: ~1.25 ms equilibration time
   # Direct 0↔2: ~20 ms equilibration time
   
   context = population.Context(
       sample=triplet_sample,
       basis="eigen",
       init_populations=[0.35, 0.35, 0.30],
       free_probs=free_probs,
       temperature=80.0
   )

**Detailed Balance Enforcement** (see also :ref:`detailed_balance`)

MarS automatically enforces detailed balance at temperature :math:`T`:

.. math::

   w_{ij}^* = \frac{w_{ij} + w_{ji}}{1 + e^{(E_i - E_j)/k_B T}}

.. math::

   w_{ji}^* = \frac{w_{ij} + w_{ji}}{1 + e^{(E_j - E_i)/k_B T}}

where :math:`w_{ij}^*` are the modified rates ensuring :math:`\frac{w_{ij}^*}{w_{ji}^*} = e^{(E_j - E_i)/k_B T}`.

**Transformation Rule**

Under basis transformation:

.. math::

   W' = |U|^T \cdot W \cdot |U|

Driven Probabilities (driven_probs)
------------------------------------

.. list-table::
   :widths: 60 40
   
   * - .. image:: _static/context/driven_probs.png
          :width: 100%
          :alt: Driven probabilities diagram
     - **Physical Meaning**
       
       Driven probabilities describe stimulated transitions that do NOT obey detailed balance

       
       **Mathematical Form**
       
       A matrix :math:`D` where :math:`d_{ij}` is the stimulated rate from :math:`|j\rangle` to :math:`|i\rangle`.
       
       **Not modified** by detailed balance.

**Example: Selective Microwave Excitation**

.. code-block:: python

   import torch
   from mars import population
   
   # Simulate selective excitation of specific transition
   # Strong driving of 0→1 transition
   # Moderate driving of 1→2 transition
   driven_probs = torch.tensor([
       [0.0,    2000.0,  0.0],      # to level 0
       [2000.0, 0.0,     500.0],    # to level 1  
       [0.0,    500.0,   0.0]       # to level 2
   ])  # s^-1
   
   # Strong 0↔1 transition: ~0.5 ms
   # Weaker 1↔2 transition: ~2 ms
   
   context = population.Context(
       sample=triplet_sample,
       basis="eigen",
       init_populations=None, # Then the initial condition is thermal equilibrium
       driven_probs=driven_probs
   )

**Transformation Rule**

Same as free probabilities:

.. math::

   D' = |U|^T \cdot D \cdot |U|

Dephasing (dephasing)
---------------------

.. list-table::
   :widths: 60 40
   
   * - .. image:: _static/context/dephasing.png
          :width: 100%
          :alt: Dephasing diagram
     - **Physical Meaning**
       
       Dephasing causes loss of coherence without changing populations. This term is used only in the relaxation of density matrix formalism (see :ref:`population_description`)
       
       **Mathematical Form**
       
       A vector :math:`\boldsymbol{\gamma}` where :math:`\gamma_i` is the dephasing rate for level :math:`|i\rangle`.
       
       **Only relevant** for density matrix calculations.

**Example: T₂ Relaxation in Density Matrix Dynamics**

.. code-block:: python

   import torch
   from mars import population
   
   # Define level-specific dephasing rates
   # Faster dephasing for higher energy states
   dephasing = torch.tensor([5e3, 8e3, 12e3])  # s^-1
   
   # Level 0: T₂ ~ 200 μs
   # Level 1: T₂ ~ 125 μs
   # Level 2: T₂ ~ 83 μs
   
   context = population.Context(
       sample=triplet_sample,
       basis="eigen",
       init_populations=[0.35, 0.30, 0.35],
       dephasing=dephasing,
       out_probs=torch.tensor([60.0, 60.0, 60.0])
   )

**Effect on Coherences**

For off-diagonal density matrix elements :math:`\rho_{ij}` (where :math:`i \neq j`):

.. math::

   \frac{d\rho_{ij}}{dt} = \cdots - \frac{\gamma_i + \gamma_j}{2} \rho_{ij}

**Transformation Rule**

Same as out probabilities:

.. math::

   \boldsymbol{\gamma}' = |U|^2 \cdot \boldsymbol{\gamma}

Combined Example
-----------------

A realistic triplet state with all relaxation mechanisms:

.. code-block:: python

   import torch
   from mars import spin_system, population
   
   # Create triplet system
   g_tensor = spin_system.Interaction(2.004, dtype=torch.float64)
   zfs = spin_system.DEInteraction([450e6, -90e6], dtype=torch.float64)
   
   triplet_system = spin_system.SpinSystem(
       electrons=[1.0],
       g_tensors=[g_tensor],
       electron_electron=[(0, 0, zfs)]
   )
   
   sample = spin_system.MultiOrientedSample(
       spin_system=triplet_system,
       ham_strain=2.5e7,
       gauss=0.0012,
       lorentz=0.0012
   )
   
   # Initial selective population (spin polarization)
   init_populations = [0.7, 0.05, 0.25]  # Strong TX population
   
   # Outgoing probabilities in ZFS basis
   out_probs = torch.tensor([100.0, 45.0, 70.0])  # s^-1
   
   # Spin-lattice relaxation in eigen basis  
   free_probs = torch.tensor([
       [0.0,    900.0,   80.0],
       [900.0,  0.0,     900.0],
       [80.0,   900.0,   0.0]
   ])  # s^-1
   
   # Dephasing rates
   dephasing = torch.tensor([6e3, 9e3, 11e3])  # s^-1
   
   # Combine in ZFS basis for populations, eigen for transitions
   context_zfs = population.Context(
       sample=sample,
       basis="zfs",
       init_populations=init_populations,
       out_probs=out_probs
   )
   
   context_eigen = population.Context(
       sample=sample, 
       basis="eigen",
       free_probs=free_probs,
       dephasing=dephasing
   )
   
   # Combine both mechanisms
   context_total = context_zfs + context_eigen