... _strain-management-mars:

Strain in MarS
=======================

"Strain" in MarS refers to the *distribution of spin Hamiltonian parameters* due to structural disorder (e.g., in frozen solutions, powders, or amorphous matrices). This leads to inhomogeneous broadening of EPR lines.

Strain is specified in MarS as the full width at half maximum (FWHM) of the parameter distribution and is expressed in the natural units of the corresponding Hamiltonian parameter (dimensionless for the g-tensor, Hz for zero-field splitting).

Strain vs. Explicit Modeling
-----------------------------

To understand strain modeling, consider two approaches to simulate disorder in a sample:

**Explicit Modeling (Monte Carlo)**

In explicit modeling, you would generate many spin systems with different parameter values drawn from a distribution, compute the spectrum for each, and average:

.. code-block:: python

   import torch
   from mars import spin_system
   
   # Generate 1000 samples from a Gaussian distribution
   n_samples = 1000
   g_mean = torch.tensor([2.002, 2.004, 2.008])
   g_std = torch.tensor([0.001, 0.001, 0.002]) / (2*math.sqrt(2*math.log(2)))  # to convert σ into FWHM
   
   # Sample g-values from Gaussian distribution
   g_samples = g_mean + g_std * torch.randn(n_samples, 3)
   
   # Create 1000 different spin systems
   spin_systems = []
   for i in range(n_samples):
       g = spin_system.Interaction(g_samples[i])
       spin_systems.append(g)
   
   # Compute spectrum for each and average

**Strain Modeling (Analytical)**

Strain modeling achieves the same result analytically using second-order perturbation theory, avoiding the need to sample many configurations:

.. code-block:: python

   import torch
   from mars import spin_system
   
   # Single spin system with strain parameters
   g = spin_system.Interaction(
       components=(2.002, 2.004, 2.008),
       strain=(0.001, 0.001, 0.002)  # FWHM of distributions
   )
   
   # Compute broadening analytically

In strain modeling, MarS computes how parameter variations affect transition energies through derivatives of the Hamiltonian matrix elements. For a transition between states :math:`|u\rangle` and :math:`|v\rangle`, the linewidth contribution is:

.. math::

   \Delta\nu_{uv} \propto \sum_{i=x,y,z} \left( \frac{\partial H_{uv}}{\partial g_i} \right)^2 \sigma_i^2

where :math:`\sigma_i` is the standard deviation of the :math:`i`-th parameter (related to FWHM by :math:`\sigma = \text{FWHM} / (2\sqrt{2\ln 2})`).

This approach is equivalent to Monte Carlo sampling in the limit of small strain, but requires only a single computation.

How Strain Works
----------------

For any interaction, strain is modeled as a Gaussian distribution of the principal values. In the case of :class:`mars.spin_system.Interaction`, strain is applied independently to :math:`T_x, T_y, T_z`. For :class:`mars.spin_system.DEInteraction`, strain is applied to the underlying :math:`D` and :math:`E` parameters.

During simulation, MarS integrates over this distribution analytically (via second-order perturbation theory) when computing the spectrum, avoiding costly Monte Carlo sampling. The result is a broadened lineshape consistent with Gaussian disorder in the Hamiltonian parameters.

Specifying Strain
-----------------

Strain is passed as a vector with the same length as the components:

.. code-block:: python

   # Isotropic g-strain
   g = Interaction(2.0023, strain=0.001)

   # Anisotropic g-strain
   g = Interaction((2.02, 2.04, 2.06), strain=(0.01, 0.01, 0.02))

   # ZFS strain on D and E
   zfs = DEInteraction([500e6, 100e6], strain=[50e6, 10e6])

Units must match those of the components (e.g., Hz for couplings, dimensionless for g).

Important Notes
---------------

- Strain in MarS is **uncorrelated by default** for :class:`mars.spin_system.Interaction` (diagonal covariance).
- Strain distributions are assumed Gaussian and static.
- MarS allows you to set any type of correlation within a single interaction using the :meth:`set_strain` method: ``interaction.set_strain(new_strain, new_correlation_matrix)``

Custom Strain Correlations
--------------------------

MarS allows you to define arbitrary linear correlations between the underlying physical parameters and the tensor components via the ``strain_correlation`` attribute.

Physical Motivation
~~~~~~~~~~~~~~~~~~~

Suppose you model a system where g-strain arises from two independent structural modes:

- A **symmetric compression** affecting all axes equally (isotropic shift)
- A **tetragonal distortion** affecting only :math:`g_z`

You can represent this with two strain parameters :math:`\delta s` (symmetric) and :math:`\delta t` (tetragonal), leading to:

.. math::

   \begin{bmatrix}
   \delta g_x \\ \delta g_y \\ \delta g_z
   \end{bmatrix}
   =
   \begin{bmatrix}
   1 & 0 \\
   1 & 0 \\
   1 & 1
   \end{bmatrix}
   \begin{bmatrix}
   \delta s \\ \delta t
   \end{bmatrix}

Mathematical Framework
~~~~~~~~~~~~~~~~~~~~~~

In MarS, the strain broadening is computed from the derivatives of the Hamiltonian matrix elements :math:`H_{uv}` with respect to the physical parameters. For a transition between states :math:`|u\rangle` and :math:`|v\rangle`, the field-dependent contribution to the linewidth is proportional to:

.. math::

   \sum_{\alpha=s,t} \left( \frac{\partial H_{uv}}{\partial p_\alpha} \right)^2 \sigma_\alpha^2
   =
   \sum_{\alpha=s,t} \left( \sum_{i=x,y,z} \frac{\partial H_{uv}}{\partial g_i} \frac{\partial g_i}{\partial p_\alpha} \right)^2 \sigma_\alpha^2

where :math:`p_\alpha = [s, t]`, and the Jacobian :math:`\partial g_i / \partial p_\alpha` is provided by the ``strain_correlation`` matrix.

Implementation in MarS
~~~~~~~~~~~~~~~~~~~~~~

In MarS, this is implemented as follows:

.. code-block:: python

   import torch
   from mars import spin_system

   # Define base g-tensor
   g = spin_system.Interaction(
       components=(2.002, 2.004, 2.008),
       strain=torch.tensor([0.001, 0.002])  # Two strain parameters: σ_s, σ_t
   )

   # Set custom correlation matrix C = ∂g_i / ∂p_α (shape: 3 × 2)
   C = torch.tensor([
       [1.0, 0.0],  # ∂g_x/∂s = 1, ∂g_x/∂t = 0
       [1.0, 0.0],  # ∂g_y/∂s = 1, ∂g_y/∂t = 0
       [1.0, 1.0]   # ∂g_z/∂s = 1, ∂g_z/∂t = 1
   ])
   g.set_strain(strain=torch.tensor([0.001, 0.002]), correlation_matrix=C)

Examples
--------

Example 1: Uncorrelated g-strains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When g-tensor components vary independently (default behavior):

.. code-block:: python

   import torch
   from mars import spin_system

   # Three independent Gaussian distributions for gx, gy, gz
   g = spin_system.Interaction(
       components=(2.002, 2.004, 2.008),
       strain=(0.001, 0.001, 0.002)
   )

   # Correlation matrix is identity (3×3):
   # [[1, 0, 0],
   #  [0, 1, 0],
   #  [0, 0, 1]]
   print(g.strain_correlation)

This represents three independent strain sources, each affecting only one principal axis.

Example 2: Fully correlated g-strains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When all g-tensor components shift together (isotropic strain):

.. code-block:: python

   import torch
   from mars import spin_system

   # Single strain parameter affects all components equally
   g = spin_system.Interaction(
       components=(2.002, 2.004, 2.008),
       strain=torch.tensor([0.001])  # Single strain parameter
   )

   # Correlation matrix (3×1):
   # [[1],
   #  [1],
   #  [1]]
   C = torch.tensor([[1.0], [1.0], [1.0]])
   g.set_strain(strain=torch.tensor([0.001]), correlation_matrix=C)

This means:

.. math::

   \begin{bmatrix}
   \delta g_x \\ \delta g_y \\ \delta g_z
   \end{bmatrix}
   =
   \begin{bmatrix}
   1 \\ 1 \\ 1
   \end{bmatrix}
   \delta s

All components fluctuate together with the same magnitude.