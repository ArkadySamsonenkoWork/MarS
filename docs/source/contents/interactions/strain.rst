.. _strain-managment-mars:

Strain in MarS
=======================

"Strain" in MarS refers to the **distribution of spin Hamiltonian parameters** due to structural disorder (e.g., in frozen solutions, powders, or amorphous matrices). This leads to inhomogeneous broadening of EPR lines.

How Strain Works
----------------

For any interaction, strain is modeled as a **Gaussian distribution** of the principal values. In the case of :class:`Interaction`, strain is applied independently to :math:`T_x, T_y, T_z`. For :class:`DEInteraction`, strain is applied to the underlying :math:`D` and :math:`E` parameters, then propagated to :math:`D_x, D_y, D_z` via the physical transformation.

During simulation, MarS integrates over this distribution **analytically** (via second-order perturbation theory) when computing the spectrum, avoiding costly Monte Carlo sampling. The result is a broadened lineshape consistent with Gaussian disorder in the Hamiltonian parameters.

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

- Strain in MarS is **uncorrelated by default** for :class:`Interaction` (diagonal covariance).
- Strain distributions are assumed Gaussian and static (no motional narrowing).
- MarS allows to set any type of correlation within a single interaction. You can set the attribute `interaction.strain_correlation = 'Correlation matrix`.

Custom Strain Correlations
--------------------------

MarS allows you to define arbitrary linear correlations between the underlying physical parameters and the tensor components via the ``strain_correlation`` attribute. This is a matrix :math:`\mathbf{C}` such that:

Suppose you model a system where g-strain arises from two independent structural modes:  
- A symmetric compression affecting all axes equally (isotropic shift),  
- A tetragonal distortion affecting only :math:`g_z`.  

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
   \end{bmatrix}.  

In MarS, the strain broadening is computed from the derivatives of the Hamiltonian matrix elements :math:`H_{uv}` with respect to the physical parameters. For a transition between states :math:`|u\rangle` and :math:`|v\rangle`, the field-dependent contribution to the linewidth is proportional to  

.. math::  

   \sum_{\alpha=s,t} \left( \frac{\partial H_{uv}}{\partial p_\alpha} \right)^2 \sigma_\alpha^2  
   =  
   \sum_{\alpha=s,t} \left( \sum_{i=x,y,z} \frac{\partial H_{uv}}{\partial g_i} \frac{\partial g_i}{\partial p_\alpha} \right)^2 \sigma_\alpha^2,  

where :math:`p_\alpha = [s, t]`, and the Jacobian :math:`\partial g_i / \partial p_\alpha` is provided by the ``strain_correlation`` matrix.  

In MarS, this is implemented as follows:  

.. code-block:: python  

   import torch  
   from mars import spin_system  

   # Define base g-tensor  
   g = spin_system.Interaction((2.002, 2.004, 2.008), strain=torch.tensor([0.001, 0.001, 0.002]))  

   # Set custom correlation matrix C = ∂g_i / ∂p_α (shape: 3 × 2)  
   C = torch.tensor([  
       [1.0, 0.0],  # ∂g_x/∂s = 1, ∂g_x/∂t = 0  
       [1.0, 0.0],  # ∂g_y/∂s = 1, ∂g_y/∂t = 0  
       [1.0, 1.0]   # ∂g_z/∂s = 1, ∂g_z/∂t = 1  
   ])  
   g.strain_correlation = C  
