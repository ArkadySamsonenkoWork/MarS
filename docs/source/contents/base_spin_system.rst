Base Spin System
==============================
The :class:`mars.spin_system.SpinSystem` class provides a flexible representation of spin systems commonly encountered in EPR spectroscopy. It supports arbitrary numbers of electron and nuclear spins, along with their mutual interactions (hyperfine, dipolar, exchange, zfs).


Examples
--------

**1. Simple triplet system (S = 1) with axial zero-field splitting**

Define a single electron with spin 1 and a D-tensor aligned with the lab frame:

.. code-block:: python

   from mars import spin_system, particles

   g_tensor = spin_system.Interaction(2.002)  # g = 2.002
   D_tensor = spin_system.DEInteraction(components=0.35 * 1e9)  # D = 0.35 GHz
   triplet = spin_system.SpinSystem(electrons=[1.0], g_tensors=[g_tensor], electron_electron=[(0, 0, D_tensor)])

This creates a three-level system with Hamiltonian :math:`H = gβBSz + D(S_z^2 - S(S+1)/3)`.

**2. Two coupled doublets with anisotropic g-tensors**

Model two interacting S = 1/2 centers with distinct g-anisotropy and a dipolar coupling:

.. code-block:: python

   g1 = spin_system.Interaction([2.002, 2.004, 2.008])  # gx, gy, gz
   g2 = spin_system.Interaction([1.998, 2.000, 2.006])

   anisotropic = spin_system.Interaction([0.01 * 1e9, 0.0 * 1e9, 0.02 * 1e9])  # Dx, Dy, Dz

   system = spin_system.SpinSystem(
       electrons=[1/2, 1/2],
       g_tensors=[g1, g2],
       electron_electron=[(0, 1, anisotropic)]
   )

The total Hilbert space dimension is 4, and the dipolar term is added via scalar product of spin operators.

**3. Electron coupled to a nucleus (hyperfine interaction)**

Simulate a nitroxide radical with a ^14N nucleus (I = 1):

.. code-block:: python

   g_el = spin_system.Interaction([2.006, 2.006, 2.002])
   hyperfine = spin_system.Interaction([20.01 * 1e6, 20.0  * 1e6, 80.0  * 1e6]) # A-tensor in MHz
   system = spin_system.SpinSystem(
       electrons=[1/2],
       g_tensors=[g_el],
       nuclei=["14N"],
       electron_nuclei=[(0, 0, hyperfine)]
   )
	

The resulting system has dimension :math:`(2S+1)(2I+1) = 6`

Useful Features
---------------

:class:`mars.spin_system.SpinSystem` provides several utility methods for advanced quantum-mechanical analysis, including access to key spin operators and basis transformations. These are especially useful when working with total spin manifolds, symmetry-adapted bases, or custom spectral models.

Basis and Operator Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following methods allow you to construct and switch between common representations of the spin Hilbert space:

.. code-block:: python

   Mz = system.get_electron_z_operator()        # Total electron S_z operator
   S2 = system.get_electron_squared_operator()  # Total electron S² operator
   Mul = system.get_spin_multiplet_basis()      # |S, M⟩ basis (eigenbasis of S² and S_z)
   PR = system.get_product_state_basis()        # Computational |m₁, m₂, …⟩ basis
   Me = system.get_electron_projections()       # Electron-only Mₑ per product state
   Mt = system.get_total_projections()          # Total M = Σmₑ + Σmₙ per product state

Below is a detailed description of each method.

`:meth:`mars.spin_system.SpinSystem.get_electron_z_operator()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns the total electron spin projection operator along the *z*-axis:

.. math::

   \hat{S}_z = \sum_{i \in \text{electrons}} \hat{S}_{i}^{(z)}

- **Shape**: ``(spin_dim, spin_dim)``
- **Basis**: Product-state basis
- **Example**: For two spin-½ electrons, returns a 4×4 diagonal matrix with entries `[1, 0, 0, -1]`.

`:meth:`mars.spin_system.SpinSystem.get_electron_squared_operator()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns the total electron spin-squared operator:

.. math::

   \hat{S}^2 = \left( \sum_i \hat{\mathbf{S}}_i \right) \cdot \left( \sum_j \hat{\mathbf{S}}_j \right)
             = \hat{S}_x^2 + \hat{S}_y^2 + \hat{S}_z^2

- **Shape**: ``(spin_dim, spin_dim)``
- **Eigenvalues**: :math:`S(S+1)`, where :math:`S` is the total electron spin quantum number.
- **Example**: For two spin-½ electrons, eigenvalues are `0` (singlet) and `2` (triplet).

`:meth:`mars.spin_system.SpinSystem.get_spin_multiplet_basis()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Constructs a unitary transformation matrix that converts from the product-state basis to the total-spin multiplet basis :math:`|S, M\rangle`.

- **Output**: A matrix whose columns are eigenvectors of :math:`\hat{S}^2` and :math:`\hat{S}_z`, sorted first by :math:`S`, then by :math:`M`.
- **Ordering**: States are arranged in ascending order of total spin :math:`S`, and within each :math:`S` manifold, by increasing :math:`M`.
- **Example**: For two spin-½ electrons, the basis order is  
  :math:`|S=0, M=0\rangle,\ |S=1, M=-1\rangle,\ |S=1, M=0\rangle,\ |S=1, M=+1\rangle`.

`:meth:`mars.spin_system.SpinSystem.get_product_state_basis()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns the identity matrix, confirming that internal operators are represented in the standard product-state basis:

.. math::

   |\psi\rangle = |m_{e_1}, m_{e_2}, \dots, m_{n_1}, m_{n_2}, \dots\rangle

- **Shape**: ``(spin_dim, spin_dim)``
- **Purpose**: Useful as a reference or for explicit basis-change operations.

`:meth:`mars.spin_system.SpinSystem.get_electron_projections()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a 1D tensor containing the total electron magnetic quantum number :math:`M_e = \sum_i m_{e_i}` for each product state.

- **Ignores nuclear spins** (sets their projections to zero).
- **Shape**: ``(spin_dim,)``
- **Example**: For one electron (S=½) and one nucleus (I=½), returns `[0.5, 0.5, -0.5, -0.5]`.

`:meth:`mars.spin_system.SpinSystem.get_total_projections()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns the total magnetic quantum number :math:`M = \sum_i m_{e_i} + \sum_j m_{n_j}` for every product state.

- **Includes both electrons and nuclei**.
- **Shape**: ``(spin_dim,)``
- **Example**: Same system as above → `[1.0, 0.0, 0.0, -1.0]`.

These methods enable flexible manipulation of spin states-whether you need to analyze symmetries, project onto total-spin subspaces, or compute expectation values in specific bases.

