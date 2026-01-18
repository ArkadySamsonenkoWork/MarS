.. _basis_transformation:

Basis Transformations
======================

The Context class in MarS supports defining relaxation parameters in multiple bases. All parameters are automatically transformed to the eigenbasis of the full Hamiltonian in the magnetic field before computation.
This flexibility allows users to specify relaxation mechanisms in the most physically intuitive representation.

Supported Bases
---------------

MarS provides four predefined bases:

1. **"eigen"** (default)
   
   The eigenbasis of the full Hamiltonian :math:`\hat{H}_0 = \hat{F} + g\mu_B B_z \hat{S}_z` at the resonance magnetic field.
   
   - States are sorted by increasing energy
   - This is the natural basis for most EPR calculations
   - No transformation needed if parameters are already defined here

2. **"zfs"**
   
   The eigenbasis of the zero-field splitting operator :math:`\hat{F}` (the field-independent part of the Hamiltonian).
   
   - States are sorted by increasing zero-field energy
   - Useful for triplet states: :math:`|T_X\rangle, |T_Y\rangle, |T_Z\rangle`

3. **"multiplet"**
   
   The total spin basis :math:`|S, M\rangle`, where:
   
   - :math:`S` is the total spin quantum number
   - :math:`M` is the projection of total spin on the z-axis
   - States sorted by increasing :math:`S`, then increasing :math:`M`

4. **"product"**
   
   The uncoupled basis :math:`|m_1, m_2, \ldots, m_n\rangle` of individual spin projections.
   
   - :math:`m_k` is the projection of the :math:`k`-th electron spin
   - States sorted by **decreasing** spin projections

Custom Transformation Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users can also provide a custom transformation matrix :math:`U` with shape :math:`[\ldots, R, 1, N, N]`:

- :math:`R` is the number of orientations
- :math:`N` is the spin system dimension
- Matrix transforms from custom basis to target basis

Transformation Rules
--------------------

Let :math:`V_{\text{new}}` be the eigenbasis of the full Hamiltonian and :math:`V_{\text{old}}` be the user-specified basis. The transformation matrix is:

.. math::

   U = V_{\text{new}}^\dagger V_{\text{old}}

This is computed using :func:`mars.population.transform.basis_transformation`:

.. code-block:: python

   from mars.population.transform import basis_transformation
   
   # basis_new and basis_old are tensors of shape [..., N, N]
   # Columns are eigenvectors
   U = basis_transformation(basis_old, basis_new)
   # Result: transformation matrix [..., N, N]

Different quantities transform according to their physical nature:


Populations and Out Probabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are scalar quantities associated with individual states and transform via probability conservation:

.. math::

   \mathbf{n}' = |U|^2 \cdot \mathbf{n}

.. math::

   \mathbf{o}' = |U|^2 \cdot \mathbf{o}

where :math:`|U|^2` denotes element-wise squaring of the transformation matrix.

**Implementation:** Uses :func:`mars.population.transform.get_transformation_coeffs` and :func:`mars.population.transform.transform_vector_to_new_basis`:

.. code-block:: python

   from mars.population.transform import (
       get_transformation_coeffs,
       transform_vector_to_new_basis
   )
   
   # Get |U|^2 coefficients
   coeffs = get_transformation_coeffs(basis_old, basis_new)
   # Shape: [..., N, N] with elements |⟨new_i|old_j⟩|²
   
   # Transform populations
   populations_old = torch.tensor([0.5, 0.3, 0.2])
   populations_new = transform_vector_to_new_basis(populations_old, coeffs)
   
   # Transform out probabilities
   out_probs_old = torch.tensor([100.0, 50.0, 75.0])
   out_probs_new = transform_vector_to_new_basis(out_probs_old, coeffs)

**Physical interpretation:** If state :math:`|i'\rangle` in the new basis is a superposition :math:`|i'\rangle = \sum_k U_{ik} |k\rangle`, then its population is the sum of populations with :math:`|U_{ik}|^2`.

Transition Probabilities (Free and Driven)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transition rates between states transform as bilinear forms:

.. math::

   W' = |U|^T \cdot W \cdot |U|

.. math::

   D' = |U|^T \cdot D \cdot |U|


**Implementation:** Uses :func:`mars.population.transform.transform_matrix_to_new_basis`:

.. code-block:: python

   from mars.population.transform import transform_matrix_to_new_basis
   # Transition probabilities between singlet state and triplet sublevels
   W_old = torch.tensor([
       [0.0,    1000.0,   1000.0, 1000.0],
       [1000.0,    0.0,   0.0,   0.0],
       [1000.0,    0.0,   0.0,   0.0],
       [1000.0,    0.0,   0.0,   0.0],
   ])
   
   # Transform to new basis
   W_new = transform_matrix_to_new_basis(W_old, coeffs)
   # Applies: coeffs @ W_old @ coeffs.T


Density Matrix
^^^^^^^^^^^^^^

For full density matrix calculations:

.. math::

   \hat{\rho}' = U \hat{\rho} U^\dagger

or in Liouville space:

.. math::

   |\rho'\rangle\rangle = (U \otimes U^*) |\rho\rangle\rangle

**Implementation:** Uses :func:`mars.population.transform.compute_density_basis_transformation` and :func:`mars.population.transform.transform_density`:

.. code-block:: python

   from mars.population.transform import (
       compute_density_basis_transformation,
       transform_density
   )
   
   # Get transformation matrix (complex, not squared)
   U = compute_density_basis_transformation(basis_old, basis_new)
   # This is just: basis_new.conj().T @ basis_old
   
   # Transform density matrix
   rho_old = torch.tensor([[0.6, 0.1+0.2j],
                           [0.1-0.2j, 0.4]])
   rho_new = transform_density(rho_old, U)
   # Applies: U @ rho_old @ U.conj().T

Relaxation Superoperator
^^^^^^^^^^^^^^^^^^^^^^^^^

The relaxation superoperator transforms in Liouville space:

.. math::

   \hat{\mathcal{R}}' = (U \otimes U^*) \hat{\mathcal{R}} (U^\dagger \otimes U^T)

**Implementation:** Uses :func:`mars.population.transform.compute_liouville_basis_transformation` and :func:`mars.population.transform.transform_liouville_superop`:

.. code-block:: python

   from mars.population.transform import (
       compute_liouville_basis_transformation,
       transform_liouville_superop
   )
   
   # Get Liouville space transformation
   T_liouville = compute_liouville_basis_transformation(basis_old, basis_new)
   # Shape: [..., N², N²]
   # This is: kron(U, U.conj())
   
   # Transform superoperator
   R_old = relaxation_superop  # Shape: [..., N², N²]
   R_new = transform_liouville_superop(R_old, T_liouville)
   # Applies: T @ R_old @ T.conj().T
