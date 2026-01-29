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
   - For the triplet system this is the same as  "xyz" basis but usually in another order (depending on D and E values): :math:`|T_Z\rangle, |T_X\rangle, |T_Y\rangle`

3. **"multiplet"**
   
   The total spin basis :math:`|S, M\rangle`, where:
   
   - :math:`S` is the total spin quantum number
   - :math:`M` is the projection of total spin on the z-axis
   - States sorted by increasing :math:`S`, then increasing :math:`M`

4. **"product"**
   
   The uncoupled basis :math:`|m_1, m_2, \ldots, m_n\rangle` of individual spin projections.
   
   - :math:`m_k` is the projection of the :math:`k`-th electron spin
   - States sorted by **decreasing** spin projections


4. **"xyz"**

   A molecular-frame basis commonly used for triplet (:math:`S = 1`) systems,
   constructed from symmetric combinations of the two-electron product states
   :math:`|\alpha\alpha\rangle`, :math:`|\alpha\beta\rangle`,
   :math:`|\beta\alpha\rangle`, and :math:`|\beta\beta\rangle`.
   After projection onto the total-spin :math:`S = 1` subspace, the orthonormal
   triplet states are:

   .. math::

      |T_X\rangle &= \frac{1}{\sqrt{2}}\bigl(-|\alpha\alpha\rangle + |\beta\beta\rangle\bigr) \\
      |T_Y\rangle &= \frac{i}{\sqrt{2}}\bigl(|\beta\beta\rangle + |\alpha\alpha\rangle\bigr) \\
      |T_Z\rangle &= \frac{1}{\sqrt{2}}\bigl(|\alpha\beta\rangle + |\beta\alpha\rangle\bigr)

5. **"zeeman"**

   The eigenbasis of the Z-projections of the Zeeman operator. It is the basis of the system in an infinite magnetic field :math:`\hat{Gz}`.
   - States are sorted by increasing energy


Custom Transformation Basis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users can also provide a custom transformation basis with shape :math:`[\ldots, R, 1, N, N]`: 

- :math:`R` is the number of orientations
- :math:`N` is the spin system dimension
- This basis should be defined in laboratory frame (with repspect to orientations) and in the basis of individual spin projections.

Specifying the Sample for Context Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When creating a :class:`mars.population.contexts.Context`, if you set some predefined basis, you must associate it with a "sample".

In most cases, the same sample is used both to define the initial relaxation parameters (and populations) and to compute the resulting spectrum.
However, advanced workflows allow you to:

- Define populations and rates using one sample
- Compute spectra for a different sample.

Both samples must have the same Hilbert space dimension and compatible orientation grids.
Notice, that if you choose "product", "multiplet", "xyz" basis, which doesn't depend on interactions,
you will get the same result for any specified sample with the same Hilbert dimension and the same particles.

Transformation Rules
--------------------

Let :math:`V_{\text{new}}` be the eigenbasis of the full Hamiltonian and :math:`V_{\text{old}}` be the specified basis.
The transformation matrix from coordinates in old basis to coordinates in a new basis is:

.. math::

   U = V_{\text{new}}^\dagger V_{\text{old}}

This is computed using :func:`mars.population.transform.basis_transformation`:

.. code-block:: python

   from mars.population.transform import basis_transformation
   
   # basis_new and basis_old are tensors of shape [..., N, N]
   # Columns are eigenvectors
   U = basis_transformation(basis_old, basis_new)
   # Result: transformation matrix [..., N, N]

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

**Physical interpretation:** If state :math:`|i'\rangle` in the new basis is a superposition :math:`|i'\rangle = \sum_k U^{*}_{ik} |k\rangle`,
then its population is the sum of populations with :math:`|U^{*}_{ik}|^2` which is equal to :math:`|U_{ik}|^2`.

**Note** If you set the initial density as parameter of Context, then the denisty will be tranformed under desnity transforamtion rule (see futher),
then the real part of transformed diagonal will be used as initial population if it is needed

Transition Probabilities (Free and Driven)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The library supports two types of transitions: **free** (spontaneous) and **driven** (induced), represented by matrices :math:`W` and :math:`D`, respectively.

Analogously to populations which transform as :math:`n'_i = \sum_k |U_{ik}|^2 n_k` - transition probabilities are transformed by weighting both the initial and final states with their respective overlap probabilities.
Specifically, the probability of transition from state :math:`j'` to :math:`i'` in the new basis is:

.. math::

   W'_{i'j'} = \sum_{k,l} |U_{ik}|^2 \, W_{kl} \, |U_{jl}|^2,

where :math:`|U_{ik}|^2 = |\langle i' | k \rangle|^2` is the probability that new basis state :math:`|i'\rangle` contains old basis state :math:`|k\rangle`.

In compact matrix form, this becomes:

.. math::

   W' = |U|^2 \, W \, (|U|^2)^\top,

and identically for driven transitions:

.. math::

   D' = |U|^2 \, D \, (|U|^2)^\top.


- **Free transition probabilities** describe spontaneous relaxation processes (e.g., spin-lattice relaxation).
- **Driven transition probabilities** model field-induced transitions (e.g., microwave-driven mixing).

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

Context Transformation Interface
----------------------------------

The :class:`mars.population.contexts.Context` class provides a convenient interface to transform all internally stored relaxation and initialization parameters into an arbitrary target basis.

Consider the following setup for a triplet spin system:

.. code-block:: python

    import torch
    from mars.population import Context

    # Define initial parameters in the "zfs" basis
    init_density = torch.tensor(
        [[0.2, 0.3, 0.5],
         [0.4, 0.3, 0.1],
         [0.2, 0.2, 0.5]],
        dtype=torch.complex128
    )

    out_probs = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype) * 100
    free_probs = torch.tensor([[0.0, 1.0, 0.0],
                               [1.0, 0.0, 1.0],
                               [0.0, 1.0, 0.0]], device=device, dtype=dtype) * 1000
    driven_probs = torch.tensor([[0.0, 1.0, 0.0],
                                 [1.0, 0.0, 1.0],
                                 [0.0, 1.0, 0.0]], device=device, dtype=dtype) * 1000

    context = Context(
        sample=triplet,
        basis="zfs",
        init_density=init_density,
        free_probs=free_probs,
        out_probs=out_probs,
        driven_probs=driven_probs,
        device=device,
        dtype=dtype
    )

Now, suppose we wish to evaluate the model at a low magnetic field (e.g., 10 mT). We first compute the eigenbasis of the full Hamiltonian:

.. code-block:: python

    F, _, _, Gz = triplet.get_hamiltonian_terms()
    field = 0.01  # 10 mT
    values, vectors = torch.linalg.eigh(F + field * Gz)
    vectors = vectors.unsqueeze(-3)  # Shape: [orientations, 1, N, N]

The ``vectors`` tensor now represents the target basis (columns are eigenvectors of the full Hamiltonian). The :class:`mars.population.contexts.Context` instance can transform all its internal quantities into this basis using the following methods:

- **Initial populations (diagonal of the density matrix):**  
  See :meth:`mars.population.contexts.Context.get_transformed_init_populations`.

  .. code-block:: python

      pop = context.get_transformed_init_populations(full_system_vectors=vectors)
      # Returns: tensor of shape [..., N]

- **Full initial density matrix:**  
  See :meth:`mars.population.contexts.Context.get_transformed_init_density`.

  .. code-block:: python

      rho = context.get_transformed_init_density(full_system_vectors=vectors)
      # Returns: tensor of shape [..., N, N]

- **Outgoing probabilities (scalar rates per state):**  
  See :meth:`mars.population.contexts.Context.get_transformed_out_probs`.

  .. code-block:: python

      out = context.get_transformed_out_probs(full_system_vectors=vectors)
      # Returns: tensor of shape [..., N]

- **Free transition probabilities (between states):**  
  See :meth:`mars.population.contexts.Context.get_transformed_free_probs`.

  .. code-block:: python

      W = context.get_transformed_free_probs(full_system_vectors=vectors)
      # Returns: tensor of shape [..., N, N]

- **Free relaxation superoperator (in Liouville space):**  
  See :meth:`mars.population.contexts.Context.get_transformed_free_superop`.

  .. code-block:: python

      R_free = context.get_transformed_free_superop(full_system_vectors=vectors)
      # Returns: tensor of shape [..., N², N²]

- **Driven relaxation superoperator (in Liouville space):**  
  See :meth:`mars.population.contexts.Context.get_transformed_driven_superop`.

  .. code-block:: python

      R_driven = context.get_transformed_driven_superop(full_system_vectors=vectors)
      # Returns: tensor of shape [..., N², N²]

