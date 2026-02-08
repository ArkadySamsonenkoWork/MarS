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

**Implementation:** Uses :func:`mars.population.transform.get_transformation_probabilities`
and :func:`mars.population.transform.transform_state_weights_to_new_basis`:

.. code-block:: python

   from mars.population.transform import (
       get_transformation_probabilities,
       transform_state_weights_to_new_basis
   )
   
   # Get |U|^2 coefficients
   probabilities = get_transformation_probabilities(basis_old, basis_new)
   # Shape: [..., N, N] with elements |⟨new_i|old_j⟩|²
   
   # Transform populations
   populations_old = torch.tensor([0.5, 0.3, 0.2])
   populations_new = transform_state_weights_to_new_basis(populations_old, probabilities)
   
   # Transform out probabilities
   out_probs_old = torch.tensor([100.0, 50.0, 75.0])
   out_probs_new = transform_state_weights_to_new_basis(out_probs_old, probabilities)

**Physical interpretation:** If state :math:`|i'\rangle` in the new basis is a superposition :math:`|i'\rangle = \sum_k U^{*}_{ik} |k\rangle`,
then its population is the sum of populations with :math:`|U^{*}_{ik}|^2` which is equal to :math:`|U_{ik}|^2`.


.. note::

   If you set the initial density as parameter of Context, then the denisty will be tranformed under desnity transforamtion rule (see futher),
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

**Implementation:** Uses :func:`mars.population.transform.transform_rate_matrix_to_new_basis`:

.. code-block:: python

   from mars.population.transform import transform_rate_matrix_to_new_basis
   # Transition probabilities between singlet state and triplet sublevels
   W_old = torch.tensor([
       [0.0,    1000.0,   1000.0, 1000.0],
       [1000.0,    0.0,   0.0,   0.0],
       [1000.0,    0.0,   0.0,   0.0],
       [1000.0,    0.0,   0.0,   0.0],
   ])
   
   # Transform to new basis
   W_new = transform_rate_matrix_to_new_basis(W_old, probabilities)
   # Applies: probabilities @ W_old @ probabilities.T

Density Matrix
^^^^^^^^^^^^^^

For full density matrix calculations:

.. math::

   \hat{\rho}' = U \hat{\rho} U^\dagger

or transforms of vectorized density matrix in Liouville space:

.. math::

   |\rho'\rangle\rangle = (U \otimes U^*) |\rho\rangle\rangle:

where :math:`\otimes` denotes the Kronecker product and :math:`U^*` is the element-wise complex conjugate of :math:`U`.

**Implementation:** Uses :func:`mars.population.transform.transform_operator_to_new_basis`:

.. code-block:: python

   from mars.population.transform import (
       transform_operator_to_new_basis
   )
   
   # Get transformation matrix (complex, not squared)
   U = basis_transformation(basis_old, basis_new)
   # This is just: basis_new.conj().T @ basis_old
   
   # Transform density matrix
   rho_old = torch.tensor([[0.6, 0.1+0.2j],
                           [0.1-0.2j, 0.4]])
   rho_new = transform_operator_to_new_basis(rho_old, U)
   # Applies: U @ rho_old @ U.conj().T


Example usage
-------------

Compute the Liouville-space transformation between eigenbases of two spin Hamiltonians and apply it to a relaxation superoperator:

.. code-block:: python

   import torch
   from mars.population.transform import (
       basis_transformation,
       compute_liouville_basis_transformation,
   )

   basis_old = torch.tensor([
       [1.0, 0.0],
       [0.0, 1.0]
   ], dtype=torch.complex64)

   basis_new = torch.tensor([
       [1.0,  1.0],
       [1.0, -1.0]
   ], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))

   # 1. Get Hilbert-space transformation (for operators like ρ)
   U = basis_transformation(basis_old, basis_new)
   # Equivalent to: U = basis_new.conj().T @ basis_old

   # 2. Get Liouville-space transformation (for vectorized ρ or superoperators)
   T = compute_liouville_basis_transformation(basis_old, basis_new)
   # Returns: kron(U, U.conj()) with shape (4, 4) for K=2

   # 3. Transform a density matrix (Hilbert space)
   rho_old = torch.tensor([[0.6, 0.1+0.2j],
                           [0.1-0.2j, 0.4]], dtype=torch.complex64)
   rho_new = U @ rho_old @ U.conj().T

   # 4. Transform via vectorization (Liouville space) – equivalent result
   rho_old_vec = rho_old.flatten()          # Row-major: [0.6, 0.1+0.2j, 0.1-0.2j, 0.4]
   rho_new_vec = T @ rho_old_vec
   assert torch.allclose(rho_new_vec.reshape(2, 2), rho_new)

.. note::

   - For the transformation of operators from Hilbert space to Liouville space, MarS uses **row-major** (C-order) vectorization.
     This corresponds to flattening the density matrix by stacking its rows sequentially—the default behavior in NumPy and PyTorch.
     For example, in a 2×2 system, the vectorized density matrix appears as:
     ``[ρ₀₀, ρ₀₁, ρ₁₀, ρ₁₁]``.

   - The transformation matrix for the vectorized density matrix, :math:`T = U \otimes U^*`, is unitary whenever :math:`U` is unitary.

Notes
-----

* The function assumes row-major (C-order) vectorization, consistent with PyTorch/NumPy ``.flatten()``.
* The returned for vectorized density matrix transformation :math:`T=U \otimes U^*` is unitary when :math:`U` is unitary.

Relaxation Superoperator
^^^^^^^^^^^^^^^^^^^^^^^^^

The relaxation superoperator transforms in Liouville space:

.. math::

   \hat{\mathcal{R}}' = (U \otimes U^*) \hat{\mathcal{R}} (U^\dagger \otimes U^T)

**Implementation:** Uses :func:`mars.population.transform.compute_liouville_basis_transformation` and :func:`mars.population.transform.transform_superop_to_new_basis`:

.. code-block:: python

   from mars.population.transform import (
       compute_liouville_basis_transformation,
       transform_superop_to_new_basis
   )
   
   # Get Liouville space transformation
   T_liouville = compute_liouville_basis_transformation(basis_old, basis_new)
   # Shape: [..., N², N²]
   # This is: kron(U, U.conj())
   
   # Transform superoperator
   R_old = relaxation_superop  # Shape: [..., N², N²]
   R_new = transform_superop_to_new_basis(R_old, T_liouville)
   # Applies: T @ R_old @ T.conj().T

Basis Transformation in Multiplied Contexts
-------------------------------------------

When constructing composite systems via Kronecker multiplication, each subsystem is typically defined in its own "intra-basis" (e.g., molecular frame, ZFS basis, zeeman basis).
Consider two subsystems with initial bases :math:`V^{(1)}_{\text{old}}` and :math:`V^{(2)}_{\text{old}}`. If the subsystems interact weakly, the composite eigenbasis may still decompose as:

.. math::

   V_{\text{new}} = V^{(1)}_{\text{new}} \otimes V^{(2)}_{\text{new}}

where the individual transformations are:

.. math::

   U_1 = (V^{(1)}_{\text{new}})^\dagger V^{(1)}_{\text{old}}, \quad
   U_2 = (V^{(2)}_{\text{new}})^\dagger V^{(2)}_{\text{old}}
	

Density Composition
^^^^^^^^^^^^^^^^^^^

For density matrices, the composite transformation follows the unitary rule:

.. math::

   \rho_{\text{total}} \rightarrow (U_1 \otimes U_2) \, \rho_{\text{total}} \, (U_1 \otimes U_2)^\dagger

Superoperator Composition
^^^^^^^^^^^^^^^^^^^^^^^^^^

For relaxation superoperators in Liouville space, two sequential operations are required:

1. Permutation of the Kronecker-sum structure to reconcile vectorization ordering. The identity

   .. math::
   
      \operatorname{vec}(\rho_1 \otimes \rho_2) \neq \operatorname{vec}(\rho_1) \otimes \operatorname{vec}(\rho_2)
   
   necessitates a commutation matrix :math:`\Pi` such that
   
   .. math::
   
      \operatorname{vec}(\rho_1 \otimes \rho_2) = \Pi \bigl[ \operatorname{vec}(\rho_1) \otimes \operatorname{vec}(\rho_2) \bigr]
   
   This permutation must be applied to the composite superoperator *before* basis transformation:
   
   .. math::
   
      \hat{\mathcal{R}}_{\text{initial}} = \Pi \,
      \bigl[ \hat{\mathcal{R}}^{(1)} \otimes \hat{\mathbb{I}}^{(2)} + \hat{\mathbb{I}}^{(1)} \otimes \hat{\mathcal{R}}^{(2)} \bigr] \,
      \Pi^\top

2. Joint basis transformation using the composite unitary matrix. After permutation, the superoperator transforms as:
   
   .. math::
   
      \hat{\mathcal{R}}_{\text{total}} \rightarrow
      \bigl[(U_1 \otimes U_2) \otimes (U_1^* \otimes U_2^*)\bigr] \;
      \hat{\mathcal{R}}_{\text{initial}} \;
      \bigl[(U_1^\dagger \otimes U_2^\dagger) \otimes (U_1^T \otimes U_2^T)\bigr]
   
   The transformation matrix
   
   .. math::
   
      T = (U_1 \otimes U_2) \otimes (U_1^* \otimes U_2^*)
   
   applies a joint change of basis to the vectorized composite state.

Population Composition
^^^^^^^^^^^^^^^^^^^^^^

Populations and transition probabilities transform differently.
They follow probability rules rather than amplitude rules. For a population vector :math:`\mathbf{n}`:

.. math::

   \mathbf{n} \rightarrow |U|^2 \mathbf{n}, \quad

Importantly, for Kronecker products the element-wise squaring distributes exactly:

.. math::

   |U_1 \otimes U_2|^2 = |U_1|^2 \otimes |U_2|^2

Furthermore, for any unitary matrix :math:`U`, the probability matrix :math:`|U|^2` satisfies:

.. math::

   |U|^2 \, \mathbf{1} = \mathbf{1}

where :math:`\mathbf{1}` is the vector of ones. This holds because unitary matrices have orthonormal rows (:math:`\sum_j |U_{ij}|^2 = 1` for all :math:`i`), so each row of :math:`|U|^2` sums to unity.

Therefore populations and outgoing probabilities can be transformed separately in each subsystem and then combined via Kronecker product. The transformation is separable:

.. math::

   \mathbf{n}_{\text{new}} = (|U_1|^2 \mathbf{n}^{(1)}) \otimes (|U_2|^2 \mathbf{n}^{(2)}) = (|U_1 \otimes U_2|^2 (\mathbf{n}^{(1)} \otimes \mathbf{n}^{(2)})

.. math::

   \boldsymbol{\Gamma}_{\text{new}} = (|U_1|^2 \boldsymbol{\Gamma}^{(1)}) \otimes \mathbf{1}^{(2)} + \mathbf{1}^{(1)} \otimes (|U_2|^2 \boldsymbol{\Gamma}^{(2)}) = (|U_1 \otimes U_2|^2 (\boldsymbol{\Gamma}^{(1)} \otimes \mathbf{1}^{(2)} + \mathbf{1}^{(1)} \otimes \boldsymbol{\Gamma}^{(2)})


Transition Probabilities Compositions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

However, for transition probability matrices the situation is more subtle.
The full transformation of the Kronecker-sum structure does not factorize cleanly:

.. math::

   \begin{aligned}
   &|U_1 \otimes U_2|^2 \, (K_1 \otimes \mathbb{I}_2 + \mathbb{I}_1 \otimes K_2) \, |U_1 \otimes U_2|^2{}^\top \\
   &= (|U_1|^2 \otimes |U_2|^2) (K_1 \otimes \mathbb{I}_2) (|U_1|^2{}^\top \otimes |U_2|^2{}^\top) + (|U_1|^2 \otimes |U_2|^2) (\mathbb{I}_1 \otimes K_2) (|U_1|^2{}^\top \otimes |U_2|^2{}^\top) \\
   &= (|U_1|^2 K_1 |U_1|^2{}^\top) \otimes (|U_2|^2 |U_2|^2{}^\top) + (|U_1|^2 |U_1|^2{}^\top) \otimes (|U_2|^2 K_2 |U_2|^2{}^\top)
   \end{aligned}

Since :math:`|U|^2` is a doubly probability matrix (not unitary), we generally have :math:`|U|^2 |U|^2{}^\top \neq \mathbb{I}`. Therefore:

.. math::

   (|U_1|^2 K_1 |U_1|^2{}^\top) \otimes (|U_2|^2 |U_2|^2{}^\top) \neq (|U_1|^2 K_1 |U_1|^2{}^\top) \otimes \mathbb{I}_2

This non-factorization means that transforming transition probabilities separately and then forming the Kronecker sum does not generally yield the same result
as forming the Kronecker sum first and then transforming the composite operator.

Nevertheless, MarS consistently interprets transition probabilities as probabilities for "state-to-state processes":
a transition from initial state :math:`|j\rangle` to final state :math:`|i\rangle` occurs with probability weight determined by both the initial-state overlap :math:`|U_{\beta j}|^2` and the final-state overlap :math:`|U_{\alpha i}|^2`.

Consequently, the MarS transformation for the composite system applies the joint probability rule to the full operator:

.. math::

   K'_{\text{total}} = |U_1 \otimes U_2|^2 \, \bigl(K_1 \otimes \mathbb{I}_2 + \mathbb{I}_1 \otimes K_2\bigr) \, |U_1 \otimes U_2|^2{}^\top

This transformation rule implies that even for two physically independent relaxation processes, the relaxation operator of the multiplied system differs from the Kronecker sum of independently transformed subsystem operators.
The bilinear dependence on both initial and final state overlaps couples the transformations, making the composite relaxation non-separable under basis change.


Here we highlight that such complexity and ambiguity of interpretation arises only for "free_probs" and "driven_probs" attributes. For the remaining parameters (including superoperators), the transformation is determined unambiguously.

For composite systems, MarS implements the dedicated :class:`mars.population.contexts.KroneckerContext`.
This class computes exact transformation coefficients between the product basis of subsystems and the true composite eigenbasis.

Let :math:`|\alpha\rangle` denote an eigenstate of the full composite Hamiltonian, expanded in the product basis of subsystems:

.. math::

   |\alpha\rangle = \sum_{i,j} c^{(\alpha)}_{ij} \; |i\rangle \otimes |j\rangle

where the expansion coefficients are given by:

.. math::

   c^{(\alpha)}_{ij} = \langle\alpha|\bigl(|i\rangle \otimes |j\rangle\bigr) = U\bigl[\alpha,\; \text{index}(i,j)\bigr]

with the full transformation matrix:

.. math::

   U = V_{\text{new}}^\dagger \bigl(V^{(1)}_{\text{old}} \otimes V^{(2)}_{\text{old}}\bigr)

The squared magnitudes :math:`|c^{(\alpha)}_{ij}|^2` represent Clebsch-Gordan probability coefficients—the probability that composite eigenstate :math:`|\alpha\rangle` contains the product-state component :math:`|i\rangle \otimes |j\rangle`. Transition probabilities then transform according to the joint-probability rule:

.. math::

   W'_{\alpha\beta} = \sum_{i,j,k,l} |c^{(\alpha)}_{ij}|^2 \; W_{(ij),(kl)} \; |c^{(\beta)}_{kl}|^2

preserving the interpretation of :math:`W_{ij}` as the probability for transitions from initial state :math:`j` to final state :math:`i`.

The :class:`mars.population.contexts.KroneckerContext` automatically computes these coefficients from the composite contexts.

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

