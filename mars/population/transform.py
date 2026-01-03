import math

import torch


def basis_transformation(basis_1: torch.Tensor, basis_2: torch.Tensor) -> torch.Tensor:
    """
    :param basis_1: The basis function. The shape is [..., K, K], where K is spin dimension size.
    The column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].

    :param basis_2: The basis function. The shape is [..., K, K], where K is spin dimension size.
    The column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].

    :return: A transformation matrix of shape [..., K, K] that transforms
            vectors from the `basis_1` coordinate system to the `basis_2` coordinate system.

    torch.Tensor
        A tensor of shape [..., K, K] containing the squared absolute values of the
        transformation coefficients between the two bases.

        For a 2×2 case, the output can be visualized as:

        ```
        ┌───────────────────────────────────────────┐
        │                                           │
        │     basis_1 states →                      │
        │    ┌─────────────┬─────────────┐          │
        │    │             │             │          │
        │    │  ⟨b2₀|b1₀⟩  | <b2₀|b1₁⟩   │          │
        │ b  │             │             │          │
        │ a  │             │             │          │
        │ s  │  ⟨b2₁|b1₀⟩  | <b2₁|b1₁⟩   │          │
        │ i  │             │             │          │
        │ s  │             │             │          │
        │ _  └─────────────┴─────────────┘          │
        │ 2                                         │
        │                                           │
        │ s                                         │
        │ t                                         │
        │ a                                         │
        │ t                                         │
        │ e                                         │
        │ s                                         │
        │ ↓                                         │
        └───────────────────────────────────────────┘
    """
    return torch.matmul(basis_2.conj().transpose(-1, -2), basis_1)


def get_transformation_coeffs(basis_old: torch.Tensor, basis_new: torch.Tensor):
    """
    Calculate the squared absolute values of transformation coefficients between two bases.

    This function computes the overlap probabilities between states in two different bases.
    The output values represent |⟨basis_2_i|basis_1_j⟩|², which are the squared magnitudes
    of probability amplitudes in quantum mechanics.


    :param basis_old: (b1) torch.Tensor
        The first basis tensor with shape [..., K, K], where K is the spin dimension size.
        Each column basis_1[:,j] represents an eigenvector in the first basis.

    :param basis_new: (b2) torch.Tensor
        The second basis tensor with shape [..., K, K], where K is the spin dimension size.
        Each column basis_2[:,i] represents an eigenvector in the second basis.

    :return: torch.Tensor
        A tensor of shape [..., K, K] containing the squared absolute values of the
        transformation coefficients between the two bases.

        For a 2×2 case, the output can be visualized as:

        ```
        ┌───────────────────────────────────────────┐
        │                                           │
        │     basis_1 states →                      │
        │    ┌─────────────┬─────────────┐          │
        │    │             │             │          │
        │    │ |⟨b2₀|b1₀⟩|²| ⟨b2₀|b1₁⟩|² │          │
        │ b  │             │             │          │
        │ a  │             │             │          │
        │ s  │ |⟨b2₁|b1₀⟩|²| ⟨b2₁|b1₁⟩|² │          │
        │ i  │             │             │          │
        │ s  │             │             │          │
        │ _  └─────────────┴─────────────┘          │
        │ 2                                         │
        │                                           │
        │ s  Element [i,j] represents the probability│
        │ t  of measuring state j in basis_1 if the │
        │ a  system was in state i in basis_2       │
        │ t                                         │
        │ e                                         │
        │ s                                         │
        │ ↓                                         │
        └───────────────────────────────────────────┘
    """

    transforms = basis_transformation(basis_old, basis_new)
    return transforms.abs().square()


def transform_matrix_to_new_basis(initial_rates: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Transform transition rates from matrix form to new basis set.
    K(b_new_1 -> b_new_2) = |⟨b_new_1|b_old_1⟩|² * |⟨b_new_2|b_old_2⟩|² * K(b_old_1 -> b_old_2)

    WARNING: This transformation applies only when initial transition levels (i, j)
    do not transform into identical levels.
    If transitions exist between levels K1 <-> K2 and they transform into identical levels
    (N = a*K1 + b*K2), correlation terms arise between levels that pure relaxation rates
    cannot describe correctly.

    :param initial_rates: Transition rates matrix. Shape [..., K, K]. Diagonal elements must be zero
    :param coeffs: Transformation coefficients (see get_transformation_coeffs). Shape [..., K, K]
    :return: Transformed rate matrix
    """
    return coeffs @ initial_rates @ coeffs.transpose(-1, -2)


def transform_vector_to_new_basis(initial_rates: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Transform a vector from old basis to new basis using transformation coefficients.

    Applies the transformation: v_new[i] = Σ_j |⟨new_i|old_j⟩|² * v_old[j]

    This can be used to transform:
    - Population vectors (state occupancies)
    - Outward transition rates
    - Any other quantities that transform linearly with basis overlap probabilities

    :param initial_rates: Values in the old basis. Shape [..., K]
    :param coeffs: Transformation coefficients |⟨new|old⟩|². Shape [..., K, K]
    :return: Transformed values in the new basis. Shape [..., K]
    """
    return torch.matmul(coeffs, initial_rates)


def transform_diagonal_rates(kinetic_diag_matrix: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Transform diagonal kinetic rates to new basis.

    This is equivalent to transform_vector_to_new_basis but named specifically
    for kinetic rate transformations for clarity.

    :param kinetic_diag_matrix: Diagonal kinetic rates in old basis. Shape [..., K]
    :param coeffs: Transformation coefficients |⟨new|old⟩|². Shape [..., K, K]
    :return: Transformed kinetic rates in new basis. Shape [..., K]
    """
    return torch.matmul(coeffs, kinetic_diag_matrix)


def compute_clebsch_gordan_coeffs(
        target_basis: torch.Tensor,
        basis_list: list[torch.Tensor]
) -> torch.Tensor:
    """
    Compute Clebsch-Gordan-like coefficients for expressing a coupled basis as
    a Kronecker product of uncoupled bases.

    For n uncoupled bases with dimensions [k1, k2, ..., kn], computes coefficients C
    that express each coupled basis state as a linear combination of tensor products:

    |coupled_m⟩ = Σ_{i1,...,in} C[i1, i2, ..., in, m] |uncoupled_1_{i1}⟩ ⊗ ... ⊗ |uncoupled_n_{in}⟩

    where ⊗ denotes the tensor (Kronecker) product.

    Example:
        For two spin-1/2 systems (uncoupled), computing coefficients for the
        total angular momentum eigenstates (coupled basis).

    :param target_basis: Coupled basis vector. Shape: [..., K, K] where K = k1*k2*...*kn
    :param basis_list: List of uncoupled basis tensors.
                             Each has shape [..., k_i, k_i]
    :return: Clebsch-Gordan coefficients. Shape: [..., k1, k2, ..., kn, K]
    """
    dims = [basis.shape[-1] for basis in basis_list]
    kron_basis = basis_list[0]
    for basis in basis_list[1:]:
        kron_basis = torch.einsum('...ij,...kl->...kjil', kron_basis, basis)
        kron_shape = list(kron_basis.shape[:-4]) + [
            kron_basis.shape[-4] * kron_basis.shape[-3],
            kron_basis.shape[-2] * kron_basis.shape[-1]
        ]
        kron_basis = kron_basis.reshape(*kron_shape)
    C_flat = torch.matmul(kron_basis.conj().transpose(-1, -2), target_basis)
    C_reshaped = C_flat.reshape(*C_flat.shape[:-2], *dims, C_flat.shape[-1])
    return C_reshaped


def compute_clebsch_gordan_probabilities(
        target_basis: torch.Tensor,
        basis_list: list[torch.Tensor]
) -> torch.Tensor:
    """
    Compute squared absolute values of Clebsch-Gordan coefficients.

    These represent the probabilities |C[i1, i2, ..., in, m]|² that a coupled
    state m is composed of the tensor product of uncoupled states (i1, i2, ..., in).

    :param target_basis: Coupled basis vectors. Shape: [..., K, K] where K = k1*k2*...*kn
    :param basis_list: List of uncoupled basis tensors.
                                  Each has shape [..., k_i, k_i]
    :return: Squared Clebsch-Gordan coefficients. Shape: [..., k1, k2, ..., kn, K]
    """
    return compute_clebsch_gordan_coeffs(target_basis, basis_list).abs().square()


def transform_kronecker_populations(
        populations_list: list[torch.Tensor],
        coeffs: torch.Tensor,
) -> torch.Tensor:
    """
        Transform populations from uncoupled (product) basis to coupled basis using
        Clebsch-Gordan coefficients.

        Computes the population of each coupled state as:
        n_coupled[m] = Σ_{i1,...,in} |C[i1, ..., in, m]|² * n_uncoupled_1[i1] * ... * n_uncoupled_n[in]

        This implements the quantum mechanical rule that populations of uncoupled subsystems
        multiply, weighted by the squared Clebsch-Gordan coefficients.

        Example:
        :param populations_list: List of population vectors for each uncoupled system.
                                Each has shape [..., k_i]
        :param coeffs: Squared Clebsch-Gordan coefficients |C|².
                          Shape: [..., k1, k2, ..., kn, K]
        :return: Populations in coupled basis. Shape: [..., K]
        """
    n_bases = len(populations_list)
    batch_shape_popul = populations_list[0].shape[:- 1]
    outer_product = populations_list[0]

    for i in range(1, n_bases):
        outer_product = outer_product.unsqueeze(-1)
        scalars_expanded = populations_list[i].reshape(*batch_shape_popul, *(1,) * i, -1)
        outer_product = outer_product * scalars_expanded

    expanded_outer = outer_product.unsqueeze(-1)
    weighted = coeffs * expanded_outer
    sum_dims = tuple(range(-n_bases - 1, -1))
    result = weighted.sum(dim=sum_dims)
    return result


def transform_kronecker_vectors(
        vector_list: list[torch.Tensor],
        coeffs: torch.Tensor,
) -> torch.Tensor:
    """
        Transform populations from uncoupled (product) basis to coupled basis using
        Clebsch-Gordan coefficients.

        Computes the population of each coupled state as:
        R_coupled[m] = Σ_{i1,...,in} |C[i1, ..., in, m]|² * (K1[i1] + K2[i2] + ... + Kn[in])

        This implements the quantum mechanical rule that populations of uncoupled subsystems
        multiply, weighted by the squared Clebsch-Gordan coefficients.

        Example:
        :param coeffs: Squared Clebsch-Gordan coefficients |C|².
                          Shape: [..., k1, k2, ..., kn, K]
        :param vector_list: List of population vectors for each uncoupled system.
                                Each has shape [..., k_i]
        :return: Populations in coupled basis. Shape: [..., K]
        """
    n_bases = len(vector_list)
    batch_shape = vector_list[0].shape[:- 1]
    k_dims = coeffs.shape[-n_bases-1:-1]

    outer_product = torch.zeros(*batch_shape, *k_dims, device=coeffs.device, dtype=coeffs.dtype)
    for i in range(0, n_bases):
        scalars = vector_list[i]
        view_shape = [1] * (len(batch_shape) + n_bases)
        view_shape[len(batch_shape) + i] = scalars.shape[-1]
        vec_expanded = scalars.reshape(*batch_shape, *view_shape[len(batch_shape):])
        outer_product = outer_product + vec_expanded

    weighted = coeffs * outer_product.unsqueeze(-1)
    sum_dims = tuple(-(i + 2) for i in range(n_bases))
    out = weighted.sum(dim=sum_dims)
    return out


def transform_kronecker_matrix(
        matrices: list[torch.Tensor],
        coeffs: torch.Tensor,
) -> torch.Tensor:
    """
    Transform rate matrices from uncoupled (product) basis to coupled basis using
    Clebsch-Gordan coefficients.

    For rate matrices R1, R2, ..., Rn in uncoupled bases, computes the rate matrix
    in the coupled basis by:
    1. Forming the Kronecker product  (R1 ⊗ I2 ⊗ ... ⊗ In + I1 ⊗ R2 ⊗ ... ⊗ In + ... + I1 ⊗ ... ⊗ Rn)
    2. Transforming via: R_coupled = U† R U
    where U is built from Clebsch-Gordan coefficients.

    For two matrices it will be:
    W_coupled[m, n] = Σ_{i1,...,in}{j1,...,jn} |C[i1, ..., in, m]|² |C[j1, ..., jn, n]|² * (W[i1, j1] + W[i2, j2] + ...)

    This preserves the tensor product structure of independent relaxation processes
    while expressing them in the coupled basis.

    WARNING: This transformation applies only when initial transition levels (i, j)
    do not transform into identical levels.
    If transitions exist between levels K1 <-> K2 and they transform into identical levels
    (N = a*K1 + b*K2), correlation terms arise between levels that pure relaxation rates
    cannot describe correctly.

    :param matrices: List of rate matrices for each uncoupled system.
                          Each has shape [..., k_i, k_i]
    :param coeffs: Squared Clebsch-Gordan coefficients |C|².
                      Shape: [..., k1, k2, ..., kn, K]
    :return: Rate matrix in coupled basis. Shape: [..., K, K]
    """
    n = len(matrices)
    batch_shape = matrices[0].shape[:-2]
    k_dims = coeffs.shape[-n - 1:-1]
    K = coeffs.shape[-1]
    total_dim = int(torch.prod(torch.tensor(k_dims)))
    U = coeffs.reshape(*coeffs.shape[:-n-1], total_dim, K)

    expanded_shape = list(batch_shape) + list(k_dims) + list(k_dims)
    R_expanded = torch.zeros(expanded_shape, device=matrices[0].device, dtype=matrices[0].dtype)

    for i, mat in enumerate(matrices):
        view_shape = [1] * (len(batch_shape) + 2 * n)
        view_shape[:len(batch_shape)] = list(batch_shape)
        view_shape[len(batch_shape) + i] = k_dims[i]
        view_shape[len(batch_shape) + n + i] = k_dims[i]

        mat_expanded = mat.reshape(view_shape)
        R_expanded = R_expanded + mat_expanded

    R_uncoupled = R_expanded.reshape(*batch_shape, total_dim, total_dim)
    return U.conj().transpose(-1, -2) @ R_uncoupled @ U


def batched_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batched Kronecker product.

    Computes the Kronecker product for batched matrices.

    :param a: torch.Tensor of shape [..., M, N]
    :param b: torch.Tensor of shape [..., P, Q]
    :return: torch.Tensor of shape [..., M*P, N*Q]
    """
    *batch_dims, M, N = a.shape
    *_, P, Q = b.shape

    a_expanded = a[..., :, None, :, None]
    b_expanded = b[..., None, :, None, :]

    result = a_expanded * b_expanded
    return result.reshape(*batch_dims, M * P, N * Q)


def compute_liouville_basis_transformation(basis_old: torch.Tensor, basis_new: torch.Tensor):
    """
    Compute the transformation matrix for superoperators between two quantum bases.

    This function calculates the unitary transformation matrix that converts Liouville-space
    superoperators (e.g., relaxation matrices) from an old basis to a new basis. The transformation
    preserves the structure of quantum operations under basis change.


    :param basis_old : torch.Tensor
        Original basis vectors. Shape: [..., K, K] where K is the Hilbert space dimension.
        Columns represent eigenvectors of the old basis.

    :param basis_new : torch.Tensor
        Target basis vectors. Shape: [..., K, K] where K is the Hilbert space dimension.
        Columns represent eigenvectors of the new basis.

    :return: torch.Tensor
        Transformation matrix for Liouville space operators. Shape: [..., K², K²]

        For a 2×2 system (K=2), the output structure can be visualized as:
        ```
        ┌───────────────────────────────────────────────────────┐
        │                                                       │
        │  Old Liouville basis states →                         │
        │  ┌─────────────┬─────────────┬─────────────┬─────────┐ │
        │  │⟨b₂₀b₂₀|b₁₀b₁₀⟩ ...       │⟨b₂₀b₂₀|b₁₁b₁₁⟩ ...      │ │
        │  │ ...         │ ...         │ ...         │ ...     │ │
        │L │⟨b₂₀b₂₁|b₁₀b₁₀⟩ ...       │⟨b₂₀b₂₁|b₁₁b₁₁⟩ ...      │ │
        │i │ ...         │ ...         │ ...         │ ...     │ │
        │o │⟨b₂₁b₂₀|b₁₀b₁₀⟩ ...       │⟨b₂₁b₂₀|b₁₁b₁₁⟩ ...      │ │
        │u │ ...         │ ...         │ ...         │ ...     │ │
        │v │⟨b₂₁b₂₁|b₁₀b₁₀⟩ ...       │⟨b₂₁b₂₁|b₁₁b₁₁⟩ ...      │ │
        │i └─────────────┴─────────────┴─────────────┴─────────┘ │
        │l                                                         │
        │l                                                         │
        │e                                                         │
        │  New Liouville basis states ↓                           │
        └─────────────────────────────────────────────────────────┘
        ```

    The transformation follows:
        R_new = T_switch @ R_old @ T_switch.conj().transpose(-1, -2)
    where T_switch = kron(basis_new.conj(), basis_new) @ kron(basis_old.conj(), basis_old).H
    """
    U = basis_new.conj().transpose(-1, -2) @ basis_old
    T_switch = batched_kron(U, U.conj())
    return T_switch


def compute_density_basis_transformation(basis_old: torch.Tensor, basis_new: torch.Tensor):
    """
    Compute the transformation matrix for density matrices between two quantum bases.

    This function calculates the unitary matrix that converts density matrices from an old basis
    to a new basis. The transformation preserves quantum state properties under basis change.

    :param basis_old : torch.Tensor
        Original basis vectors. Shape: [..., K, K] where K is the Hilbert space dimension.
        Columns represent eigenvectors of the old basis.

    :param basis_new : torch.Tensor
        Target basis vectors. Shape: [..., K, K] where K is the Hilbert space dimension.
        Columns represent eigenvectors of the new basis.

    :return:
    torch.Tensor
        Transformation matrix for density matrices. Shape: [..., K, K]

        Element [i,j] represents the complex amplitude ⟨new_i|old_j⟩.
        For visualization, see the diagram in `basis_transformation` docstring.

    Notes:
    ------
    This is equivalent to the standard basis transformation matrix U where:
        ρ_new = U @ ρ_old @ U†
    with U = basis_new.conj().transpose(-1, -2) @ basis_old
    """
    return basis_new.conj().transpose(-1, -2) @ basis_old


def transform_density(density_old: torch.Tensor, coeffs: torch.Tensor):
    """
    Transform a density matrix to a new quantum basis.

    Applies a unitary transformation to a density matrix using precomputed transformation coefficients.
    Preserves Hermiticity, trace, and positivity of the density operator.

    :param density_old : torch.Tensor
        Density matrix in original basis. Shape: [..., K, K]
        Must be Hermitian with trace 1 for physical states.

    :param coeffs : torch.Tensor
        Precomputed basis transformation matrix. Shape: [..., K, K]
        Typically from `compute_density_basis_transformation`.

    :return:
    torch.Tensor
        Density matrix in new basis. Shape: [..., K, K]

    Mathematical Formulation:
    -------------------------
    ρ_new = U @ ρ_old @ U†
    where U = transformation_matrix

    Notes:
    ------
    - Diagonal elements represent populations in the new basis
    - Off-diagonal elements represent decoherences in the new basis
    """
    return coeffs @ density_old @ coeffs.conj().transpose(-1, -2)


def transform_liouville_superop(
    superoperator_old: torch.Tensor,
    liouville_transformation: torch.Tensor
) -> torch.Tensor:
    """
    Transform a superoperator to a new quantum basis in Liouville space.

    Applies a basis transformation to Liouville-space operators (e.g., relaxation matrices,
    quantum maps) using precomputed Liouville transformation coefficients.

    :param superoperator_old : torch.Tensor
        Superoperator in original Liouville basis. Shape: [..., K², K²]

    :param liouville_transformation : torch.Tensor
        Precomputed Liouville-space transformation matrix. Shape: [..., K², K²]

    :return: torch.Tensor
        Superoperator in new Liouville basis. Shape: [..., K², K²]

    Mathematical Formulation:
    -------------------------
    R_new = T @ R_old @ T†
    where T = liouville_transformation
    """
    return liouville_transformation @ superoperator_old @ liouville_transformation.conj().transpose(-1, -2)


def transform_liouville_superop_diag(
    superoperator_diag: torch.Tensor,
    liouville_transformation: torch.Tensor
) -> torch.Tensor:
    """
    Transform a diagonal superoperator to a new quantum basis in Liouville space.

    :param superoperator_diag : torch.Tensor
        Diagonal of the superoperator in the original Liouville basis.
        Shape: [..., K²]

    :param liouville_transformation : torch.Tensor
        Precomputed Liouville-space transformation matrix.
        Shape: [..., K², K²]

    :return: torch.Tensor
        Transformed superoperator in the new Liouville basis.
        Shape: [..., K², K²]
    """
    return (liouville_transformation * superoperator_diag.unsqueeze(-2)) \
        @ liouville_transformation.conj().transpose(-1, -2)


def get_coupled_unitary(
        coeffs: torch.Tensor,
        n: int,
) -> torch.Tensor:
    """
    Compute unitary transformation matrix from uncoupled to coupled basis.

    This is equivalent to reshaping the output of compute_clebsch_gordan_coeffs
    into a matrix form [..., K, K] where K = k1*k2*...*kn.

    :param coeffs: Clebsch-Gordan coefficients. Shape: [..., k1, k2, ..., kn, K]
    :param n: number of subsystems
    :return: Unitary transformation matrix U. Shape: [..., K, K]
    """
    batch_shape = coeffs.shape[:-n - 1]
    uncoupled_dims = coeffs.shape[-n - 1:-1]
    total_dim = int(torch.prod(torch.tensor(uncoupled_dims)))
    K = coeffs.shape[-1]
    return coeffs.reshape(*batch_shape, total_dim, K)


def transform_kronecker_density(
        density_list: list[torch.Tensor],
        coeffs: torch.Tensor,
) -> torch.Tensor:
    """
    Transform density matrices from uncoupled to coupled basis.

    ρ_coupled = U @ (ρ₁ ⊗ ρ₂ ⊗ ... ⊗ ρₙ) @ Uᴴ

    :param density_list: Density matrices for subsystems. Each shape: [..., k_i, k_i]
    :param coeffs: Clebsch-Gordan coefficients. Shape: [..., k1, k2, ..., kn, K]
    :return: Density matrix in coupled basis. Shape: [..., K, K]
    """
    unitarty = get_coupled_unitary(coeffs, len(density_list))

    rho_uncoupled = density_list[0]
    for rho in density_list[1:]:
        rho_uncoupled = torch.einsum('...ij,...kl->...ikjl', rho_uncoupled, rho)
        rho_uncoupled = rho_uncoupled.reshape(
            *rho_uncoupled.shape[:-4],
            rho_uncoupled.shape[-4] * rho_uncoupled.shape[-3],
            rho_uncoupled.shape[-2] * rho_uncoupled.shape[-1]
        )
    return unitarty @ rho_uncoupled @ unitarty.conj().transpose(-1, -2)


def transform_kronecker_superoperator(
        superoperator_list: list[torch.Tensor],
        coeffs: torch.Tensor,
) -> torch.Tensor:
    """
    Transform superoperators from uncoupled to coupled Liouville basis.

    R_coupled = T ⊗ (R1 ⊗ I2 ⊗ ... ⊗ In + I1 ⊗ R2 ⊗ ... ⊗ In + ... + I1 ⊗ ... ⊗ Rn) ⊗ Tᴴ
    where T = U ⊗ U* (Liouville transformation)

    :param superoperator_list: Superoperators for subsystems. Each shape: [..., k_i², k_i²]
    :param coeffs: Clebsch-Gordan coefficients. Shape: [..., k1, k2, ..., kn, K]
    :return: Superoperator in coupled Liouville basis. Shape: [..., K², K²]
    """
    n = len(superoperator_list)
    batch_shape = superoperator_list[0].shape[:-2]
    k_dims = coeffs.shape[-n - 1:-1]
    total_dim = int(torch.prod(torch.tensor(k_dims)))

    unitarty = get_coupled_unitary(coeffs, len(superoperator_list))  # K, K
    T = torch.einsum('...ij,...kl->...ikjl', unitarty, unitarty.conj())
    T = T.reshape(
        *T.shape[:-4],
        T.shape[-4] * T.shape[-3],
        T.shape[-2] * T.shape[-1]
    )  # K^2, K^2

    expanded_shape = list(batch_shape) + [ki ** 2 for ki in k_dims] + [ki ** 2 for ki in k_dims]
    R_expanded = torch.zeros(expanded_shape, device=superoperator_list[0].device, dtype=superoperator_list[0].dtype)

    for i, superop in enumerate(superoperator_list):
        ki_sq = k_dims[i] ** 2
        view_shape = [1] * (len(batch_shape) + 2 * n)
        view_shape[:len(batch_shape)] = list(batch_shape)
        view_shape[len(batch_shape) + i] = ki_sq
        view_shape[len(batch_shape) + n + i] = ki_sq

        superop_expanded = superop.reshape(view_shape)
        R_expanded = R_expanded + superop_expanded

    R_uncoupled = R_expanded.reshape(*batch_shape, total_dim ** 2, total_dim ** 2).to(T.dtype)
    return T @ R_uncoupled @ T.conj().transpose(-1, -2)


class Liouvilleator:
    @staticmethod
    def commutator_superop(operator: torch.Tensor) -> torch.Tensor:
        """
        Compute the superoperator form of the commutator with a given operator.
        For an operator A, this superoperator L satisfies:
            L[ρ] = [A, ρ] = Aρ - ρA
        when applied to a vectorized density matrix.

        :param operator : torch.Tensor
            Operator for the commutator. Shape: [..., d, d]

        :return: torch.Tensor
            Commutator superoperator. Shape: [..., d², d²]

        Mathematical Formulation:
        -------------------------
        L = A ⊗ I - I ⊗ Aᵀ
        where ⊗ denotes the Kronecker product, and I is the identity matrix.

        Notes:
        ------
        - Vectorization follows row-major (C) order: element ρ_ij is at position i*d + j
        - Preserves Hermiticity of density matrices when used in Liouvillian evolution
        """
        d = operator.shape[-1]
        I = torch.eye(d, dtype=operator.dtype, device=operator.device)
        batch_dims = operator.shape[:-2]

        term1 = torch.einsum('...ij,kl->...ikjl', operator, I).reshape(*batch_dims, d * d, d * d)
        term2 = torch.einsum('kl,...ij->...kilj', I, operator.transpose(-1, -2)).reshape(*batch_dims, d * d, d * d)
        return term1 - term2

    @staticmethod
    def vec(rho: torch.Tensor) -> torch.Tensor:
        """
        Transform density matrix to Liouvillian space from Hilbert Space
        Example:
            rho = torch.tensor([[0.5, 0.1],
                                [0.1, 0.5]])
            return tensor([0.5, 0.1, 0.1, 0.5])
        :param rho: density matrix in matrix form. The shape is [..., N, N], where N is number of levels
        :return density matrix in vector form. The shape is [..., N**2], where N is number of levels:
        """
        shapes = rho.shape
        return rho.reshape(*shapes[:-2], shapes[-1] * shapes[-1])

    @staticmethod
    def unvec(rho: torch.Tensor) -> torch.Tensor:
        """
        Transform density matrix to Hilbert space from Liouvillian Space
        Example:
            vec_rho = torch.tensor([0.6, 0.0, 0.0, 0.4])
            return tensor([[0.6, 0.0],
                           [0.0, 0.4]])
        :param rho: density matrix in vector form. The shape is [..., N**2], where N is number of levels
        :return density matrix in matrix form. The shape is [..., N, N], where N is number of levels:
        """
        shapes = rho.shape
        dim = int(math.sqrt(shapes[-1]))
        return rho.reshape(*shapes[:-1], dim, dim)

    @staticmethod
    def hamiltonian_superop(hamiltonian: torch.Tensor) -> torch.Tensor:
        """
        Compute the Liouvillian superoperator for unitary evolution under a Hamiltonian.

        For a Hamiltonian H, this superoperator generates:
            dρ/dt = -i[H, ρ]

        :param hamiltonian : torch.Tensor
            Hamiltonian operator. Shape: [..., d, d]

        :return: torch.Tensor
            Hamiltonian superoperator. Shape: [..., d², d²]

        Mathematical Formulation:
        -------------------------
        L = -i (H ⊗ I - I ⊗ Hᵀ)

        Notes:
        ------
        - This is the unitary part of the Lindblad master equation
        - Always generates trace-preserving and positivity-preserving dynamics
        """
        return -1j * Liouvilleator.commutator_superop(hamiltonian)

    @staticmethod
    def anticommutator_superop(operator: torch.Tensor) -> torch.Tensor:
        """
        Compute the superoperator form of the anticommutator with a given operator.

        For an operator A, this superoperator L satisfies:
            L[ρ] = {A, ρ} = Aρ + ρA
        when applied to a vectorized density matrix.

        :param operator: torch.Tensor
            Operator for the anticommutator. Shape: [..., d, d]

        :return: torch.Tensor
            Anticommutator superoperator. Shape: [..., d², d²]

        Mathematical Formulation:
        -------------------------
        L = A ⊗ I + I ⊗ Aᵀ

        Notes:
        ------
        - Does not preserve trace by itself (requires combination with other terms)
        """
        d = operator.shape[-1]
        I = torch.eye(d, dtype=operator.dtype, device=operator.device)
        batch_dims = operator.shape[:-2]

        term1 = torch.einsum('...ij,kl->...ikjl', operator, I).reshape(*batch_dims, d * d, d * d)
        term2 = torch.einsum('kl,...ij->...kilj', I, operator.transpose(-1, -2)).reshape(*batch_dims, d * d, d * d)
        return term1 + term2

    @staticmethod
    def anticommutator_superop_diagonal(operator: torch.Tensor) -> torch.Tensor:
        """
        Compute the superoperator form of the anticommutator with a given DIAGONAL of a operator.
        It is similar anticommutator_superop but for the special case when the operator is diagonal.
        It takes only it's diagonal and returns also diagonal

        For an operator A, this superoperator L satisfies:
            L[ρ] = {A, ρ} = Aρ + ρA
        when applied to a vectorized density matrix.

        :param operator: torch.Tensor
            Operator for the anticommutator. Shape: [..., d,]

        :return: torch.Tensor
            Anticommutator superoperator. Shape: [..., d²]

        Mathematical Formulation:
        -------------------------
        L = A ⊗ I + I ⊗ Aᵀ
        """
        d = operator.shape[-1]
        batch_dims = operator.shape[:-1]
        i_indices = torch.arange(d, device=operator.device)
        j_indices = torch.arange(d, device=operator.device)
        i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')
        anticomm_diagonal = operator[..., i_grid] + operator[..., j_grid]
        return anticomm_diagonal.reshape(*batch_dims, d * d)

    @staticmethod
    def decay_superop(jump_operator: torch.Tensor, rate: float) -> torch.Tensor:
        """
        Compute the dissipative superoperator for quantum decay processes.

        :param  jump_operator : torch.Tensor
            Quantum jump operator. Shape: [..., d, d]
        rate : float
            Positive decay rate (Γ > 0)

        :return: torch.Tensor
            Dissipative superoperator. Shape: [..., d², d²]

        Mathematical Formulation:
        -------------------------
        L_decay = -(Γ/2) (L†L ⊗ I + I ⊗ (L†L)ᵀ)
        where L = jump_operator

        Notes:
        ------
        - This is NOT a complete Lindblad dissipator (missing +LρL† term)
        - Always negative semi-definite for population decay
        """
        decay_op = jump_operator.conj().transpose(-1, -2) @ jump_operator
        return -rate / 2 * Liouvilleator.anticommutator_superop(decay_op)

    @staticmethod
    def lindblad_dissipator_superop(w: torch.Tensor) -> torch.Tensor:
        """
        Construct Lindblad dissipator superoperator from off-diagonal rates.

        Models the dissipator term in the Lindblad equation:
            D(ρ) = Σ_{i≠j} w_{ij} [L_{ij} ρ L_{ij}^† - (1/2){L_{ij}^† L_{ij}, ρ}]
        where L_{ij} = √w_{ij} |j⟩⟨i|

        This simplifies to:
            D(ρ) = Σ_{i≠j} w_{ij} [|j⟩⟨i| ρ |i⟩⟨j| - (1/2){|i⟩⟨i|, ρ}]

        :param w : torch.Tensor
            Off-diagonal rate matrix. Shape: [..., n, n]
            Element [i,j] represents transition rate for i≠j

        :return: torch.Tensor
            Lindblad dissipator superoperator. Shape: [..., n², n²]

        Mathematical Formulation:
        -------------------------
        The dissipator consists of two parts:
        1. Jump term: Σ_{i≠j} w_{ij} |j⟩⟨i| ρ |i⟩⟨j|
        2. Decay term: -(1/2) Σ_{i≠j} w_{ij} {|i⟩⟨i|, ρ}

        Notes:
        ------
        - Only off-diagonal elements of w are used (i≠j)
        - The decay term represents decay
        - The jump term represents population transfer
        """
        n = w.shape[-1]
        batch_shape = w.shape[:-2]

        superop_jump = torch.zeros(*batch_shape, n * n, n * n, dtype=w.dtype, device=w.device)

        i_indices = torch.arange(n, device=w.device)
        j_indices = torch.arange(n, device=w.device)
        i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')

        offdiag_mask = i_grid != j_grid
        i_offdiag = i_grid[offdiag_mask]
        j_offdiag = j_grid[offdiag_mask]

        row_idx = j_offdiag * n + j_offdiag
        col_idx = i_offdiag * n + i_offdiag

        superop_jump[..., row_idx, col_idx] = w[..., i_offdiag, j_offdiag]

        w_offdiag = w * (~torch.eye(n, dtype=torch.bool, device=w.device))
        decay_rates = w_offdiag.sum(dim=-1)

        superop_decay = -0.5 * Liouvilleator.anticommutator_superop_diagonal(decay_rates)

        superop_total = superop_jump + torch.diag_embed(superop_decay)
        return superop_total

    @staticmethod
    def lindblad_decoherences_superop(gamma: torch.Tensor) -> torch.Tensor:
        """
        Construct Lindblad decoherences superoperator from on-diagonal decoherence. It models decoherences

        Models the decoherences term in the Lindblad equation:
            D(ρ) = Σ_i γ_{i} [L_{i} ρ L_{i}^† - (1/2){L_{i}^† L_{i}, ρ}]
        where L_{i} = √γ_{i} |i⟩⟨i|

        This simplifies to:
            D(ρ) = Σ_i γ_{i} [|i⟩⟨i| ρ |i⟩⟨i| - (1/2){|i⟩⟨i|, ρ}]

        :param gamma : torch.Tensor
            decoherence rate matrix. Shape: [..., n]
            Element [i] represents decoherence rate.
            For example, if γ is not zero only for i state, then the result will be - γ / 2 * rho_ij for all j != i
            In the general case:

            drho_ij / dt = - (gamma_i + gamma_j) / 2 * rho_ij for i != j

        :return: torch.Tensor
            Lindblad decoherences superoperator. Shape: [..., n², n²]
        """
        decoherences = -(gamma[..., :, None] + gamma[..., None, :]) / 2
        decoherences.diagonal(dim1=-2, dim2=-1).zero_()

        *batch, n, _ = decoherences.shape
        N = n * n
        pop_indices = torch.arange(n, device=decoherences.device) * (n + 1)

        is_coherence = torch.ones(N, dtype=torch.bool, device=decoherences.device)
        is_coherence[pop_indices] = False

        rate_vector = decoherences.reshape(*batch, N)
        rate_vector = rate_vector.clone()
        rate_vector[..., pop_indices] = 0
        return torch.diag_embed(rate_vector)