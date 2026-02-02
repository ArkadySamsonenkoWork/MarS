from __future__ import annotations

from abc import ABC, abstractmethod
import math
import warnings
import itertools
import functools

import torch
from torch import nn

import typing as tp

from .. import spin_model
from . import transform


def transform_to_complex(vector):
    if vector.dtype == torch.float32:
        return vector.to(torch.complex64)
    elif vector.dtype == torch.float64:
        return vector.to(torch.complex128)
    else:
        return vector


def _normalize_to_components(
        contexts: tp.List[tp.Union[Context, SummedContext]]
) -> tp.Tuple[tp.List[tp.List[Context]], tp.List[int], tp.List[int], tp.List[int], torch.device, torch.dtype]:
    """
    Normalize input contexts into uniform component lists with metadata.

    Converts all inputs to lists of atomic Context objects while preserving:
    - Component structure (flattening SummedContexts)
    - System properties (dimension, )

    :param contexts: List of Context or SummedContext objects to normalize
    :return:
        component_lists: List of component lists for each input context
        num_with_population: Count of population-equipped components per input context
        num_with_density: Count of density-equipped components per input context
        spin_system_dimension: Count of size of spin system
        device:
        dtype:
    """
    dtype, device = None, None
    for ctx in contexts:
        ref_ctx = ctx
        dtype, device = ref_ctx.dtype, ref_ctx.device
        if dtype is not None:
            break

    for ctx in contexts[1:]:
        if ctx.dtype is not None:
            if ctx.dtype != dtype or ctx.device != device:
                raise ValueError("All contexts must share the same dtype and device")

    component_lists = []
    num_with_population = []
    num_with_density = []
    spin_system_dim = []

    for ctx in contexts:
        if isinstance(ctx, Context):
            components = [ctx]
            n_population = 1 if ctx.contexted_init_population is not None else 0
            n_density = 1 if ctx.contexted_init_density is not None else 0
            spin_system_dim.append(ctx.spin_system_dim)
        elif isinstance(ctx, SummedContext):
            components = list(ctx.component_contexts)
            n_population = ctx.num_contexts_with_populations
            n_density = ctx.num_contexts_with_density
            spin_system_dim.append(ctx.spin_system_dim)
        else:
            raise TypeError(f"Unsupported context type: {type(ctx)}")

        if not components:
            continue

        component_lists.append(components)
        num_with_population.append(n_population)
        num_with_density.append(n_density)

    return component_lists, num_with_population, num_with_density, spin_system_dim, device, dtype


def _create_zero_context(
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        init_populations: tp.Optional[torch.Tensor],
        init_density: tp.Optional[torch.Tensor],
        has_basis: bool
) -> Context:
    """
    Create a zero-initialized context with specified properties.

    :param dim: Dimension of the spin system
    :param device: Device to place tensors on
    :param dtype: Floating-point dtype for real components
    :param init_populations: Initial population of a context
    :param init_density: Initial density of a context
    :param has_basis: Whether to include an identity basis matrix
    :return: Minimal context object with zero-initialized properties
    """
    if has_basis:
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        basis = torch.eye(dim, device=device, dtype=complex_dtype)
    else:
        basis = None

    context = Context(
        basis=basis,
        init_populations=init_populations,
        init_density=init_density,
        device=device,
        dtype=dtype
    )
    context.spin_system_dim = dim
    return context


def _group_components_by_position(
        component_lists: tp.List[tp.List[Context]],
        num_with_population: tp.List[int],
        num_with_density: tp.List[int],
        spin_system_dim: tp.List[int],
        device: torch.device,
        dtype: torch.dtype,
) -> tp.List[tp.Sequence[Context]]:
    """
    Group components by position with intelligent zero-padding for Kronecker multiplication.

    This function handles the multiplication of SummedContext objects by distributing
    the Kronecker product over addition.

    Mathematical Background:
    ------------------------
    When multiplying summed contexts, we apply distributivity:
        (K₁ + K₂) ⊗ I + I ⊗ K₃ = (K₁ ⊗ I + I ⊗ K₃) + (K₂ ⊗ I + I ⊗ 0)
        (n₁ + n₂) ⊗ n₃ = n₁ ⊗ n₃ + n₂ ⊗ n₃

    Population Distribution in SummedContext:
    -----------------------------------------
    In SummedContext, components are sorted so that:
    1. All components WITH populations come first (positions 0 to num_with_population-1)
    2. All components WITHOUT populations come after


    Example 1 - Equal length, all populated:
        component_lists = [
            [ctx_A1(pop=0.5), ctx_A2(pop=0.3), ctx_A3(pop=None)],  # 2 populated at start
            [ctx_B1(pop=0.4), ctx_B2(pop=0.2), ctx_B3(pop=None)]   # 2 populated at start
        ]
        num_with_population = [2, 2]

        Returns: [
            [ctx_A1(pop=0.5), ctx_B1(pop=0.4)],  # Position 0: both populated
            [ctx_A2(pop=0.3), ctx_B2(pop=0.2)],  # Position 1: both populated
            [ctx_A1(pop=0.5, K=None), ctx_B1(pop=0.2, K=None)],  # Position 2:
            [ctx_A1(pop=0.3, K=None), ctx_B1(pop=0.4, K=None)],  # Position 3:
            [ctx_A3(pop=None), ctx_B3(pop=None)]  # Position 4: both unpopulated
        ]

    Example 2 - Unequal lengths, different population counts:
        component_lists = [
            [ctx_A1(pop=0.5), ctx_A2(pop=0.3), ctx_A3(pop=None), ctx_A4(pop=None)],
            [ctx_B1(pop=0.4)]
        ]
        num_with_population = [2, 1]
        spin_system_dim = [3, 3]

        Padding analysis:
        - Position 0: A1 (populated), B1 (populated) → no padding needed
        - Position 1: A2 (populated), B missing → pad B with ZERO POPULATION
        - Position 2: A3 (unpopulated), B missing → pad B with None
        - Position 3: A4 (unpopulated), B missing → pad B with None

        Returns: [
            [ctx_A1(pop=0.5), ctx_B1(pop=0.4)],           # Both populated
            [ctx_A2(pop=0.3), zero_B(pop=0.4, K=None)],      # A2 populated
            [ctx_A3(pop=None), zero_B(pop=None)],         # Both unpopulated
            [ctx_A4(pop=None), zero_B(pop=None)]          # Both unpopulated
        ]

    :param component_lists: Normalized component lists per context (one list per input context).
                           Components are already sorted: populated first, unpopulated after.
    :param num_with_population: Number of components with initial populations per input context.
                                Defines the boundary: positions [0, n_pop) have populations.
    :param num_with_density: Number of components with initial density matrices per input context.
    :param spin_system_dim: Spin system dimensions for each input context.
    :param device: Torch device for tensor allocation.
    :param dtype: Torch dtype for floating-point tensors.
    :return: List of context groups, where each group contains one component from each
             input context at the same position. Zero-padding preserves position semantics.
    """
    if not component_lists:
        return []

    n_contexts = len(component_lists)

    populated_indices = [list(range(n_pop)) for n_pop in num_with_population]
    population_combinations = list(itertools.product(*populated_indices))
    max_unpopulated = max(
        len(components) - n_pop
        for components, n_pop in zip(component_lists, num_with_population)
    )

    result_groups = []
    for combo_indices in population_combinations:
        group = []
        for ctx_idx, pop_idx in enumerate(combo_indices):
            ctx = component_lists[ctx_idx][pop_idx]
            group.append(ctx)
        result_groups.append(group)

    for unpop_offset in range(max_unpopulated):
        group = []
        for ctx_idx, (components, n_pop) in enumerate(zip(component_lists, num_with_population)):
            unpop_idx = n_pop + unpop_offset

            if unpop_idx < len(components):
                ctx = components[unpop_idx]
                group.append(ctx)
            else:
                has_basis = any(c.basis is not None for c in components)
                zero_ctx = _create_zero_context(
                    dim=spin_system_dim[ctx_idx],
                    device=device,
                    dtype=dtype,
                    init_populations=None,
                    init_density=None,
                    has_basis=has_basis
                )
                group.append(zero_ctx)

        result_groups.append(group)

    return result_groups


def multiply_contexts(contexts: tp.List[tp.Union[Context, SummedContext, KroneckerContext]]) ->\
        tp.Union[KroneckerContext, SummedContext]:
    """
    This function models a system consisting of multiple interacting or non-interacting
    subsystems (e.g., electron-nuclear spin systems). Each subsystem
    is described by its own context, and the composite system follows quantum mechanical
    tensor product rules.

    Key physical principles:
    1. State space: The Hilbert space of the composite system is the tensor product of
       subsystem Hilbert spaces: H = H₁ ⊗ H₂ ⊗ ... ⊗ Hₙ
    2. Initial state: The initial density matrix is the tensor product of subsystem states:
       ρ = ρ₁ ⊗ ρ₂ ⊗ ... ⊗ ρₙ
    3. Dynamics: Each subsystem evolves according to its own Hamiltonian and relaxation
       operators, which are embedded into the composite space using tensor products with
       identity operators: K = K1 ⊗ I + I ⊗ K2

    Transformation rules:
    - Basis transformations use Clebsch-Gordan coefficients to map between product bases
      and coupled bases
    - Initial populations are transformed using tensor products of transformation matrices
    - Relaxation superoperators are transformed using Kronecker products of basis
      transformation matrices

    Composite contexts can be created using the @ operator:
        composite_context = context1 @ context2 @ context3

    Warning!
    -Composite context only determined for not-None basis (eigen).
      If it is not Eigen, please considet other types of basis
    - Callables (time-dependent parameters) are not supported

    :return:
        A single Context object or the summ of Contexts
    """
    component_lists, num_with_population, num_with_density, spin_system_dim, device, dtype =\
        _normalize_to_components(contexts)
    padded_component_lists = _group_components_by_position(
        component_lists, num_with_population, num_with_density, spin_system_dim, device, dtype
    )
    results = [_multiply_homogeneous_contexts(group) for group in padded_component_lists]
    return results[0] if len(results) == 1 else SummedContext(results)


def _multiply_homogeneous_contexts_to_context(contexts: tp.Sequence[Context]) -> Context:
    """ Build the tensor-product (Kronecker) composition of multiple homogeneous `Contexts.

    This Function is not used for the manual code!!. It doesn't work correct for superoperators transformation
    if free_probs or drive_probs are set

    This function returns a new Context describing the composite system that is the
    tensor product of the provided `contexts`. Behavior / rules follow the same
    conventions used in your concatenation function but adapted to tensor-product:

    - Hilbert space dimension is the product of individual system dimensions.

    - init_populations are combined with Kronecker product.
      If ALL values are None -> resulting parameter is None. If ANY is None,
      The output is ZERO tensor of the corresponding dimension.

    -`out_probs` / `dephasing`: combined as sum of **diagonal** local operators
        `sum_i I ⊗ ... ⊗ diag(v_i) ⊗ ... ⊗ I` (i.e. vector x I + I x vector interpreted as diag).
        If ALL values are None -> resulting parameter is None. If ANY is -None, missing
          vectors are replaced with zero vectors of the corresponding dimension.

    - Square matrices (free_probs, driven_probs) are combined into a sum of local operators:
        K_total = sum_i (I ⊗ ... ⊗ K_i ⊗ ... ⊗ I)
      If ALL values are None -> resulting parameter is None. If ANY is non-None, missing
      matrices are replaced with zero matrices of the corresponding dimension.

    - Initial densities (init_density) combine by Kronecker product (complex dtype).
      If ALL densities are None -> resulting init_density is None. If ANY is non-None,
      missing ones are treated as zero matrices (so the total density may end up zero).

    - Callables (time-dependent parameters) are NOT supported and will raise TypeError.

    - Batch shapes are broadcast; all non-None parameters must be broadcast-compatible.

    - Requires that all contexts share the same device/dtype.

    :param contexts:
        contexts: List of Context objects to concatenate.

    :return:
        New Context object representing the direct-sum system.

    Raises:
        ValueError: If contexts list is empty or contains incompatible dtypes/devices
        TypeError: If any parameter is callable (time-dependent)
    """
    dtype = contexts[0].dtype
    device = contexts[0].device
    time_dimension = contexts[0].time_dimension

    flag = any([context.eigen_basis_flag for context in contexts])
    if flag:
        raise NotImplementedError(
            "Multiplication of contexts does not support contexts that use the 'eigen' basis (or a None basis). "
            "This is because the resulting behavior would be ambiguous or ill-defined."
            "Please specify a defined physical basis such as 'zfs', 'multiplet', 'product',"
            "or another supported option."
        )

    dims = [ctx.spin_system_dim for ctx in contexts]
    total_dim = functools.reduce(lambda x, y: x * y, dims)
    n = len(contexts)

    def _check_callables(values: tp.List, param_name: str):
        """Check if any value is callable and raise TypeError if so."""
        if any(callable(v) for v in values if v is not None):
            raise TypeError(f"Callable {param_name} not supported in tensor product")

    def _get_common_batch_shape(values: tp.List[tp.Optional[torch.Tensor]], is_matrix: bool = False) -> torch.Size:
        """Extract and broadcast batch shapes from non-None values."""
        batch_shapes = []
        for v in values:
            if v is not None:
                shape = v.shape[:-2] if is_matrix else v.shape[:-1]
                batch_shapes.append(shape)
        return torch.Size(torch.broadcast_shapes(*batch_shapes)) if batch_shapes else torch.Size([])

    def _process_init_populations():
        """Combine init_populations via Kronecker product."""
        values = [ctx.init_populations for ctx in contexts]
        _check_callables(values, "init_populations")

        if all(v is None for v in values):
            return None
        if any(v is None for v in values):
            batch_shape = _get_common_batch_shape(values, is_matrix=False)
            return torch.zeros(batch_shape + (total_dim,), dtype=dtype, device=device)

        batch_shape = _get_common_batch_shape(values, is_matrix=False)
        operators = []
        for i, val in enumerate(values):
            expanded = val.expand(batch_shape + (dims[i],))
            operators.append(expanded)
        return transform.batched_multi_kron([v.unsqueeze(-1) for v in operators]).squeeze(-1)

    def _process_diagonal_vectors(attr_name: str):
        """Combine diagonal vectors (out_probs, dephasing) using batched_sum_kron_diagonal."""
        values = [getattr(ctx, attr_name) for ctx in contexts]
        _check_callables(values, attr_name)

        if all(v is None for v in values):
            return None

        batch_shape = _get_common_batch_shape(values, is_matrix=False)
        operators = []
        for i, val in enumerate(values):
            if val is not None:
                expanded = val.expand(batch_shape + (dims[i],))
            else:
                expanded = torch.zeros(batch_shape + (dims[i],), dtype=dtype, device=device)
            operators.append(expanded)

        return transform.batched_sum_kron_diagonal(operators)

    def _process_matrices(attr_name: str):
        """Combine matrices using batched_sum_kron."""
        values = [getattr(ctx, attr_name) for ctx in contexts]
        _check_callables(values, attr_name)

        if all(v is None for v in values):
            return None

        batch_shape = _get_common_batch_shape(values, is_matrix=True)

        operators = []
        for i, val in enumerate(values):
            expanded = val.expand(batch_shape + (dims[i], dims[i]))
            operators.append(expanded)
        return transform.batched_sum_kron(operators)

    def _process_basis():
        """Combine bases via Kronecker product."""
        bases = [ctx.basis for ctx in contexts]
        _check_callables(bases, "basis")

        has_any_basis = any(b is not None for b in bases)
        if not has_any_basis:
            return None
        batch_shape = _get_common_batch_shape(bases, is_matrix=True)
        basis_dtype = next((b.dtype for b in bases if b is not None), dtype)
        operators = []
        for i, basis in enumerate(bases):
            if basis is not None:
                expanded = basis.expand(batch_shape + (dims[i], dims[i]))
            else:
                expanded = torch.eye(dims[i], dtype=basis_dtype, device=device).expand(batch_shape + (dims[i], dims[i]))
            operators.append(expanded)

        return transform.batched_multi_kron(operators)

    def _process_init_density():
        """Combine init_density via Kronecker product."""
        densities = [ctx.init_density for ctx in contexts]
        _check_callables(densities, "init_density")

        if all(d is None for d in densities):
            return None
        if any(d is None for d in densities):
            batch_shape = _get_common_batch_shape(densities, is_matrix=True)
            complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            return torch.zeros(batch_shape + (total_dim,), dtype=complex_dtype, device=device)
        operators = []
        batch_shape = _get_common_batch_shape(densities, is_matrix=True)
        for i, dens in enumerate(densities):
            expanded = dens.expand(batch_shape + (dims[i], dims[i]))
            operators.append(expanded)
        return transform.batched_multi_kron(operators)

    return Context(
        basis=_process_basis(),
        init_populations=_process_init_populations(),
        init_density=_process_init_density(),
        free_probs=_process_matrices("free_probs"),
        driven_probs=_process_matrices("driven_probs"),
        out_probs=_process_diagonal_vectors("out_probs"),
        dephasing=_process_diagonal_vectors("dephasing"),
        relaxation_superop=_process_matrices("driven_probs"),
        profile=None,
        time_dimension=time_dimension,
        dtype=dtype,
        device=device
    )


def _multiply_homogeneous_contexts(
    contexts: tp.Sequence[Context],
) -> KroneckerContext:
    """ Build the tensor-product (Kronecker) composition of multiple homogeneous `Contexts.

    This function returns a new KroneckerContext describing the composite system that is the
    tensor product of the provided `contexts`. Behavior / rules follow the same
    conventions used in your concatenation function but adapted to tensor-product:

    - Hilbert space dimension is the product of individual system dimensions.

    - init_populations are combined with Kronecker product.
      If ALL values are None -> resulting parameter is None. If ANY is None,
      The output is ZERO tensor of the corresponding dimension.

    -`out_probs` : combined as sum of **diagonal** local operators
        `sum_i I ⊗ ... ⊗ diag(v_i) ⊗ ... ⊗ I` (i.e. vector x I + I x vector interpreted as diag).
        If ALL values are None -> resulting parameter is None. If ANY is -None, missing
          vectors are replaced with zero vectors of the corresponding dimension.

    - Square matrices (free_probs, driven_probs) are combined into a sum of local operators:
        K_total = sum_i (I ⊗ ... ⊗ K_i ⊗ ... ⊗ I)
      If ALL values are None -> resulting parameter is None. If ANY is non-None, missing
      matrices are replaced with zero matrices of the corresponding dimension.

    - Initial densities (init_density) combine by Kronecker product (complex dtype).
      If ALL densities are None -> resulting init_density is None. If ANY is non-None,
      missing ones are treated as zero matrices (so the total density may end up zero).

    - Initial densities (init_density) combine by Kronecker product (complex dtype).
      If ALL densities are None -> resulting init_density is None. If ANY is non-None,
      missing ones are treated as zero matrices (so the total density may end up zero).

    - superoperators are combined into a sum of local super operators:
        K_total = sum_i (I ⊗ ... ⊗ K_i ⊗ ... ⊗ I)

    - Callables (time-dependent parameters) are supported.

    - Batch shapes are broadcast; all non-None parameters must be broadcast-compatible.

    - Requires that all contexts share the same device/dtype.

    :param contexts:
        contexts: List of Context objects to concatenate.

    :return:
        New KroneckerContext object representing the direct-sum system.

    Raises:
        ValueError: If contexts list is empty or contains incompatible dtypes/devices
        TypeError: If any parameter is callable (time-dependent)
    """
    flag = any([context.eigen_basis_flag for context in contexts])
    if flag:
        raise NotImplementedError(
            "Multiplication of contexts does not support contexts that use the 'eigen' basis (or a None basis). "
            "This is because the resulting behavior would be ambiguous or ill-defined."
            "Please specify a defined physical basis such as 'zfs', 'multiplet', 'product',"
            "or another supported option."
        )
    return KroneckerContext(contexts)


class BaseContext(nn.Module, ABC):
    """Abstract base class defining the interface for a spin-system "Context".

    A Context encapsulates the physical model of relaxation and initial state in time-resolved EPR.
    It specifies:
      - The basis in which relaxation parameters (transition probabilities, loss rates, etc.) are defined,
      - The initial population vector or density matrix,
      - Time-dependence profiles for any parameter
      - Transformation rules to map all quantities into the field-dependent Hamiltonian eigenbasis.

    MarS distinguishes two relaxation paradigms:
      1. **Population-based**: Only diagonal elements (populations) evolve; off-diagonal coherences are neglected.
      2. **Density-matrix-based**: Full dynamic including coherences and dephsing.
    """

    def __init__(self, time_dimension: int = -3,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device("cpu")):
        """
        :param time_dimension: Dimension index where time-dependent values should be broadcasted.

                               Negative values index from the end of tensor dimensions.
        """
        super().__init__()
        self.time_dimension = time_dimension
        self.liouvilleator = transform.Liouvilleator

    @property
    @abstractmethod
    def time_dependant(self) -> bool:
        """Indicates whether any relaxation parameter depends explicitly on
        time."""
        pass

    @abstractmethod
    def __len__(self):
        pass

    @property
    @abstractmethod
    def contexted_init_population(self) -> bool:
        """True if the Context provides explicit initial populations (not
        thermal equilibrium)."""
        pass

    @abstractmethod
    def get_time_dependent_values(self, time: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """Evaluate time-dependent profile at specified time points.

        :param time: Time points tensor for evaluation
        :return: Profile values shaped for broadcasting along the
            specified time dimension.
        """
        pass

    @abstractmethod
    def get_transformed_init_populations(
            self, full_system_vectors: tp.Optional[torch.Tensor], normalize: bool = False
    ) -> tp.Optional[torch.Tensor]:
        """
        :param full_system_vectors:

        :param normalize: If True the returned populations are normalized along the last axis
            so they sum to 1. If False, populations are returned as-is.
        :return: Transformed populations with shape `[..., N]` (or `None` if no populations
            were provided).
        """
        pass

    @abstractmethod
    def get_transformed_init_density(
            self, full_system_vectors: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        """Return initial density matrix transformed into the Hamiltonian
        eigenbasis.

        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os `[...., M, N, N]`,
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
        forces the creator to compute these vectors.
        The default behavior, whether to calculate vectors or not,
        depends on the specific Spectra Manager and its settings.

        :return: density matrix  populations with shape `[... N, N]`
        """
        pass

    @abstractmethod
    def get_transformed_free_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ):
        """Return spontaneous (thermal) transition probabilities in the
        eigenbasis.

        These transitions are constrained by detailed balance at the specified temperature.
        Examples include spin-lattice relaxation (T1 processes) that drive the system toward
        thermal equilibrium.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time: Optional time points for evaluation if transition probabilities are
            time-dependent.

        :return: Transition rate matrix W with shape [..., N, N], where W_{ij} (i≠j) is the
            rate from state j to state i. Diagonal elements are not used directly but are
            computed internally to ensure probability conservation.

        Transformation rule for rates:
        If W^old is defined in the working basis, then in the eigenbasis:
            `W^new_{ij} = Σ_{k≠l} |<ψ_i^new|ψ_k^old>|² · |<ψ_j^new|ψ_l^old>|² · W^old_{kl}`

        Thermal correction (detailed balance):
        After transformation, rates are modified to satisfy:
            `W^new_{ij}/W^new_{ji} = exp(-(E_i - E_j)/k_B·T)`
        where E_i are eigenenergies and T is the temperature.
        """
        pass

    @abstractmethod
    def get_transformed_driven_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ):
        """Return induced (non-thermal) transition probabilities in the
        eigenbasis.

        These transitions are **not** subject to detailed balance.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian at each orientation/field,
                        shape `[..., M, N, N]`, where M is number of transitions, N is number of levels.
        :param time: Time points tensor for evaluation
        :return: Matrix of shape `[..., N, N]`.
        """
        pass

    @abstractmethod
    def get_transformed_out_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ) -> tp.Optional[torch.Tensor]:
        """Return loss (out-of-system) probabilities in the eigenbasis.

        These represent irreversible decay processes that remove population from the spin
        system entirely. Examples include:
        - Phosphorescence decay from triplet states
        - Chemical reaction products leaving the observed spin system

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Optional time_dep_values for evaluation if loss rates are time-dependent.

        :return: Loss rate vector O with shape `[..., N]`, where O_i is the rate at which
            population is lost from state i.

        Transformation rule:
            `O^new_i = Σ_k |<ψ_i^new|ψ_k^old>|² · O^old_k`

        Physical constraint: Loss rates must be non-negative (O_i ≥ 0).
        """
        pass

    @abstractmethod
    def get_transformed_free_superop(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None) -> tp.Optional[torch.Tensor]:
        """Return the spontaneous relaxation superoperator in Liouville space.

        This superoperator includes all spontaneous processes (thermal transitions, losses,
        dephsing) and is transformed to the eigenbasis. It is subsequently modified to
        obey detailed balance at the specified temperature.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Optional time_dep_values for evaluation if loss rates are time-dependent.

        :return: Superoperator R_free with shape `[..., N², N²]`, where N is the number of
            energy levels. This superoperator acts on vectorized density matrices.

        Construction method:
        The superoperator is built using Lindblad formalism from the constituent processes:
        1. Loss terms: `L_k = √O_k |k⟩⟨k|`
        2. Thermal transitions: `L_{kl} = √W_{kl} |k⟩⟨l|`
        3. Dephsing terms: `L_{kl} = √γ_{kl} |k⟩⟨l|` for `k≠l`

        Transformation rule:
            R_new = (U ⊗ U*) · R_old · (U ⊗ U*)
            where U is the basis transformation matrix and ⊗ denotes Kronecker product

        Thermal correction:
        After transformation, diagonal elements corresponding to population transfer are
        modified to satisfy detailed balance:
            `R_{iijj}^new = R_{iijj}^old · exp(-(E_i-E_j)/k_B·T) / (1 + exp(-(E_i-E_j)/k_B·T))`
            `R_{jjii}^new = R_{jjii}^old · 1 / (1 + exp(-(E_i-E_j)/k_B·T))`

        where E_i are eigenenergies and T is temperature.
        """
        pass

    @abstractmethod
    def get_transformed_driven_superop(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ) -> tp.Optional[torch.Tensor]:
        """Return the induced relaxation superoperator in Liouville space.

        This superoperator contains only non-thermal (driven) processes that are NOT
        constrained by detailed balance. It is transformed to the eigenbasis without
        thermal correction.

        Construction method:
        Built using Lindblad formalism from induced transition rates:
            `L_{kl} = √D_{kl} |k⟩⟨l|`

        Transformation rule:
            `R_new = (U ⊗ U*) · R_old · (U ⊗ U*)`
            where U is the basis transformation matrix and ⊗ denotes Kronecker product

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Optional time_dep_values for evaluation if loss rates are time-dependent.
        :return: Superoperator R_driven with shape `[..., N², N²]`.

        Unlike the free superoperator, NO thermal correction is applied to these elements,
        as they represent processes that actively drive the system away from equilibrium.

        Note: If a user provides a complete superoperator directly (bypassing individual
        rates), it is interpreted as an induced superoperator and no thermal correction
        is applied.
        """
        pass


class TransformedContext(BaseContext):
    """Concrete base class implementing basis transformation logic for Context
    subclasses.

    This class provides the machinery to transform physical quantities between different
    basis representations. The core assumption is that relaxation parameters (transition
    rates, initial populations, etc.) are often most naturally defined in a basis that
    differs from the field-dependent eigenbasis required for dynamics calculations.

    Supported transformations:
    1. Vector transformation: For initial populations and loss rates
    2. Matrix transformation: For transition probability matrices
    3. Density matrix transformation: For initial quantum states
    4. Superoperator transformation: For Liouville-space relaxation operators

    Basis specification:
    - Can be provided as explicit transformation matrices or as string identifiers:
        * "eigen": Hamiltonian eigenbasis at the resonance field (no transformation needed)
        * "zfs": Zero-field splitting basis (eigenvectors of the field-independent part)
        * "xyz": The common molecular triplet basis: Tx, Ty, Tz.
        * "multiplet": Total spin multiplet basis |S, M⟩
        * "product": Product basis of individual spin projections |m_s1, m_s2, ...⟩

    The class automatically selects appropriate transformation methods based on basis type
    and caches transformation coefficients to avoid redundant computations.
    """
    def _setup_transformers(self):
        """Configure transformation methods based on the specified basis.

        This method sets up the appropriate transformation functions for vectors, matrices,
        density matrices, and superoperators based on whether a basis transformation is
        needed (self.basis is not None).

        When no transformation is needed (eigenbasis):
        - All transformation methods become identity operations (_transformed_skip)
        """
        if self.basis is None:
            self.transformed_vector = self._transformed_skip
            self.transformed_matrix = self._transformed_skip
            self.transformed_density = self._transformed_skip
            self.transformed_superop = self._transformed_skip
            self.transformed_populations = self._transformed_skip
        else:
            self.transformed_vector = self._transformed_vector_basis
            self.transformed_populations = self._transformed_population_basis
            self.transformed_matrix = self._transformed_matrix_basis

            self.transformed_density = self._transformed_density_basis
            self.transformed_superop = self._transformed_superop_basis

    @abstractmethod
    def _transformed_skip(
            self, system_data: tp.Optional[torch.Tensor],
            full_system_vectors: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        return system_data

    @abstractmethod
    def _transformed_vector_basis(
            self, vector: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ) -> tp.Optional[torch.Tensor]:
        """Transform a vector from one basis to another."""
        pass

    def _transformed_population_basis(
            self, vector: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ) -> tp.Optional[torch.Tensor]:
        """Transform a vector from one basis to another."""
        pass

    @abstractmethod
    def _transformed_matrix_basis(
            self, matrix: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ) -> tp.Optional[torch.Tensor]:
        """Transform a matrix from one basis to another."""
        pass

    def _transformed_density_basis(
            self, density: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ) -> tp.Optional[torch.Tensor]:
        """Transform a density matrix from one basis to another."""
        raise NotImplementedError

    def _transformed_superop_basis(
            self, superop: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ) -> tp.Optional[torch.Tensor]:
        """Transform a super operator from one basis to another."""
        raise NotImplementedError

    def get_transformed_free_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ) -> tp.Optional[torch.Tensor]:
        """Return spontaneous (thermal) transition probabilities in the
        eigenbasis.

        These transitions are constrained by detailed balance at the specified temperature.
        Examples include spin-lattice relaxation (T1 processes) that drive the system toward
        thermal equilibrium.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time: Optional time points for evaluation if transition probabilities are
            time-dependent.

        :return: Transition rate matrix W with shape `[..., N, N]`, where `W_{ij} (i≠j)` is the
            rate from state j to state i. Diagonal elements are not used directly but are
            computed internally to ensure probability conservation.

        Transformation rule for rates:
        If `W^old` is defined in the working basis, then in the eigenbasis:
           `W^new_{ij} = Σ_{k≠l} |<ψ_i^new|ψ_k^old>|² · |<ψ_j^new|ψ_l^old>|² · W^old_{kl}`

        Thermal correction (detailed balance):
        After transformation, rates are modified to satisfy:
            `W^new_{ij}/W^new_{ji} = exp(-(E_i - E_j)/k_B·T)`
        where `E_i` are eigenenergies and T is the temperature.
        """
        _free_probs = self._get_free_probs_tensor(time_dep_values)
        return self.transformed_matrix(_free_probs, full_system_vectors)

    def get_transformed_driven_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ) -> tp.Optional[torch.Tensor]:
        """Return induced (non-thermal) transition probabilities in the
        eigenbasis.

        These transitions are NOT constrained by detailed balance and represent external
        driving forces or non-equilibrium processes.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time: Optional time points for evaluation if transition probabilities are
            time-dependent.

        :return: Transition rate matrix D with shape `[..., N, N]`, where `D_{ij} (i≠j)` is the
            non-thermal rate from state j to state i.

        Transformation rule is the same as for free probabilities:
            `D^new_{ij} = Σ_{k≠l} |<ψ_i^new|ψ_k^old>|² · |<ψ_j^new|ψ_l^old>|² · D^old_{kl}`

        Note: No thermal correction is applied to these rates as they represent non-equilibrium
        processes that actively drive the system away from thermal equilibrium.
        """
        _driven_probs = self._get_driven_probs_tensor(time_dep_values)
        return self.transformed_matrix(_driven_probs, full_system_vectors)

    def get_transformed_out_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ) -> tp.Optional[torch.Tensor]:
        """Return loss (out-of-system) probabilities in the eigenbasis.

        These represent irreversible decay processes that remove population from the spin
        system entirely. Examples include:
        - Phosphorescence decay from triplet states
        - Chemical reaction products leaving the observed spin system

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Optional time_dep_values for evaluation if loss rates are time-dependent.

        :return: Loss rate vector O with shape [..., N], where O_i is the rate at which
            population is lost from state i.

        Transformation rule:
            ``O^new_i = Σ_k |<ψ_i^new|ψ_k^old>|² · O^old_k``

        Physical constraint: Loss rates must be non-negative (O_i ≥ 0).
        """
        _out_probs = self._get_out_probs_tensor(time_dep_values)
        return self.transformed_vector(_out_probs, full_system_vectors)

    @property
    def free_superop(self):
        """:return: free superoperator created from other relaxation parameters (free_probs, dephasing, out_probs)."""
        if self._default_free_superop is None:
            if (self.free_probs is None) and (self.out_probs is None) and (self.dephasing is None):
                return None
            return self._create_free_superop
        else:
            return self._default_free_superop

    @property
    def driven_superop(self):
        """
        :return: free superoperator created from other relaxation parameters (driven_probs) or user-defined.

            superoperator
        """
        if self._default_driven_superop is None:
            if self.driven_probs is None:
                return None
            return self._create_driven_superop
        else:
            return self._default_driven_superop

    def get_transformed_free_superop(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ) -> tp.Optional[torch.Tensor]:
        """Return the spontaneous relaxation superoperator in Liouville space.

        This superoperator includes all spontaneous processes (thermal transitions, losses,
        dephasing) and is transformed to the eigenbasis. It is subsequently modified to
        obey detailed balance at the specified temperature.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Optional time_dep_values for evaluation if loss rates are time-dependent.

        :return: Superoperator R_free with shape [..., N², N²], where N is the number of
            energy levels. This superoperator acts on vectorized density matrices.

        Construction method:
        The superoperator is built using Lindblad formalism from the constituent processes:
        1. Loss terms: ``L_k = √O_k |k⟩⟨k|``
        2. Thermal transitions: L_{kl} = ``√W_{kl} |k⟩⟨l|``
        3. Dephasing terms: L_{kl} = ``√γ_{kl} |k⟩⟨l|`` for ``k≠l``

        Transformation rule:
            ``R_new = (U ⊗ U*) · R_old · (U ⊗ U*)``
            where U is the basis transformation matrix and ⊗ denotes Kronecker product

        Thermal correction:
        After transformation, diagonal elements corresponding to population transfer are
        modified to satisfy detailed balance:
                R_{iijj}^new = R_{iijj}^old · exp(-(E_i-E_j)/k_B·T) / (1 + exp(-(E_i-E_j)/k_B·T))
            R_{jjii}^new = R_{jjii}^old · 1 / (1 + exp(-(E_i-E_j)/k_B·T))

        where E_i are eigenenergies and T is temperature.
        """
        _relaxation_superop = self._get_free_superop_tensor(time_dep_values)
        return self.transformed_superop(_relaxation_superop, full_system_vectors)

    def get_transformed_driven_superop(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ):
        """Return the induced relaxation superoperator in Liouville space.

        This superoperator contains only non-thermal (driven) processes that are NOT
        constrained by detailed balance. It is transformed to the eigenbasis without
        thermal correction.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Optional time_dep_values for evaluation if loss rates are time-dependent.

        :return: Superoperator R_driven with shape [..., N², N²].

        Construction method:
        Built using Lindblad formalism from induced transition rates:
            L_{kl} = √D_{kl} |k⟩⟨l|
        If the initial superoperator is given then it is used as superoperator

        Transformation rule:
            R_new = (U ⊗ U*) · R_old · (U ⊗ U*)
            where U is the basis transformation matrix and ⊗ denotes Kronecker product

        Unlike the free superoperator, NO thermal correction is applied to these elements,
        as they represent processes that actively drive the system away from equilibrium.

        Note: If a user provides a complete superoperator directly (bypassing individual
        rates), it is interpreted as an induced superoperator and no thermal correction
        is applied.
        """
        _relaxation_superop = self._get_driven_superop_tensor(time_dep_values)
        return self.transformed_superop(_relaxation_superop, full_system_vectors)

    def _extract_free_populations_superop(self, time_dep_values):
        if (self.out_probs is not None) and (self.free_probs is not None):
            _out_probs = self._get_out_probs_tensor(time_dep_values)
            _free_probs = self._get_free_probs_tensor(time_dep_values)
            return self.liouvilleator.lindblad_dissipator_superop(_free_probs) +\
                torch.diag_embed(
                    self.liouvilleator.anticommutator_superop_diagonal(-0.5 * _out_probs), dim1=-1, dim2=-2)

        elif (self.out_probs is not None) and (self.free_probs is None):
            _out_probs = self._get_out_probs_tensor(time_dep_values)
            return torch.diag_embed(
                self.liouvilleator.anticommutator_superop_diagonal(-0.5 * _out_probs), dim1=-1, dim2=-2)

        elif (self.out_probs is None) and (self.free_probs is not None):
            _free_probs = self._get_free_probs_tensor(time_dep_values)
            return self.liouvilleator.lindblad_dissipator_superop(_free_probs)

        else:
            return None

    def _create_driven_superop(
            self,
            time_dep_values: tp.Optional[torch.Tensor] = None
    ):
        if self.driven_probs is None:
            return None
        else:
            _driven_probs = self._get_driven_probs_tensor(time_dep_values)
            _relaxation_superop = self.liouvilleator.lindblad_dissipator_superop(_driven_probs)
            return _relaxation_superop

    def _create_free_superop(
            self,
            time_dep_values: tp.Optional[torch.Tensor] = None
    ):
        """For the dephasing, free_probs, out_probs create free-superoperator.

        :param time_dep_values: Optional time_dep_values for evaluation
            if loss rates are time-dependent.
        :return: free superoperator tensor or None
        """
        if (self.free_probs is None) and (self.dephasing is None) and (self.out_probs is None):
            return None

        _density_condition = (self.out_probs is not None) or (self.free_probs is not None)
        if self.dephasing is not None and _density_condition:
            _dephasing = self._get_dephasing_tensor(time_dep_values)
            _relaxation_superop =\
                self.liouvilleator.lindblad_dephasing_superop(_dephasing) +\
                self._extract_free_populations_superop(time_dep_values)
            return _relaxation_superop

        elif (self.dephasing is None) and _density_condition:
            return self._extract_free_populations_superop(time_dep_values)

        else:
            _dephasing = self._get_dephasing_tensor(time_dep_values)
            _relaxation_superop = self.liouvilleator.lindblad_dephasing_superop(_dephasing)
            return _relaxation_superop


class Context(TransformedContext):
    """Primary implementation of a spin relaxation context for a sample.

    This class provides a flexible interface for specifying relaxation models through:
    - Basis specification (explicit matrix or string identifier)
    - Initial state definition (populations or full density matrix)
    - Transition rate matrices (spontaneous, induced, loss terms)
    - Dephasing rates (for density-matrix approach)
    - Time-dependent profile functions

    Physical interpretation of parameters:
    - free_probs: Thermal (Boltzmann-weighted) transition probabilities between states
    - driven_probs: Non-thermal transition probabilities not constrained by detailed balance
    - out_probs: Irreversible loss rates from states (e.g., phosphorescence)
    - dephasing: Probabilities of dephasing relaxation (decay of off-diagonal terms of a density matrix)
    - init_populations: Initial state populations in the working basis
    - init_density: Complete initial quantum state (includes coherences)

    The class supports both simple constant rates and time-dependent functions for all
    rate parameters, enabling modeling of complex time-evolving systems.

    Algebraic operations:
    - Addition (+): Combines multiple relaxation mechanisms acting on the SAME system
    - Tensor product (@): Combines independent subsystems into a kronecker quantum system

    These operations follow the physical rules described in the MarS documentation and
    enable construction of sophisticated relaxation models from simpler components.
    """

    def __init__(
            self,
            basis: tp.Optional[tp.Union[torch.Tensor, str, None]] = None,
            sample: tp.Optional[spin_model.MultiOrientedSample] = None,
            init_populations: tp.Optional[tp.Union[torch.Tensor, tp.List[float]]] = None,
            init_density: tp.Optional[torch.Tensor] = None,

            free_probs: tp.Optional[tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]] = None,
            driven_probs: tp.Optional[tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]] = None,
            out_probs: tp.Optional[
                tp.Union[torch.Tensor, tp.List[float], tp.Callable[[torch.Tensor], torch.Tensor]]] = None,

            dephasing: tp.Optional[
                tp.Union[torch.Tensor, tp.List[float], tp.Callable[[torch.Tensor], torch.Tensor]]] = None,
            relaxation_superop: tp.Optional[tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]] = None,

            profile: tp.Optional[tp.Callable[[torch.Tensor], torch.Tensor]] = None,
            time_dimension: int = -3,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cpu")
    ):
        """
        :param basis: torch.Tensor or str or None, optional.

        Basis specifier. Three allowed forms:
          -`str`: one of {"zfs", "multiplet", "product", "eigen"}. If a string is
            given, `sample` **must** be provided so the basis can be constructed.
            * "zfs"       : eigenvectors of the zero-field Hamiltonian (unsqueezed)
            * "xyz": The common molecular triplet basis: |Tx>, |Ty⟩, |Tz⟩. It is defined only for triplet states
            * "multiplet" : total spin multiplet basis |S, M⟩
            * "product"   : Basis of individual spin projections |ms1, ms2, ..., is1, ...⟩
            * "zeeman"    : the basis of the Hamiltonian in infinite magnetic field - Eigen vectors of Gz
            * "eigen"     : use the eigen basis at the resonance fields (represented as `None`)
            In all cases except the product case, the basis is sorted in ascending order of eigenvalues.
            In the product basis, sorting occurs in descending order of projections.
          - `torch.Tensor`: explicit basis tensor. Expected shapes:
                `[N, N]` for a single basis or `[R / 1, K / 1, N, N]` for R orientations and K transitions
            Tensor must be square in its last two dimensions.
          - `None`: indicates the eigen basis will be used (no transformation).

        :param sample: MultiOrientedSample or None, optional
            Required when `basis` is specified as a `str`. Provides helper methods
            for building basis tensors for the requested basis type.

        :param init_populations: torch.Tensor or list[float] or None, optional
            The param is ignored if init_density is provided!

            Initial populations at the working basis or as a list. Shape `[..., N]`.
            If provided, it will be converted to a `torch.tensor`.

        :param init_density: torch.Tensor, optional.
            Initial density of the spin system. Shape [..., N, N]
            If provided then init_populations will be ignored for computations,
            however for the most cases it is better to set only one among init_populations / init_density

        :param free_probs:   torch.Tensor or callable or None, optional
            Thermal (Boltzmann-weighted) transition probabilities.
            It can be set as symmetrix matrix of mean transition probabilities. Accepts either:
              - a tensor shaped `[..., N, N]`
                [[0,  w],
                 [w, 0]]
              , or
              - a callable `f(time) -> tensor` that returns the tensor at requested times.

        :param driven_probs: torch.Tensor [..., N, N] or None
            Probabilities of driven transitions (e.g. due to external driving).
            DR matrix is a matrix of driven transitions that are not connected by thermal equilibrium:
             [[0,  dr_1],
             [dr_2, 0]]

        :param out_probs: torch.Tensor or list[float] or callable or None, optional
            Out-of-system transition probabilities (loss terms). Expected shapes:
              - `[..., N]` (or `[..., T, N]`), or
              - Python list of length `N` (converted to tensor), or
              - callable `f(time) -> tensor`.

        :param dephasing: torch.Tensor or callable or None, optional
            dephasing vector probabilities with shape [N].
            Each element set the Decreasing of the non-diagonal matrix elements of density matrix
            d <i|rho|j> / dt = -(dephasing[i] + dephasing[j]) / 2 * <i|rho|j>
            If relaxation_superop is given, then this parameter is ignored
        For implementation of dephasing, out_probs, driven_probs, free_probs we use Lindblad form of relaxation.

        :param relaxation_superop: torch.Tensor or callable or None, optional
            Full superoperator of relaxation rates for density matrix
            with shape [N*N, N*N]. Any elements can be given.
            If it is given then  driven_probs are ignored
            but free_probs, dephasing, out_probs are computed
            After transformation the thermal correction is not used for this term

        :param profile: callable or None, optional
            Callable `profile(time: torch.Tensor) -> torch.Tensor` that returns
            time-dependent scalars/arrays used by `get_time_dependent_values`.
            If None, `get_time_dependent_values` will raise if called.

        :param time_dimension: int, optional
            Axis index where time should be broadcasted in returned tensors.
            Default -3 to match the code broadcasting conventions.
        """
        super().__init__(time_dimension=time_dimension, dtype=dtype, device=device)
        self.transformation_probabilities = None
        self.transformation_unitary = None
        self.transformation_liouville = None

        self.init_populations = self._set_init_populations(init_populations, init_density, dtype, device)
        init_density_real, init_density_imag = self._set_init_density(init_density)
        self.register_buffer("_init_density_real", init_density_real)
        self.register_buffer("_init_density_imag", init_density_imag)

        if isinstance(out_probs, torch.Tensor) or out_probs is None:
            pass
        else:
            out_probs = torch.tensor(out_probs, device=device, dtype=dtype)
        self.register_buffer("out_probs", out_probs)

        self.free_probs = free_probs
        self.driven_probs = driven_probs

        dephasing = self._get_init_dephasing(dephasing, relaxation_superop, device=device, dtype=dtype)
        self.register_buffer("dephasing", dephasing)
        self._default_free_superop = None
        self._default_driven_superop = relaxation_superop

        self.profile = profile

        self.eigen_basis_flag = False
        if isinstance(basis, str):
            self.basis = self._create_basis_from_string(basis, sample)
        elif isinstance(basis, torch.Tensor):
            if basis.shape[-1] != basis.shape[-2]:
                raise ValueError("Basis tensor must be square (last two dimensions must match)")
            self.basis = basis
        elif basis is None:
            self.eigen_basis_flag = True
            self.basis = basis
        else:
            raise ValueError("Basis must be either None, string or tensor")
        self._setup_prob_getters()
        self._setup_transformers()
        self._spin_system_dim = None

    @property
    def spin_system_dim(self) -> tp.Optional[int]:
        """Returns the dimension N of the spin system Hilbert space.

        Determined from available attributes in the following order:
        1. From `init_populations` if provided (shape [..., N])
        2. From `init_density` if provided (shape [..., N, N])
        3. From `basis` tensor if provided (shape [..., N, N])
        4. From rate matrices (`free_probs`, `driven_probs`) if they are tensors (shape [..., N, N])
        5. From vector parameters (`out_probs`, `dephasing`) if they are tensors (shape [..., N])
        6. From supeoperator of relaxatin (shape [..., N^2, N^2])

        Returns 0 if no dimension can be inferred.
        """
        if self._spin_system_dim is not None:
            return self._spin_system_dim

        if self.init_populations is not None:
            self._spin_system_dim = self.init_populations.shape[-1]
            return self._spin_system_dim

        if self._init_density_real is not None:
            self._spin_system_dim = self._init_density_real.shape[-1]
            return self._spin_system_dim

        if self.basis is not None:
            self._spin_system_dim = self.basis.shape[-1]
            return self._spin_system_dim

        for param in [self.free_probs, self.driven_probs]:
            if isinstance(param, torch.Tensor):
                self._spin_system_dim = param.shape[-1]
                return self._spin_system_dim

        if self._default_driven_superop is not None:
            self._spin_system_dim = int(math.sqrt(self._default_driven_superop.shape[-1]))
            return self._spin_system_dim

        return None

    @spin_system_dim.setter
    def spin_system_dim(self, dim: int) -> None:
        """
        Set the spin system dimension for creation of Zero Contexts.
        Use it if you need padding Context

        :param dim: the dimension of zero Context
        :return: None
        """
        self._spin_system_dim = dim

    @property
    def device(self) -> tp.Optional[torch.device]:
        """Returns the device context.

        Determined from available attributes in the following order:
        1. From `init_populations` if provided (shape [..., N])
        2. From `init_density` if provided (shape [..., N, N])
        3. From `basis` tensor if provided (shape [..., N, N])
        4. From rate matrices (`free_probs`, `driven_probs`) if they are tensors (shape [..., N, N])
        5. From vector parameters (`out_probs`, `dephasing`) if they are tensors (shape [..., N])
        6. From supeoperator of relaxatin (shape [..., N^2, N^2])
        """
        if self.init_populations is not None:
            return self.init_populations.device

        if self._init_density_real is not None:
            return self._init_density_real.device

        if self.basis is not None:
            return self.basis.device

        for param in [self.free_probs, self.driven_probs]:
            if isinstance(param, torch.Tensor):
                return param.device

        if self._default_driven_superop is not None:
            return self._default_driven_superop.device

        return None

    @property
    def dtype(self) -> tp.Optional[torch.dtype]:
        """Returns the dtype of a context.

        Determined from available attributes in the following order:
        1. From `init_populations` if provided (shape [..., N])
        2. From `init_density` if provided (shape [..., N, N])
        3. From `basis` tensor if provided (shape [..., N, N])
        4. From rate matrices (`free_probs`, `driven_probs`) if they are tensors (shape [..., N, N])
        5. From vector parameters (`out_probs`, `dephasing`) if they are tensors (shape [..., N])
        6. From supeoperator of relaxatin (shape [..., N^2, N^2])

        """
        if self.init_populations is not None:
            return self.init_populations.dtype

        if self._init_density_real is not None:
            return self._init_density_real.dtype

        if self.basis is not None:
            return self.basis.real.dtype

        for param in [self.free_probs, self.driven_probs]:
            if isinstance(param, torch.Tensor):
                return param.dtype
        if self._default_driven_superop is not None:
            return self._default_driven_superop.real.dtype

        return None

    def __len__(self):
        """This is just entire context. Let's set its len as 1"""
        return 1

    @property
    def time_dependant(self) -> bool:
        """Indicates whether the system parameters are time-dependent
        profile."""
        return self.profile is not None

    @property
    def contexted_init_population(self) -> bool:
        """Indicates whether initial populations are provided via context."""
        return (self.init_populations is not None) or (self._init_density_real is not None)

    @property
    def contexted_init_density(self) -> bool:
        """Indicates whether the initial density matrix is taken from context
        rather than thermal equilibrium.

        Returns True if either `init_populations` or a user-defined initial density matrix is provided;
        otherwise False.
        """
        return (self.init_populations is not None) or (self._init_density_real is not None)

    def close_context(self) -> None:
        """
        Reset Context cach parameters: basis coefficients
        """
        self.transformation_probabilities = None
        self.transformation_unitary = None
        self.transformation_liouville = None

    @property
    def init_density(self) -> tp.Optional[torch.Tensor]:
        """Returns the initial density matrix.

        The matrix is constructed in the following order of precedence:
        1. If a user-provided complex density matrix is available, it is returned density matrix.
        2. If only `init_populations` are given, a diagonal density matrix is created from them.
        3. If neither is provided, returns None.

        :return: Initial density matrix as a complex-valued torch.Tensor, or None if unspecified.
        """
        if self._init_density_real is None:
            if self.init_populations is None:
                return None
            self._init_density_real = torch.diag_embed(self.init_populations, dim1=-1, dim2=-2)
            self._init_density_imag = torch.zeros_like(self._init_density_real)
        return torch.complex(self._init_density_real, self._init_density_imag)

    def _set_init_density(self, init_density: tp.Optional[torch.Tensor]) ->\
            tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]:
        if init_density is None:
            return None, None
        else:
            return init_density.real, init_density.imag

    def _set_init_populations(self,
                              init_populations: tp.Optional[tp.Union[torch.Tensor, list[float]]],
                              init_density: tp.Optional[torch.Tensor],
                              dtype: torch.dtype, device: torch.device) -> tp.Optional[torch.Tensor]:
        if init_density is None:
            if init_populations is None:
                return None
            elif init_populations is not None:
                return torch.tensor(init_populations, dtype=dtype, device=device)
        else:
            return init_populations

    def _get_init_dephasing(self,
                        dephasing: tp.Optional[
                            tp.Union[torch.Tensor, tp.List[float], tp.Callable[[torch.Tensor], torch.Tensor]]
                        ],
                        relaxation_superop: tp.Optional[torch.Tensor], dtype: torch.dtype, device: torch.device)\
            -> tp.Optional[tp.Union[tp.Callable[[torch.Tensor], torch.Tensor], torch.Tensor]]:
        if relaxation_superop is None:
            if isinstance(dephasing, torch.Tensor) or dephasing is None:
                pass
            else:
                dephasing = torch.tensor(dephasing, device=device, dtype=dtype)
            return dephasing
        else:
            return None

    def _compute_transformation_probabilities(self, full_system_vectors: tp.Optional[torch.Tensor]):
        """Compute and cache basis transformation probabilities for the initial
        population and all other real values tranformations."""
        if self.transformation_probabilities is not None:
            return self.transformation_probabilities
        else:
            self.transformation_probabilities = transform.get_transformation_probabilities(
                self.basis, full_system_vectors
            )
            return self.transformation_probabilities

    def _compute_transformation_unitary(self, full_system_vectors: tp.Optional[torch.Tensor]):
        """Compute and cache basis transformation coefficients for the density
        matrix transformation."""
        if self.transformation_unitary is not None:
            return self.transformation_unitary
        else:
            self.transformation_unitary = transform.basis_transformation(
                self.basis, full_system_vectors
            )
            return self.transformation_unitary

    def _compute_transformation_liouville(self, full_system_vectors: tp.Optional[torch.Tensor]):
        """Compute and cache basis transformation coefficients for the
        superoperator transformation."""
        if self.transformation_liouville is not None:
            return self.transformation_liouville
        else:
            self.transformation_liouville = transform.compute_liouville_basis_transformation(
                self.basis, full_system_vectors
            )
            return self.transformation_liouville

    def _transformed_skip(
            self, system_data: tp.Optional[torch.Tensor],
            full_system_vectors: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        return system_data

    def _transformed_vector_basis(
            self, vector: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform a vector from one basis to another."""
        if vector is None:
            return None
        else:
            probabilities = self._compute_transformation_probabilities(full_system_vectors)
            return transform.transform_state_weights_to_new_basis(vector, probabilities)

    def _transformed_population_basis(
            self, vector: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform a population from one basis to another."""
        return self._transformed_vector_basis(vector, full_system_vectors)

    def _transformed_matrix_basis(
            self, matrix: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform a matrix from one basis to another."""
        if matrix is None:
            return None
        else:
            probabilities = self._compute_transformation_probabilities(full_system_vectors)
            return transform.transform_rate_matrix_to_new_basis(matrix, probabilities)

    def _transformed_density_basis(
            self, density_matrix: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform density matrix from one basis to another."""
        if density_matrix is None:
            return None
        else:
            coeffs = self._compute_transformation_unitary(full_system_vectors)
            return transform.transform_operator_to_new_basis(density_matrix, coeffs)

    def _transformed_superop_basis(
            self, relaxation_superop: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform relaxation superoperator from one basis to another."""
        if relaxation_superop is None:
            return None
        else:
            coeffs = self._compute_transformation_liouville(full_system_vectors)
            return transform.transform_superop_to_new_basis(transform_to_complex(relaxation_superop), coeffs)

    def _create_basis_from_string(self, basis_type: str, sample: tp.Optional[spin_model.MultiOrientedSample]):
        """Factory method to create basis from string identifier."""
        if basis_type == "eigen":
            self.eigen_basis_flag = True
            return None
        if sample is None:
            raise ValueError("Sample must be provided when basis is specified as a string method")

        if basis_type == "zfs":
            return sample.get_zero_field_splitting_basis().unsqueeze(-3)
        elif basis_type == "xyz":
            return sample.get_xyz_basis().unsqueeze(-3)
        elif basis_type == "multiplet":
            return sample.get_spin_multiplet_basis()\
                .unsqueeze(-3).unsqueeze(-4).to(sample.complex_dtype)
        elif basis_type == "product":
            return sample.get_product_state_basis()\
                .unsqueeze(-3).unsqueeze(-4).to(sample.complex_dtype)
        elif basis_type == "zeeman":
            return sample.get_zeeman_basis().unsqueeze(-3)
        else:
            raise KeyError(
                "Basis must be one of:\n"
                "1) torch.Tensor with shape [R, N, N] or [N, N], where R is number of orientations\n"
                "2) str: 'zfs', 'multiplet', 'product', 'eigen'\n"
                "3) None (will use eigen basis at given magnetic fields)"
            )

    def _setup_single_getter(
            self, getter: tp.Optional[tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]]) ->\
            tp.Callable[[torch.Tensor], torch.Tensor]:
        if callable(getter):
            return lambda t: getter(t)
        else:
            return lambda t: getter

    def _setup_prob_getters(self):
        """Setup getter methods for probabilities based on callable status at
        initialization."""
        current_free_probs = self.free_probs
        self._get_free_probs_tensor = self._setup_single_getter(current_free_probs)

        current_driven_probs = self.driven_probs
        self._get_driven_probs_tensor = self._setup_single_getter(current_driven_probs)

        current_out_probs = self.out_probs
        self._get_out_probs_tensor = self._setup_single_getter(current_out_probs)

        current_dephasing = self.dephasing
        self._get_dephasing_tensor = self._setup_single_getter(current_dephasing)

        current_free_superop = self.free_superop
        self._get_free_superop_tensor = self._setup_single_getter(current_free_superop)

        current_driven_superop = self.driven_superop
        self._get_driven_superop_tensor = self._setup_single_getter(current_driven_superop)

    def get_time_dependent_values(self, time: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """Evaluate time-dependent profile at specified time points.

        Evaluate time-dependent values at specified time points.
        :param time: Time points tensor for evaluation
        :return: Profile values shaped for broadcasting along the
            specified time dimension.
        """
        return self.profile(time)[(...,) + (None,) * (-(self.time_dimension+1))]

    # Must be rebuild futher
    def get_transformed_init_populations(
            self, full_system_vectors: tp.Optional[torch.Tensor], normalize: bool = True
    ) -> tp.Optional[torch.Tensor]:
        """Return initial populations transformed into the field-dependent
        Hamiltonian eigenbasis.

        This method handles the critical transformation from the working basis (where initial
        populations are defined) to the eigenbasis of the field-dependent Hamiltonian (where
        dynamics are computed).

        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
        forces the creator to compute these vectors
        The default behavior, whether to calculate vectors or not,
        depends on the specific Spectra Manager and its settings.

        Transformation rule:
        If |ψ_k^old> are basis states in working basis and |ψ_j^new> in eigenbasis, then:
            p_j^new = Σ_k |<ψ_j^new|ψ_k^old>|² · p_k^old
        This ensures conservation of probability under basis change.

        :param normalize: If True (default) the returned populations are normalized along the last axis
        so they sum to 1 (useful for probabilities). If False, populations are returned
        as-is.
        :return: Initial populations with shape [...N]
        """
        if self._init_density_real is not None:
            density = self.get_transformed_init_density(full_system_vectors)
            populations = torch.diagonal(density.real, dim1=-2, dim2=-1)
        elif self.init_populations is not None:
            populations = self.transformed_populations(self.init_populations, full_system_vectors)
        else:
            return None

        if normalize:
            return populations / torch.sum(populations, dim=-1, keepdim=True)
        else:
            return populations

    def get_transformed_init_density(
            self, full_system_vectors: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        """Return initial density matrix transformed into the field-dependent
        eigenbasis.

        This method is used in the density-matrix paradigm where full quantum state evolution
        is computed, including coherences between energy levels.

        Physical interpretation:
        - Diagonal elements represent populations
        - Off-diagonal elements represent quantum coherences between states

        Transformation rule:
        If U is the unitary transformation matrix between bases (U_{jk} = <ψ_j^new|ψ_k^old>),
            ρ^new = U · ρ^old · U⁺
        where U⁺ is the conjugate transpose of U.

        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
        forces the creator to compute these vectors
        The default behavior, whether to calculate vectors or not,
        depends on the specific Spectra Manager and its settings.

        :return: Initial densities with shape [...N, N]
        """
        return self.transformed_density(self.init_density, full_system_vectors)

    def __add__(self, other: BaseContext) -> SummedContext:
        """"""
        if isinstance(other, SummedContext):
            return SummedContext([self] + list(other.component_contexts))
        else:
            return SummedContext([self, other])

    def __matmul__(self, other: BaseContext) -> tp.Union[KroneckerContext, SummedContext]:
        """"""
        if isinstance(other, SummedContext):
            return multiply_contexts([[self], other])
        elif isinstance(other, Context):
            return _multiply_homogeneous_contexts([self, other])
        elif isinstance(other, KroneckerContext):
            return _multiply_homogeneous_contexts([self, *other.component_contexts])
        else:
            raise NotImplementedError("Only Context can be multiplied on a context")


class KroneckerContext(TransformedContext):
    """Context representing a composite quantum system formed by tensor product
    of subsystems.

    This class models a quantum system consisting of multiple interacting or non-interacting
    subsystems (e.g., electron-nuclear spin systems, multiple chromophores). Each subsystem
    is described by its own context, and the composite system follows quantum mechanical
    tensor product rules.

    Key physical principles:
    1. State space: The Hilbert space of the composite system is the tensor product of
       subsystem Hilbert spaces: H = H₁ ⊗ H₂ ⊗ ... ⊗ Hₙ
    2. Initial state: The initial density matrix is the tensor product of subsystem states:
       ρ = ρ₁ ⊗ ρ₂ ⊗ ... ⊗ ρₙ
    3. Dynamics: Each subsystem evolves according to its own Hamiltonian and relaxation
       operators, which are embedded into the composite space using tensor products with
       identity operators: K = K1 ⊗ I + I ⊗ K2

    Transformation rules:
    - Basis transformations use Clebsch-Gordan coefficients to map between product bases
      and coupled bases
    - Initial populations are transformed using tensor products of transformation matrices
    - Relaxation superoperators are transformed using Kronecker products of basis
      transformation matrices

    Composite contexts can be created using the @ operator:
        composite_context = context1 @ context2 @ context3

    Warning!
    Composite context only determined for not-None basis (eigen).
    If it is not Eigen, please considet other types of basis
    """
    def __init__(self,
                 contexts: tp.Sequence[Context],
                 time_dimension: int = -3,
                 ):
        """Initialize a composite context from multiple subsystem contexts.

        :param contexts: List of contexts representing subsystems. The order matters as it
            defines the tensor product structure (first context is leftmost in tensor products).
        :param time_dimension: Dimension index for time broadcasting.

        Note: All subsystem contexts should be compatible in terms of:
        - Time dependence properties (all time-dependent or all stationary)
        - Data types and computational devices
        - Dimensional compatibility for tensor products
        """
        super().__init__(time_dimension=time_dimension)
        self.component_contexts = nn.ModuleList(contexts)

        self.transformation_probabilities = None
        self.transformation_unitary = None

        self._setup_prob_getters()
        self._setup_transformers()

    @property
    def basis(self) -> tp.Optional[torch.Tensor]:
        """Returns the composite basis matrix representing the tensor product of all subsystem bases.

        The composite basis is computed as the Kronecker product of individual subsystem bases:
            U_composite = U₁ ⊗ U₂ ⊗ ... ⊗ Uₙ

        Handling of None bases:
        - If ALL subsystems have None basis (eigenbasis): returns None
        - If ANY subsystem has a defined basis: None bases are replaced with identity matrices
          of appropriate dimension before computing the Kronecker product

        Batch dimensions:
        - All batch dimensions from subsystem bases are preserved and broadcasted
        - The resulting tensor has shape [B, N_total, N_total] where:
            * B represents the broadcasted batch dimensions
            * N_total = N₁ × N₂ × ... × Nₙ is the total Hilbert space dimension

        Physical interpretation:
        This basis defines the transformation from the product basis (tensor product of subsystem
        computational bases) to the working basis of the composite system. It enables consistent
        treatment of initial states, relaxation operators, and Hamiltonians across the composite space.
        """
        basis_values = [context.basis for context in self.component_contexts]
        batch_shapes = [ctx.basis.shape[:-2] for ctx in self.component_contexts]
        if all(b is None for b in basis_values):
            return None
        common_batch_shape = torch.Size(torch.broadcast_shapes(*batch_shapes))

        expanded_bases = []
        for basis in basis_values:
            if basis.dim() <= 2:
                expanded = basis.expand(common_batch_shape + basis.shape[-2:])
            else:
                expanded = basis.expand(common_batch_shape + basis.shape[-2:])
            expanded_bases.append(expanded)

        composite_basis = expanded_bases[0].clone()
        for basis in expanded_bases[1:]:
            composite_basis = transform.batched_kron(composite_basis, basis)

        return composite_basis

    @property
    def time_dependant(self):
        """Indicates whether the system parameters are time-dependent
        profile."""
        for context in self.component_contexts:
            if context.time_dependant:
                return True
        return False

    @property
    def contexted_init_population(self):
        """Indicates whether initial populations are provided via context."""
        if all([context.contexted_init_population for context in self.component_contexts]):
            return True
        else:
            return False

    @property
    def contexted_init_density(self):
        """Indicates whether the initial density matrix is taken from context
        rather than thermal equilibrium.

        Returns True if either `init_populations` or a user-defined initial density matrix is provided;
        otherwise False.
        """
        if all([context.contexted_init_density for context in self.component_contexts]):
            return True
        else:
            return False

    @property
    def device(self):
        """Context computation device"""
        return self.component_contexts[0].device

    @property
    def spin_system_dim(self):
        """Context spin system dimension"""
        size = 1
        for ctx in self.component_contexts:
            size *= ctx.spin_system_dim
        return size

    @property
    def dtype(self):
        """Context float dtype"""
        return self.component_contexts[0].dtype

    def close_context(self) -> None:
        """
        Reset Context cach parameters: basis coefficients
        """
        self.transformation_probabilities = None
        self.transformation_unitary = None
        for ctx in self.component_contexts:
            ctx.close_context()

    def __len__(self):
        """This is just entire context. Let's set its len as 1"""
        return 1

    def _compute_transformation_probabilities(self, full_system_vectors: tp.Optional[torch.Tensor]):
        """Compute Clebsch-Gordan transformation coefficients for composite
        system.

        :param full_system_vectors: Eigenvectors of the full composite Hamiltonian.
        :return: Transformation coefficients that can be used to transform vectors, matrices,
            and operators between bases.

        Mathematical formulation:
        If |α⟩ are basis states of subsystem 1 and |β⟩ of subsystem 2, and |γ⟩ are eigenstates
        of the composite system, then the transformation coefficients are:
            C_{γ,(α,β)} = ⟨γ|α,β⟩

        These coefficients are cached after computation to avoid redundant calculations.
        """
        if self.transformation_probabilities is not None:
            return self.transformation_probabilities
        else:
            bases = [context.basis for context in self.component_contexts]
            self.transformation_probabilities = transform.compute_clebsch_gordan_probabilities(
                full_system_vectors, bases
            )
            return self.transformation_probabilities

    def _compute_transformation_unitary(self, full_system_vectors: tp.Optional[torch.Tensor]):
        """Compute and cache superoperator transformation coefficients."""
        if self.transformation_unitary is not None:
            return self.transformation_unitary
        else:
            bases = [context.basis for context in self.component_contexts]
            self.transformation_unitary = transform.compute_clebsch_gordan_coeffs(full_system_vectors, bases)
            return self.transformation_unitary

    def get_time_dependent_values(self, time: torch.Tensor) -> tp.Optional[torch.Tensor]:
        for context in self.component_contexts:
            if context.profile is not None:
                return context.profile(time)[(...,) + (None,) * -(context.time_dimension+1)]

    def _check_callable(
            self, list_of_values: list[tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor], None]]):
        if all(callable(item) for item in list_of_values):
            return True
        elif all(not callable(item) for item in list_of_values):
            return False
        else:
            raise ValueError(
                "All elements of the union meaning \n"
                "(all free probs or all driven probs) must be either callable or not callable."
            )

    def _setup_single_getter(
            self, getter_lst: list[tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]]):
        if getter_lst:
            if self._check_callable(getter_lst):
                return lambda t: [
                    getter(t) for getter in getter_lst
                ]
            else:
                return lambda t: [
                    getter for getter in getter_lst
                ]
        else:
            return lambda t: None

    def _setup_transformers(self):
        self.transformed_vector = self._transformed_vector_basis
        self.transformed_populations = self._transformed_population_basis
        self.transformed_matrix = self._transformed_matrix_basis

        self.transformed_density = self._transformed_density_basis
        self.transformed_superop = self._transformed_superop_basis

    def _setup_prob_getters(self):
        """Setup getter methods for probabilities based on callable status at
        initialization."""
        current_free_probs_lst = [
            context.free_probs for context in self.component_contexts if context.free_probs is not None
        ]
        self._get_free_probs_tensor = self._setup_single_getter(current_free_probs_lst)

        current_driven_probs_lst = [
            context.driven_probs for context in self.component_contexts if context.driven_probs is not None
        ]
        self._get_driven_probs_tensor = self._setup_single_getter(current_driven_probs_lst)

        current_out_probs_lst = [
            context.out_probs for context in self.component_contexts if context.out_probs is not None
        ]
        self._get_out_probs_tensor = self._setup_single_getter(current_out_probs_lst)

        current_dephasing_lst = [
            context.dephasing for context in self.component_contexts if context.dephasing is not None
        ]
        self._get_dephasing_tensor = self._setup_single_getter(current_dephasing_lst)

        current_free_superop_lst = [
            context.free_superop for context in self.component_contexts if context.free_superop is not None
        ]
        self._get_free_superop_tensor = self._setup_single_getter(current_free_superop_lst)

        current_driven_superop_lst = [
            context.driven_superop for context in self.component_contexts if context.driven_superop is not None
        ]
        self._get_driven_superop_tensor = self._setup_single_getter(current_driven_superop_lst)

    def _transformed_skip(
            self, system_data: tp.Optional[torch.Tensor],
            full_system_vectors: tp.Optional[torch.Tensor]):
        return system_data

    def _transformed_population_basis(
            self, vector_lst: tp.Optional[list[torch.Tensor]], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform a population_lst from set of basis to one single basis."""
        if vector_lst is None:
            return None
        else:
            coeffs = self._compute_transformation_probabilities(full_system_vectors)
            return transform.transform_kronecker_populations(vector_lst, coeffs)

    def _transformed_vector_basis(
            self, vector_lst: tp.Optional[list[torch.Tensor]], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform a vector_lst from set of basis to one single basis."""
        if vector_lst is None:
            return None
        else:
            coeffs = self._compute_transformation_probabilities(full_system_vectors)
            return transform.transform_kronecker_rate_vector(vector_lst, coeffs)

    def _transformed_matrix_basis(
            self, matrix_lst: tp.Optional[list[torch.Tensor]], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform a matrix_lst from set of basis to one single basis."""
        if matrix_lst is None:
            return None
        else:
            coeffs = self._compute_transformation_probabilities(full_system_vectors)
            return transform.transform_kronecker_rate_matrix(matrix_lst, coeffs)

    def _transformed_density_basis(
            self, density_matrix_lst: tp.Optional[list[torch.Tensor]], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform density_matrix_lst from one basis to another."""
        if density_matrix_lst is None:
            return None
        else:
            coeffs = self._compute_transformation_unitary(full_system_vectors)
            return transform.transform_kronecker_operator(density_matrix_lst, coeffs)

    def _transformed_superop_basis(
            self, relaxation_superop_lst: tp.Optional[list[torch.Tensor]],
            full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform relaxation relaxation_superop_lst from one basis to another."""
        if relaxation_superop_lst is None:
            return None
        else:
            coeffs = self._compute_transformation_unitary(full_system_vectors)
            relaxation_superop_lst = [transform_to_complex(operator) for operator in relaxation_superop_lst]
            return transform.transform_kronecker_superoperator(relaxation_superop_lst, coeffs)

    def get_transformed_init_populations(self, full_system_vectors: tp.Optional[torch.Tensor], normalize: bool = False):
        """Return initial populations transformed into the field-dependent
        Hamiltonian eigenbasis.

        This method handles the critical transformation from the working basis (where initial
        populations are defined) to the eigenbasis of the field-dependent Hamiltonian (where
        dynamics are computed).

        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os `[...., M, N, N]`,
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
        forces the creator to compute these vectors
        The default behavior, whether to calculate vectors or not,
        depends on the specific Spectra Manager and its settings.

        Transformation rule:
        1) Firstly, the initial populations in kronecker basis is created:
        `n = n1 ⊗ n2 ⊗ n3 ... ⊗ n_k`

        2) Then the transformation to the field-dependent Hamiltonian eigenbasis. is performed
        If `|ψ_k^old>` are basis states in working basis and `|ψ_j^new>` in eigenbasis, then:
            `p_j^new = Σ_k |<ψ_j^new|ψ_k^old>|² · p_k^old`
        This ensures conservation of probability under basis change.

        :param normalize: If True the returned populations are normalized along the last axis
        so they sum to 1. If False, populations are returned
        as-is.
        :return: Initial populations with shape `[...N]`
        """
        populations = [
            context.init_populations for context in self.component_contexts if context.init_populations is not None
        ]
        if populations:
            probabilities = self._compute_transformation_probabilities(full_system_vectors)
            return transform.transform_kronecker_populations(populations, probabilities)
        else:
            return None

    def get_transformed_init_density(self, full_system_vectors: tp.Optional[torch.Tensor]):
        """Return initial density matrix transformed into the field-dependent
        eigenbasis.

        This method is used in the density-matrix paradigm where full quantum state evolution
        is computed, including coherences between energy levels.

        Physical interpretation:
        - Diagonal elements represent populations
        - Off-diagonal elements represent quantum coherences between states

        Transformation rule:
        1) Firstly, the initial populations in kronecker basis is created:
        `ρ = ρ1 ⊗ ρ2 ⊗ ρ3 ... ρ n_k`

        2) Then, if U is the unitary transformation matrix between bases `(U_{jk} = <ψ_j^new|ψ_k^old>)`,
            `ρ^new = U · ρ^old · U⁺`
        where U⁺ is the conjugate transpose of U.

        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os `[...., M, N, N]`,
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
        forces the creator to compute these vectors
        The default behavior, whether to calculate vectors or not,
        depends on the specific Spectra Manager and its settings.

        :return: Initial densities with shape `[...N, N]`
        """
        component_densities = []
        for context in self.component_contexts:
            if context.init_density is not None:
                component_densities.append(context.init_density)
            else:
                return None
        if not component_densities:
            return None
        coeff = self._compute_transformation_unitary(full_system_vectors)
        return transform.transform_kronecker_operator(component_densities, coeff)

    def __add__(self, other: BaseContext) -> SummedContext:
        """"""
        if isinstance(other, SummedContext):
            return SummedContext([self] + list(other.component_contexts))
        else:
            return SummedContext([self, other])

    def __matmul__(self, other: BaseContext) -> KroneckerContext:
        """"""
        if isinstance(other, SummedContext):
            return multiply_contexts([self, other])
        elif isinstance(other, KroneckerContext):
            _multiply_homogeneous_contexts([*self.component_contexts, *other.component_contexts])
        else:
            return _multiply_homogeneous_contexts([*self.component_contexts, other])


class SummedContext(BaseContext):
    """Context representing the sum of multiple relaxation mechanisms acting on
    the same system.

    This class models scenarios where multiple independent physical processes contribute to
    relaxation of a single quantum system. Examples include:

    Mathematical formulation:
    The total relaxation is described by the sum of individual contributions:
        `K_total = K₁ + K₂ + ... + Kₙ`   (for population dynamics)
        `R_total = R₁ + R₂ + ... + Rₙ` (for density matrix dynamics)

    Key properties:
    1. All component contexts must describe the same physical system (same Hilbert space)
    2. Each mechanism can be defined in its own basis and will be transformed to a common basis
    3. Time dependencies can differ between mechanisms

    This context type is essential when relaxation arises from multiple distinct physical
    processes that can be modeled separately but act simultaneously on the system.

    Summed contexts can be created using the + operator:
        summed_context = context1 + context2 + context3
    """
    def __init__(self, contexts: list[tp.Union[Context, KroneckerContext]]):
        """Initialize a summed context from multiple component contexts.

        :param contexts: List of contexts representing different relaxation mechanisms acting
            on the same quantum system.

        Note: All component contexts should be compatible in terms of:
        - Describing the same physical system (same number of energy levels)
        - Having compatible time dependence properties

        inner component_contexts is sorted in the order: KroneckerContext,
        contexts with not None basis and contexts with None basis

        """
        # Sort rule 1) KroneckerContext, 2) Context: 2.1) Basis is not None, 2.2) Contexted is not None
        super().__init__()
        sorted_context_list = sorted(contexts, key=self._get_context_priority)
        _num_contexts_with_basis = sum(
            1 for ctx in contexts if isinstance(ctx, KroneckerContext) or ctx.basis is not None)

        _num_contexts_with_populations = sum(
            1 for ctx in contexts if ctx.contexted_init_population)
        _num_contexts_with_density = sum(
            1 for ctx in contexts if ctx.contexted_init_density)

        self.component_contexts = nn.ModuleList(sorted_context_list)
        self._num_contexts_with_basis = _num_contexts_with_basis

        self._num_contexts_with_populations = _num_contexts_with_populations
        self._num_contexts_with_density = _num_contexts_with_density

    def _get_context_priority(self, context):
        """
        The function is used to sort contexts in the next order
        1) KroneckerContext
        2) Context with not-none basis and with population
        3) Context with not-none basis and without population
        4) Context with none basis and with population
        5) Context with none basis and without population

        :param context: The relaxation context
        :return:
        """
        is_kronecker = isinstance(context, KroneckerContext)
        has_basis = context.basis is not None
        has_population = hasattr(context, "population") and context.population is not None

        if is_kronecker:
            return 0
        elif has_basis and has_population:
            return 1
        elif has_basis and not has_population:
            return 2
        elif not has_basis and has_population:
            return 3
        else:
            return 4

    @property
    def num_contexts_with_basis(self) -> int:
        """
        Return the number of contexts with defined (not None) initial basis.
        For KroneckerContext it is assumed that basis is defined

        :return: The number of contexts with defined initial basis
        """
        return self._num_contexts_with_basis


    @property
    def num_contexts_with_populations(self) -> int:
        """
        Return the number of contexts with defined initial populations

        :return: The number of contexts with defined initial populations
        """
        return self._num_contexts_with_populations

    @property
    def num_contexts_with_density(self) -> int:
        """
        Return the number of contexts with defined initial density

        :return: The number of contexts with defined initial density
        """
        return self._num_contexts_with_density

    def get_time_dependent_values(self, time: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """Evaluate time-dependent profile at specified time points.

        :param time: Time points tensor for evaluation
        :return: Profile values shaped for broadcasting along the
            specified time dimension.
        """
        for context in self.component_contexts:
            if context.profile is not None:
                return context.profile(time)[(...,) + (None,) * -(context.time_dimension+1)]

    def close_context(self) -> None:
        """
        Reset Context cach parameters: basis coefficients
        """
        for ctx in self.component_contexts:
            ctx.close_context()

    def __len__(self):
        """This is just entire context. Let's set its len as 1"""
        return len(self.component_contexts)

    @property
    def time_dependant(self):
        """Indicates whether the system parameters are time-dependent
        profile."""
        for context in self.component_contexts:
            if context.time_dependant:
                return True
        return False

    @property
    def device(self):
        """Context computation device"""
        return self.component_contexts[0].device

    @property
    def spin_system_dim(self):
        """Context spin system dimension"""
        for ctx in self.component_contexts:
            if isinstance(ctx.spin_system_dim, int):
                return ctx.spin_system_dim
        raise NotImplementedError("spin_system_dim is equel to None")

    @property
    def dtype(self):
        """Context float dtype"""
        return self.component_contexts[0].dtype

    @property
    def contexted_init_population(self):
        """Indicates whether initial populations are provided via context."""
        if any([context.contexted_init_population for context in self.component_contexts]):
            return True
        else:
            return False

    @property
    def contexted_init_density(self):
        """Indicates whether the initial density matrix is taken from context
        rather than thermal equilibrium.

        Returns True if either `init_populations` or a user-defined initial density matrix is provided;
        otherwise False.
        """
        if any([context.contexted_init_density for context in self.component_contexts]):
            return True
        else:
            return False

    def get_transformed_init_populations(self, full_system_vectors: tp.Optional[torch.Tensor], normalize: bool = False):
        """
        :param full_system_vectors:

        Eigenvectors of the full set of energy levels. The shape os `[..., M, N, N]`,
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
        forces the creator to compute these vectors
        The default behavior, whether to calculate vectors or not,
        depends on the specific Spectra Manager and its settings.

        :param normalize: If True the returned populations are normalized along the last axis
        so they sum to 1. If False, populations are returned
        as-is.
        :return: Initial populations with shape `[..., N]`
        """
        result = None
        for context in self.component_contexts:
            populations = context.get_transformed_init_populations(full_system_vectors, False)
            if populations is not None:
                result = populations if result is None else result + populations
        return result

    def get_transformed_init_density(
            self, full_system_vectors: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        """
        :param full_system_vectors:

        Eigenvectors of the full set of energy levels. The shape os `[...., M, N, N]`,
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
        forces the creator to compute these vectors
        The default behavior, whether to calculate vectors or not,
        depends on the specific Spectra Manager and its settings.

        :return: density matrix  populations with shape `[... N, N]`
        """
        result = None
        for context in self.component_contexts:
            density = context.get_transformed_init_density(full_system_vectors)
            if density is not None:
                result = density if result is None else result + density
        return result

    def get_transformed_free_probs(
        self,
        full_system_vectors: tp.Optional[torch.Tensor],
        time_dep_values: tp.Optional[torch.Tensor] = None
    ):
        """
        :param full_system_vectors:

        Eigenvectors of the full set of energy levels. The shape os `[...., M, N, N]`,
        where M is number of transitions, N is number of levels
        The parameter of the creator 'output_eigenvector- == True'
        forces the creator to calculate these vectors
        The default behavior, whether to calculate vectors or not,
        depends on the specific Spectra Manager and its settings.

        :param time_dep_values: Optional time_dep_values for evaluation if loss rates are time-dependent.
        :return: torch.Tensor or None
            Transformed out probabilities shaped `[..., N]` or `[..., R, M, N]`.
        """
        result = None
        for context in self.component_contexts:
            probs = context.get_transformed_free_probs(full_system_vectors, time_dep_values)
            if probs is not None:
                result = probs if result is None else result + probs
        return result

    def get_transformed_driven_probs(
        self,
        full_system_vectors: tp.Optional[torch.Tensor],
        time_dep_values: tp.Optional[torch.Tensor] = None
    ):
        """
        :param full_system_vectors:

            Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
            where M is number of transitions, N is number of levels
            For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
            forces the creator to compute these vectors
        The default behavior, whether to calculate vectors or not,
        depends on the specific Spectra Manager and its settings.

        :param time_dep_values: the values computed at get_time_dependent_values
        :return: driven probability of transition.
        """
        result = None
        for context in self.component_contexts:
            probs = context.get_transformed_driven_probs(full_system_vectors, time_dep_values)
            if probs is not None:
                result = probs if result is None else result + probs
        return result

    def get_transformed_out_probs(
        self,
        full_system_vectors: tp.Optional[torch.Tensor],
        time_dep_values: tp.Optional[torch.Tensor] = None
    ):
        """
        :param full_system_vectors:

        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        The parameter of the creator 'output_eigenvector- == True'
        forces the creator to calculate these vectors
        The default behavior, whether to calculate vectors or not,
        depends on the specific Spectra Manager and its settings.

        :param time_dep_values: the values computed at get_time_dependent_values
        :return: torch.Tensor or None
            Transformed free probabilities shaped `[..., N, N]` or `[..., R, M, N, N]`.
        """
        result = None
        for context in self.component_contexts:
            probs = context.get_transformed_out_probs(full_system_vectors, time_dep_values)
            if probs is not None:
                result = probs if result is None else result + probs
        return result

    def get_transformed_free_superop(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ):
        """Return the spontaneous relaxation superoperator in Liouville spac.

        This method provides the complete Liouville-space superoperator for spontaneous
        relaxation processes, including thermal transitions, population losses, and dephasing.
        The superoperator is transformed to the eigenbasis and modified to obey detailed balance.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Pre-computed time-dependent profile values, if applicable.
        :return: Thermally corrected relaxation superoperator with shape [..., N², N²].

        Construction workflow:
        1. Build constituent operators:
           - Lindblad dissipators from thermal transition rates
           - Anticommutator terms from loss rates
           - Dephasing superoperators from dephasing rates
        2. Sum these contributions to form the raw superoperator
        3. Transform the superoperator to the eigenbasis using:
              R_new = (U ⊗ U*) · R_old · (U ⊗ U*)⁺
           where U is the basis transformation matrix and ⊗ denotes Kronecker product
        4. Apply thermal correction to population transfer elements:
              R_new_{iijj} = R_old_{iijj} · exp(-(E_i-E_j)/k_B·T) / (1 + exp(-(E_i-E_j)/k_B·T))
              R_new_{jjii} = R_old_{jjii} · 1 / (1 + exp(-(E_i-E_j)/k_B·T))
        """
        result = None
        for context in self.component_contexts:
            probs = context.get_transformed_free_superop(full_system_vectors, time_dep_values)
            if probs is not None:
                result = probs if result is None else result + probs
        return result

    def get_transformed_driven_superop(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ):
        """Return the spontaneous relaxation superoperator in Liouville spac.

        This method provides the complete Liouville-space superoperator for spontaneous
        relaxation processes, including thermal transitions, population losses, and dephasing.
        The superoperator is transformed to the eigenbasis and modified to obey detailed balance.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Pre-computed time-dependent profile values, if applicable.
        :return: Thermally corrected relaxation superoperator with shape [..., N², N²].

        Construction workflow:
        1. Build constituent operators:
           - Lindblad dissipators from thermal transition rates
           - Anticommutator terms from loss rates
           - Dephasing superoperators from dephasing rates
        2. Sum these contributions to form the raw superoperator
        3. Transform the superoperator to the eigenbasis using:
              R_new = (U ⊗ U*) · R_old · (U ⊗ U*)⁺
           where U is the basis transformation matrix and ⊗ denotes Kronecker product
        """
        result = None
        for context in self.component_contexts:
            probs = context.get_transformed_driven_superop(full_system_vectors, time_dep_values)
            if probs is not None:
                result = probs if result is None else result + probs
        return result

    def __add__(self, other: tp.Union[Context, KroneckerContext, SummedContext]) -> SummedContext:
        """"""
        if isinstance(other, SummedContext):
            return SummedContext(list(self.component_contexts) + list(other.component_contexts))
        else:
            return SummedContext(list(self.component_contexts) + [other])

    def __matmul__(self, other: tp.Union[Context, SummedContext]) -> tp.Union[SummedContext, Context]:
        """"""
        return multiply_contexts([self, other])
