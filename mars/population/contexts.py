from abc import ABC, abstractmethod

import torch
from torch import nn

import typing as tp

from .. import spin_system
from . import transform


def transform_to_complex(vector):
    if vector.dtype == torch.float32:
        return vector.to(torch.complex64)
    elif vector.dtype == torch.float64:
        return vector.to(torch.complex128)
    else:
        return vector


class BaseContext(nn.Module, ABC):
    """
    Abstract base class defining the interface for a spin-system "Context".

    A Context encapsulates the physical model of relaxation and initial state in time-resolved EPR.
    It specifies:
      - The basis in which relaxation parameters (transition probabilities, loss rates, etc.) are defined,
      - The initial population vector or density matrix,
      - Time-dependence profiles for any parameter
      - Transformation rules to map all quantities into the field-dependent Hamiltonian eigenbasis.

    MarS distinguishes two relaxation paradigms:
      1. **Population-based**: Only diagonal elements (populations) evolve; off-diagonal coherences are neglected.
      2. **Density-matrix-based**: Full dynamic including coherences and decoherence.
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
        """Indicates whether any relaxation parameter depends explicitly on time."""
        pass

    @property
    @abstractmethod
    def contexted_init_population(self) -> bool:
        """True if the Context provides explicit initial populations (not thermal equilibrium)."""
        pass

    @abstractmethod
    def get_time_dependent_values(self, time: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """
        Evaluate time-dependent profile at specified time points
        :param time: Time points tensor for evaluation
        :return: Profile values shaped for broadcasting along the specified time dimension.
        """
        pass

    @abstractmethod
    def get_transformed_init_populations(
            self, full_system_vectors: tp.Optional[torch.Tensor], normalize: bool = True
    ) -> tp.Optional[torch.Tensor]:
        """
        :param full_system_vectors:
        :param normalize: If True (default) the returned populations are normalized along the last axis
            so they sum to 1 (useful for probabilities). If False, populations are returned as-is.
        :return: Transformed populations with shape `[..., N]` (or `None` if no populations
            were provided).
        """
        pass

    @abstractmethod
    def get_transformed_init_density(
            self, full_system_vectors: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        """
        Return initial density matrix transformed into the Hamiltonian eigenbasis

        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to compute these vectors

        :return: density matrix  populations with shape [... N, N]
        """
        pass

    @abstractmethod
    def get_transformed_free_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ):
        """
        Return spontaneous (thermal) transition probabilities in the eigenbasis.

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
            W^new_{ij} = Σ_{k≠l} |<ψ_i^new|ψ_k^old>|² · |<ψ_j^new|ψ_l^old>|² · W^old_{kl}

        Thermal correction (detailed balance):
        After transformation, rates are modified to satisfy:
            W^new_{ij}/W^new_{ji} = exp(-(E_i - E_j)/k_B·T)
        where E_i are eigenenergies and T is the temperature.
        """
        pass

    @abstractmethod
    def get_transformed_driven_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ):
        """
        Return induced (non-thermal) transition probabilities in the eigenbasis.
        These transitions are **not** subject to detailed balance.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian at each orientation/field,
                        shape [..., M, N, N], where M is number of transitions, N is number of levels.
        :param time: Time points tensor for evaluation
        :return: Matrix of shape [..., N, N].
        """
        pass

    @abstractmethod
    def get_transformed_out_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ) -> tp.Optional[torch.Tensor]:
        """
        Return loss (out-of-system) probabilities in the eigenbasis.

        These represent irreversible decay processes that remove population from the spin
        system entirely. Examples include:
        - Phosphorescence decay from triplet states
        - Chemical reaction products leaving the observed spin system

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Optional time_dep_values for evaluation if loss rates are time-dependent.

        :return: Loss rate vector O with shape [..., N], where O_i is the rate at which
            population is lost from state i.

        Transformation rule:
            O^new_i = Σ_k |<ψ_i^new|ψ_k^old>|² · O^old_k

        Physical constraint: Loss rates must be non-negative (O_i ≥ 0).
        """
        pass

    @abstractmethod
    def get_transformed_free_superop(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None) -> tp.Optional[torch.Tensor]:
        """
        Return the spontaneous relaxation superoperator in Liouville space.

        This superoperator includes all spontaneous processes (thermal transitions, losses,
        decoherence) and is transformed to the eigenbasis. It is subsequently modified to
        obey detailed balance at the specified temperature.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Optional time_dep_values for evaluation if loss rates are time-dependent.

        :return: Superoperator R_free with shape [..., N², N²], where N is the number of
            energy levels. This superoperator acts on vectorized density matrices.

        Construction method:
        The superoperator is built using Lindblad formalism from the constituent processes:
        1. Loss terms: L_k = √O_k |k⟩⟨k|
        2. Thermal transitions: L_{kl} = √W_{kl} |k⟩⟨l|
        3. Decoherence terms: L_{kl} = √γ_{kl} |k⟩⟨l| for k≠l

        Transformation rule:
            R_new = (U ⊗ U*) · R_old · (U ⊗ U*)
            where U is the basis transformation matrix and ⊗ denotes Kronecker product

        Thermal correction:
        After transformation, diagonal elements corresponding to population transfer are
        modified to satisfy detailed balance:
            R_{iijj}^new = R_{iijj}^old · exp(-(E_i-E_j)/k_B·T) / (1 + exp(-(E_i-E_j)/k_B·T))
            R_{jjii}^new = R_{jjii}^old · 1 / (1 + exp(-(E_i-E_j)/k_B·T))

        where E_i are eigenenergies and T is temperature.
        """
        pass

    @abstractmethod
    def get_transformed_driven_superop(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ) -> tp.Optional[torch.Tensor]:
        """
        Return the induced relaxation superoperator in Liouville space.

        This superoperator contains only non-thermal (driven) processes that are NOT
        constrained by detailed balance. It is transformed to the eigenbasis without
        thermal correction.

        Construction method:
        Built using Lindblad formalism from induced transition rates:
            L_{kl} = √D_{kl} |k⟩⟨l|

        Transformation rule:
            R_new = (U ⊗ U*) · R_old · (U ⊗ U*)
            where U is the basis transformation matrix and ⊗ denotes Kronecker product

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Optional time_dep_values for evaluation if loss rates are time-dependent.
        :return: Superoperator R_driven with shape [..., N², N²].

        Unlike the free superoperator, NO thermal correction is applied to these elements,
        as they represent processes that actively drive the system away from equilibrium.

        Note: If a user provides a complete superoperator directly (bypassing individual
        rates), it is interpreted as an induced superoperator and no thermal correction
        is applied.
        """
        pass


class TransformedContext(BaseContext):
    """
    Concrete base class implementing basis transformation logic for Context subclasses.

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
        * "multiplet": Total spin multiplet basis |S, M⟩
        * "product": Product basis of individual spin projections |m_s1, m_s2, ...⟩

    The class automatically selects appropriate transformation methods based on basis type
    and caches transformation coefficients to avoid redundant computations.
    """
    def _setup_transformers(self):
        """
        Configure transformation methods based on the specified basis.

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
        """
        Return spontaneous (thermal) transition probabilities in the eigenbasis.

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
            W^new_{ij} = Σ_{k≠l} |<ψ_i^new|ψ_k^old>|² · |<ψ_j^new|ψ_l^old>|² · W^old_{kl}

        Thermal correction (detailed balance):
        After transformation, rates are modified to satisfy:
            W^new_{ij}/W^new_{ji} = exp(-(E_i - E_j)/k_B·T)
        where E_i are eigenenergies and T is the temperature.
        """
        _free_probs = self._get_free_probs_tensor(time_dep_values)
        return self.transformed_matrix(_free_probs, full_system_vectors)

    def get_transformed_driven_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time_dep_values: tp.Optional[torch.Tensor] = None
    ) -> tp.Optional[torch.Tensor]:
        """
        Return induced (non-thermal) transition probabilities in the eigenbasis.

        These transitions are NOT constrained by detailed balance and represent external
        driving forces or non-equilibrium processes.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time: Optional time points for evaluation if transition probabilities are
            time-dependent.

        :return: Transition rate matrix D with shape [..., N, N], where D_{ij} (i≠j) is the
            non-thermal rate from state j to state i.

        Transformation rule is the same as for free probabilities:
            D^new_{ij} = Σ_{k≠l} |<ψ_i^new|ψ_k^old>|² · |<ψ_j^new|ψ_l^old>|² · D^old_{kl}

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
        """
        Return loss (out-of-system) probabilities in the eigenbasis.

        These represent irreversible decay processes that remove population from the spin
        system entirely. Examples include:
        - Phosphorescence decay from triplet states
        - Chemical reaction products leaving the observed spin system

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Optional time_dep_values for evaluation if loss rates are time-dependent.

        :return: Loss rate vector O with shape [..., N], where O_i is the rate at which
            population is lost from state i.

        Transformation rule:
            O^new_i = Σ_k |<ψ_i^new|ψ_k^old>|² · O^old_k

        Physical constraint: Loss rates must be non-negative (O_i ≥ 0).
        """
        _out_probs = self._get_out_probs_tensor(time_dep_values)
        return self.transformed_vector(_out_probs, full_system_vectors)

    @property
    def free_superop(self):
        if self._default_free_superop is None:
            if (self.free_probs is None) and (self.out_probs is None) and (self.decoherences is None):
                return None
            return self._create_free_superop
        else:
            return self._default_free_superop

    @property
    def driven_superop(self):
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
        """
        Return the spontaneous relaxation superoperator in Liouville space.

        This superoperator includes all spontaneous processes (thermal transitions, losses,
        decoherence) and is transformed to the eigenbasis. It is subsequently modified to
        obey detailed balance at the specified temperature.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Optional time_dep_values for evaluation if loss rates are time-dependent.

        :return: Superoperator R_free with shape [..., N², N²], where N is the number of
            energy levels. This superoperator acts on vectorized density matrices.

        Construction method:
        The superoperator is built using Lindblad formalism from the constituent processes:
        1. Loss terms: L_k = √O_k |k⟩⟨k|
        2. Thermal transitions: L_{kl} = √W_{kl} |k⟩⟨l|
        3. Decoherence terms: L_{kl} = √γ_{kl} |k⟩⟨l| for k≠l

        Transformation rule:
            R_new = (U ⊗ U*) · R_old · (U ⊗ U*)
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
        """
        Return the induced relaxation superoperator in Liouville space.

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
            return self.liouvilleator.lindblad_dissipator_superop(_free_probs) + \
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
        if (self.free_probs is None) and (self.decoherences is None) and (self.out_probs is None):
            return None

        _density_condition = (self.out_probs is not None) or (self.free_probs is not None)
        if self.decoherences is not None and _density_condition:
            _decoherences = self._get_decoherences_tensor(time_dep_values)
            _relaxation_superop =\
                self.liouvilleator.lindblad_decoherences_superop(_decoherences) +\
                self._extract_free_populations_superop(time_dep_values)
            return _relaxation_superop

        elif (self.decoherences is None) and _density_condition:
            return self._extract_free_populations_superop(time_dep_values)

        else:
            _decoherences = self._get_decoherences_tensor(time_dep_values)
            _relaxation_superop = self.liouvilleator.lindblad_decoherences_superop(_decoherences)
            return _relaxation_superop


class Context(TransformedContext):
    """
    Primary implementation of a spin relaxation context for a sample

    This class provides a flexible interface for specifying relaxation models through:
    - Basis specification (explicit matrix or string identifier)
    - Initial state definition (populations or full density matrix)
    - Transition rate matrices (spontaneous, induced, loss terms)
    - Decoherence rates (for density-matrix paradigm)
    - Time-dependent profile functions

    Physical interpretation of parameters:
    - free_probs: Thermal (Boltzmann-weighted) transition rates between states
    - driven_probs: Non-thermal transition rates not constrained by detailed balance
    - out_probs: Irreversible loss rates from states (e.g., phosphorescence)
    - decoherences: Rates of quantum coherence decay between states
    - init_populations: Initial state populations in the working basis
    - init_density: Complete initial quantum state (includes coherences)

    The class supports both simple constant rates and time-dependent functions for all
    rate parameters, enabling modeling of complex time-evolving systems.

    Algebraic operations:
    - Addition (+): Combines multiple relaxation mechanisms acting on the SAME system
    - Tensor product (@): Combines independent subsystems into a composite quantum system

    These operations follow the physical rules described in the MarS documentation and
    enable construction of sophisticated relaxation models from simpler components.
    """

    def __init__(
            self,
            basis: tp.Optional[tp.Union[torch.Tensor, str, None]] = None,
            sample: tp.Optional[spin_system.MultiOrientedSample] = None,
            init_populations: tp.Optional[tp.Union[torch.Tensor, tp.List[float]]] = None,
            init_density: tp.Optional[torch.Tensor] = None,

            free_probs: tp.Optional[tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]] = None,
            driven_probs: tp.Optional[tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]] = None,
            out_probs: tp.Optional[
                tp.Union[torch.Tensor, tp.List[float], tp.Callable[[torch.Tensor], torch.Tensor]]] = None,

            decoherences: tp.Optional[tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]] = None,
            relaxation_superop: tp.Optional[tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]] = None,

            profile: tp.Optional[tp.Callable[[torch.Tensor], torch.Tensor]] = None,
            time_dimension: int = -3,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cpu")
    ):
        """
        :param basis: torch.Tensor or str or None, optional
        Basis specifier. Three allowed forms:
          -`str`: one of {"zfs", "multiplet", "product", "eigen"}. If a string is
            given, `sample` **must** be provided so the basis can be constructed.
            * "zfs"       : eigenvectors of the zero-field Hamiltonian (unsqueezed)
            * "multiplet" : total spin multiplet basis |S, M⟩
            * "product"   : computational product basis |ms1, ms2, ..., is1, ...⟩
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
            If provided, it will be converted to a `torch.tensor` and optionally
            normalized by `get_transformed_init_populations`.

        :param init_density: torch.Tensor, optional.
            Initial density of the spin system. Shape [..., N, N]
            If provided then init_populations will be ignored and populations will be computed as
            diagonal elements of init_density (as it needed)

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

        :param decoherences: torch.Tensor or callable or None, optional
            decoherences relaxation rates with shape [N].
            Each element set the Decreasing of the non-diagonal matrix elements of density matrix
            d <i|rho|j> / dt = -(decoherences[i] + decoherences[j]) / 2 * <i|rho|j>
            If relaxation_superop is given, then this parameter is ignored
        For implementation of decoherences, out_probs, driven_probs, free_probs we use Lindblad form of relaxation.

        :param relaxation_superop: torch.Tensor or callable or None, optional
            Full superoperator of relaxation rates for density matrix
            with shape [N*N, N*N]. Any elements can be given.
            If it is given then  driven_probs are ignored
            but free_probs, decoherences, out_probs are computed
            After transformation the thermal correction is not used for this term

        :param profile: callable or None, optional
            Callable `profile(time: torch.Tensor) -> torch.Tensor` that returns
            time-dependent scalars/arrays used by `get_time_dependent_values`.
            If None, `get_time_dependent_values` will raise if called.

        :param time_dimension: int, optional
            Axis index where time should be broadcasted in returned tensors.
            Default -5 to match the code's broadcasting conventions.
        """
        super().__init__(time_dimension=time_dimension, dtype=dtype, device=device)
        self.transformation_basis_coeff = None
        self.transformation_superop_coeff = None

        self.init_populations = self._set_init_populations(init_populations, init_density, dtype, device)
        init_density_real, init_density_imag = self._set_init_density(init_density)
        self.register_buffer("_init_density_real", init_density_real)
        self.register_buffer("_init_density_imag", init_density_imag)

        self.out_probs = out_probs
        self.free_probs = free_probs
        self.driven_probs = driven_probs

        self.decoherences = self._set_init_decoherences(decoherences, relaxation_superop)
        self._default_free_superop = None
        self._default_driven_superop = relaxation_superop

        self.profile = profile

        if isinstance(basis, str):
            if sample is None:
                raise ValueError("Sample must be provided when basis is specified as a string method")
            self.basis = self._create_basis_from_string(basis, sample)
        elif isinstance(basis, torch.Tensor):
            if basis.shape[-1] != basis.shape[-2]:
                raise ValueError("Basis tensor must be square (last two dimensions must match)")
            self.basis = basis
        else:
            self.basis = basis
        self._setup_prob_getters()
        self._setup_transformers()

    @property
    def time_dependant(self):
        return self.profile is not None

    @property
    def contexted_init_population(self):
        return self.init_populations is not None

    @property
    def contexted_init_density(self):
        return (self.init_populations is not None) or (self._init_density_real is not None)

    @property
    def init_density(self):
        if self._init_density_real is None:
            if self.init_populations is None:
                return None
            self._init_density_real = torch.diag_embed(self.init_populations, dim1=-1, dim2=-2)
            self._init_density_imag = torch.zeros_like(self._init_density_real)
        return torch.complex(self._init_density_real, self._init_density_imag)

    def _set_init_density(self, init_density: tp.Optional[torch.Tensor]):
        if init_density is None:
            return None, None
        else:
            return init_density.real, init_density.imag

    def _set_init_populations(self,
                              init_populations: tp.Optional[tp.Union[torch.Tensor, list[float]]],
                              init_density: tp.Optional[torch.Tensor],
                              dtype: torch.dtype, device: torch.device):
        if init_density is None:
            if init_populations is None:
                return None
            elif init_populations is not None:
                return torch.tensor(init_populations, dtype=dtype, device=device)
        else:
            return torch.diagonal(init_density, dim1=-1, dim2=-2)

    def _set_init_decoherences(self,
                             decoherences: tp.Optional[torch.Tensor],
                             relaxation_superop: tp.Optional[torch.Tensor])\
            -> tp.Optional[tp.Union[tp.Callable[[torch.Tensor], torch.Tensor], torch.Tensor]]:
        if relaxation_superop is None:
            return decoherences
        else:
            return None

    def _compute_transformation_basis_coeff(self, full_system_vectors: tp.Optional[torch.Tensor]):
        """Compute and cache basis transformation coefficients."""
        if self.transformation_basis_coeff is not None:
            return self.transformation_basis_coeff
        else:
            self.transformation_basis_coeff = transform.get_transformation_coeffs(
                self.basis, full_system_vectors
            )
            return self.transformation_basis_coeff

    def _compute_transformation_density_coeff(self, full_system_vectors: tp.Optional[torch.Tensor]):
        """Compute and cache basis transformation coefficients."""
        if self.transformation_basis_coeff is not None:
            return self.transformation_basis_coeff
        else:
            self.transformation_basis_coeff = transform.basis_transformation(
                self.basis, full_system_vectors
            )
            return self.transformation_basis_coeff

    def _compute_transformation_superop_coeff(self, full_system_vectors: tp.Optional[torch.Tensor]):
        """Compute and cache basis transformation coefficients."""
        if self.transformation_superop_coeff is not None:
            return self.transformation_superop_coeff
        else:
            self.transformation_superop_coeff = transform.compute_liouville_basis_transformation(
                self.basis, full_system_vectors
            )
            return self.transformation_superop_coeff

    def _transformed_skip(
            self, system_data: tp.Optional[torch.Tensor],
            full_system_vectors: tp.Optional[torch.Tensor]):
        return system_data

    def _transformed_vector_basis(
            self, vector: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform a vector from one basis to another."""
        if vector is None:
            return None
        else:
            coeffs = self._compute_transformation_basis_coeff(full_system_vectors)
            return transform.transform_vector_to_new_basis(vector, coeffs)

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
            coeffs = self._compute_transformation_basis_coeff(full_system_vectors)
            return transform.transform_matrix_to_new_basis(matrix, coeffs)

    def _transformed_density_basis(
            self, density_matrix: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform density matrix from one basis to another."""
        if density_matrix is None:
            return None
        else:
            coeffs = self._compute_transformation_density_coeff(full_system_vectors)
            return transform.transform_density(density_matrix, coeffs)

    def _transformed_superop_basis(
            self, relaxation_superop: tp.Optional[torch.Tensor], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform relaxation superoperator from one basis to another."""
        if relaxation_superop is None:
            return None
        else:
            coeffs = self._compute_transformation_superop_coeff(full_system_vectors)
            return transform.transform_liouville_superop(transform_to_complex(relaxation_superop), coeffs)

    def _create_basis_from_string(self, basis_type: str, sample: tp.Optional[spin_system.MultiOrientedSample]):
        """Factory method to create basis from string identifier."""
        if basis_type == "zfs":
            zero_field_term = sample.build_zero_field_term()
            _, zfs_eigenvectors = torch.linalg.eigh(zero_field_term)
            return zfs_eigenvectors.unsqueeze(-3)
        elif basis_type == "multiplet":
            return sample.base_spin_system.get_spin_multiplet_basis()\
                .unsqueeze(-3).unsqueeze(-4).to(sample.complex_dtype)
        elif basis_type == "product":
            return sample.base_spin_system.get_product_state_basis()\
                .unsqueeze(-3).unsqueeze(-4).to(sample.complex_dtype)
        elif basis_type == "eigen":
            return None
        else:
            raise KeyError(
                "Basis must be one of:\n"
                "1) torch.Tensor with shape [R, N, N] or [N, N], where R is number of orientations\n"
                "2) str: 'zfs', 'multiplet', 'product', 'eigen'\n"
                "3) None (will use eigen basis at given magnetic fields)"
            )

    def _setup_single_getter(
            self, getter: tp.Optional[tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]]):
        if callable(getter):
            return lambda t: getter(t)
        else:
            return lambda t: getter

    def _setup_prob_getters(self):
        """Setup getter methods for probabilities based on callable status at initialization."""
        current_free_probs = self.free_probs
        self._get_free_probs_tensor = self._setup_single_getter(current_free_probs)

        current_driven_probs = self.driven_probs
        self._get_driven_probs_tensor = self._setup_single_getter(current_driven_probs)

        current_out_probs = self.out_probs
        self._get_out_probs_tensor = self._setup_single_getter(current_out_probs)

        current_decoherences = self.decoherences
        self._get_decoherences_tensor = self._setup_single_getter(current_decoherences)

        current_free_superop = self.free_superop
        self._get_free_superop_tensor = self._setup_single_getter(current_free_superop)

        current_driven_superop = self.driven_superop
        self._get_driven_superop_tensor = self._setup_single_getter(current_driven_superop)

    def get_time_dependent_values(self, time: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """
        Evaluate time-dependent profile at specified time points
        Evaluate time-dependent values at specified time points.
        :param time: Time points tensor for evaluation
        :return: Profile values shaped for broadcasting along the specified time dimension.
        """
        return self.profile(time)[(...,) + (None,) * (-(self.time_dimension+1))]

    def get_transformed_init_populations(
            self, full_system_vectors: tp.Optional[torch.Tensor], normalize: bool = True
    ) -> tp.Optional[torch.Tensor]:
        """
        Return initial populations transformed into the field-dependent Hamiltonian eigenbasis.
        This method handles the critical transformation from the working basis (where initial
        populations are defined) to the eigenbasis of the field-dependent Hamiltonian (where
        dynamics are computed).

        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to compute these vectors

        Transformation rule:
        If |ψ_k^old> are basis states in working basis and |ψ_j^new> in eigenbasis, then:
            p_j^new = Σ_k |<ψ_j^new|ψ_k^old>|² · p_k^old
        This ensures conservation of probability under basis change.

        :param normalize: If True (default) the returned populations are normalized along the last axis
        so they sum to 1 (useful for probabilities). If False, populations are returned
        as-is.
        :return: Initial populations with shape [...N]
        """
        populations = self.transformed_populations(self.init_populations, full_system_vectors)
        if normalize and (populations is not None):
            return populations / torch.sum(populations, dim=-1, keepdim=True)
        else:
            return populations

    def get_transformed_init_density(
            self, full_system_vectors: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        """
        Return initial density matrix transformed into the field-dependent eigenbasis.
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
        For some cases it can be None. The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to compute these vectors

        :return: Initial densities with shape [...N, N]
        """
        return self.transformed_density(self.init_density, full_system_vectors)

    def __add__(self, other: BaseContext):
        """
        """
        if isinstance(other, SummedContext):
            return SummedContext([self] + list(other.component_contexts))
        else:
            return SummedContext([self, other])

    def __matmul__(self, other: BaseContext):
        """
        """
        if isinstance(other, SummedContext):
            raise NotImplementedError("multiplication with SummedContext is not implemented.")
        elif isinstance(other, CompositeContext):
            CompositeContext([self, *other.component_contexts], time_dimension=self.time_dimension)
        else:
            return CompositeContext([self, other], time_dimension=self.time_dimension)


class CompositeContext(TransformedContext):
    """
    Context representing a composite quantum system formed by tensor product of subsystems.

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
       identity operators.

    Transformation rules:
    - Basis transformations use Clebsch-Gordan coefficients to map between product bases
      and coupled bases
    - Initial populations are transformed using tensor products of transformation matrices
    - Relaxation superoperators are transformed using Kronecker products of basis
      transformation matrices

    Composite contexts can be created using the @ operator:
        composite_context = context1 @ context2 @ context3
    """
    def __init__(self,
                 contexts: list[TransformedContext],
                 time_dimension: int = -3,
                 ):
        """
        Initialize a composite context from multiple subsystem contexts.

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
        self.transformation_basis_coeff = None
        self._setup_prob_getters()
        self._setup_transformers()

    @property
    def time_dependant(self):
        for context in self.component_contexts:
            if context.profile is not None:
                return True
        return False

    @property
    def contexted_init_population(self):
        if [None for context in self.component_contexts if context.contexted_init_population is not None]:
            return True
        else:
            return False

    @property
    def contexted_init_density(self):
        if [context.init_populations for context in self.component_contexts if context.contexted_init_desnity]:
            return True
        else:
            return False

    def _compute_transformation_basis_coeff(self, full_system_vectors: tp.Optional[torch.Tensor]):
        """
        Compute Clebsch-Gordan transformation coefficients for composite system.

        :param full_system_vectors: Eigenvectors of the full composite Hamiltonian.
        :return: Transformation coefficients that can be used to transform vectors, matrices,
            and operators between bases.

        Mathematical formulation:
        If |α⟩ are basis states of subsystem 1 and |β⟩ of subsystem 2, and |γ⟩ are eigenstates
        of the composite system, then the transformation coefficients are:
            C_{γ,(α,β)} = ⟨γ|α,β⟩

        These coefficients are cached after computation to avoid redundant calculations.
        """
        if self.transformation_basis_coeff is not None:
            return self.transformation_basis_coeff
        else:
            basises = [context.basis for context in self.component_contexts]
            self.transformation_basis_coeff = transform.compute_clebsch_gordan_probabilities(full_system_vectors, basises)
            return self.transformation_basis_coeff

    def _compute_transformation_superop_coeff(self, full_system_vectors: tp.Optional[torch.Tensor]):
        """Compute and cache superoperator transformation coefficients."""
        if self.transformation_basis_coeff is not None:
            return self.transformation_basis_coeff
        else:
            basises = [context.basis for context in self.component_contexts]
            self.transformation_basis_coeff = transform.compute_clebsch_gordan_coeffs(full_system_vectors, basises)
            return self.transformation_basis_coeff

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
        """Setup getter methods for probabilities based on callable status at initialization."""
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

        current_decoherences_lst = [
            context.decoherences for context in self.component_contexts if context.decoherences is not None
        ]
        self._get_decoherences_tensor = self._setup_single_getter(current_decoherences_lst)

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
            coeffs = self._compute_transformation_basis_coeff(full_system_vectors)
            return transform.transform_kronecker_populations(vector_lst, coeffs)

    def _transformed_vector_basis(
            self, vector_lst: tp.Optional[list[torch.Tensor]], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform a vector_lst from set of basis to one single basis."""
        if vector_lst is None:
            return None
        else:
            coeffs = self._compute_transformation_basis_coeff(full_system_vectors)
            return transform.transform_kronecker_vectors(vector_lst, coeffs)

    def _transformed_matrix_basis(
            self, matrix_lst: tp.Optional[list[torch.Tensor]], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform a matrix_lst from set of basis to one single basis."""
        if matrix_lst is None:
            return None
        else:
            coeffs = self._compute_transformation_basis_coeff(full_system_vectors)
            return transform.transform_kronecker_matrix(matrix_lst, coeffs)

    def _transformed_density_basis(
            self, density_matrix_lst: tp.Optional[list[torch.Tensor]], full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform density_matrix_lst from one basis to another."""
        if density_matrix_lst is None:
            return None
        else:
            coeffs = self._compute_transformation_superop_coeff(full_system_vectors)
            return transform.transform_kronecker_density(density_matrix_lst, coeffs)

    def _transformed_superop_basis(
            self, relaxation_superop_lst: tp.Optional[list[torch.Tensor]],
            full_system_vectors: tp.Optional[torch.Tensor]
    ):
        """Transform relaxation superoperator from one basis to another."""
        if relaxation_superop_lst is None:
            return None
        else:
            coeffs = self._compute_transformation_superop_coeff(full_system_vectors)
            return transform.transform_kronecker_superoperator(relaxation_superop_lst, coeffs)

    def get_transformed_init_populations(self, full_system_vectors: tp.Optional[torch.Tensor], normalize: bool = True):
        """
        Return initial populations transformed into the field-dependent Hamiltonian eigenbasis.
        This method handles the critical transformation from the working basis (where initial
        populations are defined) to the eigenbasis of the field-dependent Hamiltonian (where
        dynamics are computed).

        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to compute these vectors

        Transformation rule:
        1) Firstly, the initial populations in kronecker basis is created:
        n = n1 ⊗ n2 ⊗ n3 ... ⊗ n_k

        2) Then the transformation to the field-dependent Hamiltonian eigenbasis. is performed
        If |ψ_k^old> are basis states in working basis and |ψ_j^new> in eigenbasis, then:
            p_j^new = Σ_k |<ψ_j^new|ψ_k^old>|² · p_k^old
        This ensures conservation of probability under basis change.

        :param normalize: If True (default) the returned populations are normalized along the last axis
        so they sum to 1 (useful for probabilities). If False, populations are returned
        as-is.
        :return: Initial populations with shape [...N]
        """
        populations = [
            context.init_populations for context in self.component_contexts if context.init_populations is not None
        ]
        if populations:
            _transformation_basis_coeff = self._compute_transformation_basis_coeff(full_system_vectors)
            return transform.transform_kronecker_populations(populations, _transformation_basis_coeff)
        else:
            return None

    def get_transformed_init_density(self, full_system_vectors: tp.Optional[torch.Tensor]):
        """
        Return initial density matrix transformed into the field-dependent eigenbasis.
        This method is used in the density-matrix paradigm where full quantum state evolution
        is computed, including coherences between energy levels.


        Physical interpretation:
        - Diagonal elements represent populations
        - Off-diagonal elements represent quantum coherences between states

        Transformation rule:
        1) Firstly, the initial populations in kronecker basis is created:
        ρ = ρ1 ⊗ ρ2 ⊗ ρ3 ... ρ n_k

        2) Then, if U is the unitary transformation matrix between bases (U_{jk} = <ψ_j^new|ψ_k^old>),
            ρ^new = U · ρ^old · U⁺
        where U⁺ is the conjugate transpose of U.

        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to compute these vectors

        :return: Initial densities with shape [...N, N]
        """
        component_densities = []
        for context in self.component_contexts:
            if context.init_density is not None:
                component_densities.append(context.init_density)
            else:
                return None
        if not component_densities:
            return None
        _transformation_basis_coeff = self._compute_transformation_superop_coeff(full_system_vectors)
        return transform.transform_kronecker_density(component_densities, _transformation_basis_coeff)

    def __matmul__(self, other: BaseContext):
        """
        """
        if isinstance(other, SummedContext):
            raise NotImplementedError("multiplication with SummedContext is not implemented.")
        elif isinstance(other, CompositeContext):
            CompositeContext([*self.component_contexts, *other.component_contexts], time_dimension=self.time_dimension)
        else:
            return CompositeContext([*self.component_contexts, other], time_dimension=self.time_dimension)


class SummedContext(BaseContext):
    """
    Context representing the sum of multiple relaxation mechanisms acting on the same system.

    This class models scenarios where multiple independent physical processes contribute to
    relaxation of a single quantum system. Examples include:

    Mathematical formulation:
    The total relaxation is described by the sum of individual contributions:
        K_total = K₁ + K₂ + ... + Kₙ    (for population dynamics)
        R_total = R₁ + R₂ + ... + Rₙ    (for density matrix dynamics)

    Key properties:
    1. All component contexts must describe the same physical system (same Hilbert space)
    2. Each mechanism can be defined in its own basis and will be transformed to a common basis
    3. Time dependencies can differ between mechanisms

    This context type is essential when relaxation arises from multiple distinct physical
    processes that can be modeled separately but act simultaneously on the system.

    Summed contexts can be created using the + operator:
        summed_context = context1 + context2 + context3
    """
    def __init__(self, contexts: list[BaseContext]):
        """
        Initialize a summed context from multiple component contexts.

        :param contexts: List of contexts representing different relaxation mechanisms acting
            on the same quantum system.

        Note: All component contexts should be compatible in terms of:
        - Describing the same physical system (same number of energy levels)
        - Having compatible time dependence properties
        """
        super().__init__()
        self.component_contexts = nn.ModuleList(contexts)

    def get_time_dependent_values(self, time: torch.Tensor) -> tp.Optional[torch.Tensor]:
        for context in self.component_contexts:
            if context.profile is not None:
                return context.profile(time)[(...,) + (None,) * -(context.time_dimension+1)]

    @property
    def time_dependant(self):
        for context in self.component_contexts:
            if context.profile is not None:
                return True
        return False

    @property
    def contexted_init_population(self):
        if [None for context in self.component_contexts if context.init_populations is not None]:
            return True
        else:
            return False

    @property
    def contexted_init_density(self):
        if [context.init_populations for context in self.component_contexts if context.contexted_init_desnity]:
            return True
        else:
            return False

    def get_transformed_init_populations(self, full_system_vectors: tp.Optional[torch.Tensor], normalize: bool = True):
        """
        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to compute these vectors

        :param normalize: If True (default) the returned populations are normalized along the last axis
        so they sum to 1 (useful for probabilities). If False, populations are returned
        as-is.
        :return: Initial populations with shape [...N]
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
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to compute these vectors

        :return: density matrix  populations with shape [... N, N]
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
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to calculate these vectors

        :param time_dep_values:
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
            For some cases it can be None. The parameter of the creator 'full_system_vectors_flag == True'
            forces the creator to compute these vectors

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
        The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to calculate these vectors

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
        """
        Return the spontaneous relaxation superoperator in Liouville spac

        This method provides the complete Liouville-space superoperator for spontaneous
        relaxation processes, including thermal transitions, population losses, and decoherence.
        The superoperator is transformed to the eigenbasis and modified to obey detailed balance.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Pre-computed time-dependent profile values, if applicable.
        :return: Thermally corrected relaxation superoperator with shape [..., N², N²].

        Construction workflow:
        1. Build constituent operators:
           - Lindblad dissipators from thermal transition rates
           - Anticommutator terms from loss rates
           - Decoherence superoperators from decoherence rates
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
        """
        Return the spontaneous relaxation superoperator in Liouville spac

        This method provides the complete Liouville-space superoperator for spontaneous
        relaxation processes, including thermal transitions, population losses, and decoherence.
        The superoperator is transformed to the eigenbasis and modified to obey detailed balance.

        :param full_system_vectors: Eigenvectors of the full Hamiltonian.
        :param time_dep_values: Pre-computed time-dependent profile values, if applicable.
        :return: Thermally corrected relaxation superoperator with shape [..., N², N²].

        Construction workflow:
        1. Build constituent operators:
           - Lindblad dissipators from thermal transition rates
           - Anticommutator terms from loss rates
           - Decoherence superoperators from decoherence rates
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

    def __add__(self, other: BaseContext):
        """
        """
        if isinstance(other, SummedContext):
            return SummedContext(list(self.component_contexts) + list(other.component_contexts))
        else:
            return SummedContext(list(self.component_contexts) + [other])

    def __matmul__(self, other: BaseContext):
        """
        """
        raise NotImplementedError("multiplication with SummedContext is not implemented.")