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
    Abstract base class defining how a spin-system 'context' exposes
    state transformations and transition probabilities.

    A "context" encapsulates:
      - a basis or basis-selection rule for transforming vectors/matrices
        between the full-system basis and the working basis,
      - the set of transition probabilities (free/driven/out-of-system),
      - an initial population vector,
      - an optional time-dependence profile for parameters.

    Concrete subclasses must implement methods that return transformed
    objects suitable for use by higher-level code that computes dynamics."""
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
    def time_dependant(self):
        pass

    @property
    @abstractmethod
    def contexted_init_population(self):
        pass

    @abstractmethod
    def get_time_dependent_values(self, time: torch.Tensor) -> torch.Tensor | None:
        """
        Evaluate time-dependent profile at specified time points
        Evaluate time-dependent values at specified time points.
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
            so they sum to 1 (useful for probabilities). If False, populations are returned
            as-is.
        :return: Transformed populations with shape `[..., N]` (or `None` if no populations
            were provided).
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_transformed_free_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time: tp.Optional[torch.Tensor] = None
    ):
        """
        :param full_system_vectors:
        :param time:
        :return:
        """
        pass

    @abstractmethod
    def get_transformed_driven_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time: tp.Optional[torch.Tensor] = None
    ):
        pass

    @abstractmethod
    def get_transformed_out_probs(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time: tp.Optional[torch.Tensor] = None
    ):
        pass

    @abstractmethod
    def get_transformed_free_superop(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time: tp.Optional[torch.Tensor] = None
    ):
        pass

    @abstractmethod
    def get_transformed_driven_superop(
            self,
            full_system_vectors: tp.Optional[torch.Tensor],
            time: tp.Optional[torch.Tensor] = None
    ):
        pass

class TransformedContext(BaseContext):
    def _setup_transformers(self):
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
    ):
        """
        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to calculate these vectors

        :param time_dep_values:
        :return: torch.Tensor or None
            Transformed free probabilities shaped `[..., N, N]` or `[..., R, M, N, N]`.
        """
        _free_probs = self._get_free_probs_tensor(time_dep_values)
        return self.transformed_matrix(_free_probs, full_system_vectors)

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
        _driven_probs = self._get_driven_probs_tensor(time_dep_values)
        return self.transformed_matrix(_driven_probs, full_system_vectors)

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
            Transformed out probabilities shaped `[..., N]` or `[..., R, M, N]`.
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
    ):
        """
        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to compute these vectors

        :param time_dep_values: the values computed at get_time_dependent_values
        :return: relaxation superoperator with shape [... N^2, N^2].
        After all transofrmation the next rule is applied to superoperator:
        Riijj_new = Riijj * exp(-(Ei - Ej)) / 1 + exp(-(Ei - Ej))
        Rjjii_new = Rjjii * 1 / 1 + exp(Ei - Ej)
        """
        _relaxation_superop = self._get_free_superop_tensor(time_dep_values)
        return self.transformed_superop(_relaxation_superop, full_system_vectors)

    def get_transformed_driven_superop(
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
        :return: relaxation superoperator with shape [... N^2, N^2].
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
    def __init__(
            self,
            basis: tp.Optional[torch.Tensor | str | None] = None,
            sample: tp.Optional[spin_system.MultiOrientedSample] = None,
            init_populations: tp.Optional[torch.Tensor | list[float]] = None,
            init_density: tp.Optional[torch.Tensor] = None,

            free_probs: tp.Optional[torch.Tensor | tp.Callable[[torch.Tensor], torch.Tensor]] = None,
            driven_probs: tp.Optional[torch.Tensor | tp.Callable[[torch.Tensor], torch.Tensor]] = None,
            out_probs: tp.Optional[torch.Tensor | list[float] | tp.Callable[[torch.Tensor], torch.Tensor]] = None,

            decoherences: tp.Optional[torch.Tensor | tp.Callable[[torch.Tensor], torch.Tensor]] = None,
            relaxation_superop: tp.Optional[torch.Tensor | tp.Callable[[torch.Tensor], torch.Tensor]] = None,

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
            Each element set the INCREASING of the non-diagonal matrix elements of density matrix
            d <i|rho|j> / dt = -(decoherences[i] + decoherences[j]) / 2 * <i|rho|j>
            If relaxation_superop is given, then this parameter is ignored

        For implementation of decoherences, out_probs, driven_probs, free_probs we use Lindblad form of relaxation.

        :param relaxation_superop: torch.Tensor or callable or None, optional
            Full superoperator of relaxation rates for density matrix
            with shape [N*N, N*N]. Any elements can be given.
            If it is given then decoherences, out_probs, driven_probs, free_probs are ignored
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

    def _create_basis_from_string(self, basis_type: str, sample):
        """Factory method to create basis from string identifier."""
        if basis_type == "zfs":
            zero_field_term = sample.build_zero_field_term()
            _, zfs_eigenvectors = torch.linalg.eigh(zero_field_term)
            return zfs_eigenvectors.unsqueeze(-3)
        elif basis_type == "multiplet":
            return sample.base_spin_system.get_spin_multiplet_basis().unsqueeze(-3).unsqueeze(-4)
        elif basis_type == "product":
            return sample.base_spin_system.get_product_state_basis().unsqueeze(-3).unsqueeze(-4)
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

    def get_time_dependent_values(self, time: torch.Tensor) -> torch.Tensor | None:
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
        populations = self.transformed_populations(self.init_populations, full_system_vectors)
        if normalize and (populations is not None):
            return populations / torch.sum(populations, dim=-1, keepdim=True)
        else:
            return populations

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
    def __init__(self,
                 contexts: list[TransformedContext],
                 time_dimension: int = -3,
                 ):
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
        """Compute and cache basis transformation coefficients."""
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

    def get_time_dependent_values(self, time: torch.Tensor) -> torch.Tensor | None:
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
    def __init__(self, contexts: list[BaseContext]):
        super().__init__()
        self.component_contexts = nn.ModuleList(contexts)

    def get_time_dependent_values(self, time: torch.Tensor) -> torch.Tensor | None:
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
        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to compute these vectors

        :param time_dep_values: the values computed at get_time_dependent_values

        :return: relaxation superoperator with shape [... N^2, N^2].
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
        :param full_system_vectors:
        Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
        where M is number of transitions, N is number of levels
        For some cases it can be None. The parameter of the creator 'full_system_vectors_flag == True'
        forces the creator to compute these vectors

        :param time_dep_values: the values computed at get_time_dependent_values

        :return: relaxation superoperator with shape [... N^2, N^2].
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