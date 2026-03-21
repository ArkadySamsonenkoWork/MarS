import torch

import typing as tp
from abc import ABC, abstractmethod
import math
from enum import Enum

from . import transform
from .. import constants


OperatorComponentsType = tp.List[tp.Tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]]


def _expand_op_kronecker(op: torch.Tensor, left_dim: int, right_dim: int) -> torch.Tensor:
    """Helper to expand a single operator."""
    N = op.shape[-1]
    L, R = left_dim, right_dim
    device, dtype = op.device, op.dtype

    I_left = torch.eye(L, device=device, dtype=dtype) if L > 0 else None
    I_right = torch.eye(R, device=device, dtype=dtype) if R > 0 else None

    if L == 0:
        result = torch.einsum("...ij,kl->...ikjl", op, I_right)
        result = result.reshape(op.shape[:-2] + (N * R, N * R))
    elif R == 0:
        result = torch.einsum("ij,...kl->...ikjl", I_left, op)
        result = result.reshape(op.shape[:-2] + (L * N, L * N))
    else:
        result = torch.einsum("ij,...kl,mn->...ikmjln", I_left, op, I_right)
        result = result.reshape(op.shape[:-2] + (L * N * R, L * N * R))
    return result


def _expand_op_zeros(op: torch.Tensor, left_dim: int, right_dim: int) -> torch.Tensor:
    """
     Embed  operator component into a larger zero-padded matrix.

     It is used for open quantum systems where the relaxation only acts on a
     subspace.

     Mathematical formulation

     O' = [ 0_L  0   0  ]
          [ 0    O   0  ]
          [ 0    0   0_R ]
     Resulting shape: (L + N + R) x (L + N + R).

     :param left_dim: Size of the top-left zero block (L).
     :param right_dim: Size of the bottom-right zero block (R).
     """
    return torch.nn.functional.pad(op, (left_dim, right_dim, left_dim, right_dim))


class ThermalBalanceMode(Enum):
    """
    Enumeration of thermal balance enforcement modes for spectral density.

    These modes control how detailed balance is enforced on the spectral density
    matrix to ensure physical consistency with thermal equilibrium.

    Attributes:
        SKIP: No thermal balance enforcement. Use for high-temperature limit
            (k_B T >> ℏω) or non-thermal baths.
        SYMMETRIC: Symmetrize spectral density using Boltzmann factors. Both
            upward and downward transitions are adjusted to satisfy detailed
            balance while preserving average coupling strength.
        COMPLEMENT: Fill missing transitions using detailed balance. Only
            upward transitions (ω < 0) that are zero are filled from their
            downward counterparts. Downward transitions (ω > 0) are preserved.
    """
    SKIP = "skip"
    SYMMETRIC = "symmetric"
    COMPLEMENT = "complement"


class BasisRadfieldManager(torch.nn.Module):
    def __init__(self, eigen_basis_flag: bool):
        """
        :param basis: basis initialized in the appropriate Context
        """
        super().__init__()
        self.eigen_basis_flag = eigen_basis_flag
        self._setup_transformers()

    def _setup_transformers(self):
        """Configure transformation methods based on the specified basis.

        This method sets up the appropriate transformation functions for operators

        When no transformation is needed (eigenbasis):
        - All transformation methods become identity operations (_transformed_skip)
        """
        if self.eigen_basis_flag:
            self.apply_basis_transform = self._transform_skip
        else:
            self.apply_basis_transform = self._transform_unitary

    def _transform_skip(
            self, operators: tp.List[torch.Tensor],  transformation_unitary: tp.Optional[torch.Tensor]
    ) -> tp.List[torch.Tensor]:
        return operators

    def _transform_unitary(
            self, operators: tp.List[torch.Tensor], transformation_unitary: torch.Tensor
    ) -> tp.List[torch.Tensor]:
        return [transform.transform_operator_to_new_basis(
            op.to(transformation_unitary.dtype), transformation_unitary
        ) for op in operators]


class RedfieldRelaxationChannel(torch.nn.Module):
    """
    Abstract base class for computing Redfield relaxation tensors.

    Units Convention:
    ┌─────────────────────────────────────────┐
    │ Input energies/frequencies:  Hz          │
    │ Spectral density input ω:    rad/s       │
    │ Spectral density output J:   rad/s       │
    │ All rates (W, Γ, R):         rad/s       │
    │ Coupling operators A:        dimensionless│
    └─────────────────────────────────────────┘

    The Redfield tensor scales as |A|² × J(ω) so the total dimension
    of this multiplication must be rad / s
    By default we ask to set |A|^2 as dimensionless then
    J(ω) must return rad/s to produce rates in rad/s.

    This class implements the logic for constructing the Redfield tensor
    based on system-bath coupling operators and spectral density functions.
    It supports both secular and non-secular approximations.

    Attributes:
        secular (bool): If True, applies the secular approximation (keeps only
            energy-conserving terms). Default is True.
        operator_components (tp.List[torch.Tensor]): List of system coupling
            operators in the eigenbasis, shaped (..., 1, N, N).
        spectral_density_func (Callable): The user-provided spectral density function.
        thermal_balance_mode (ThermalBalanceMode): Strategy for enforcing detailed balance.
    """
    def __init__(self,
        operator_components: tp.List[tp.Tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]],
        spectral_density_func: tp.Callable[[torch.Tensor], torch.Tensor],
        thermal_balance_mode: tp.Optional[str] = None):
        """
        Initialize the Redfield relaxation channel with thermal balance settings.

        This constructor configures how detailed balance is enforced on the spectral
        density matrix. The selected mode determines whether transition rates are
        modified to satisfy physical thermal equilibrium conditions at a given temperature.

        :param operator_components:
            list of tuples, where each tuple contains two base operator tensors
            used to construct a coupling channel.

            The coupling Hamiltonian is constructed as a sum over channels k:
            H_coupling = sum_k (O_static_k + field * O_dependent_k).

            For each tuple (O_static, O_dependent):
            1. The first operator (field-independent)
            is not multiplied by the external field.
            2. The second operator (field-dependent)
            is multiplied by the external field
               (e.g., Magnetic Field in Tesla).

            If a specific term is not necessary for the system, the corresponding tensor
            should be None. These tensors are constructed from fundamental operators
            (e.g., spin components Sx, Sy, Sz) and should be defined in the same basis
            as the context managing them.

            The final units of O_static we ask to be dimensionless,
            O_dependent must be 1 / Tesla.

            The Redfield tensor scales as |A|² × J(ω) so the total
            dimension of this multiplication must be rad / s
            By default we ask to set |A|^2 as dimensionless then
            J(ω) must return rad/s to produce rates in rad/s.

        :param spectral_density_func:
            Function calculating the spectral density J(ω).
                Signature: func(omega_rad_s: Tensor) -> Tensor
                - Input: Transition frequencies in rad/s (can be negative).
                - Output: Spectral density values.
                - Units: Must match `dimension_convention`.
                For 'rad_per_s', output must be rad/s.

            The Redfield rate is: W = |A|² × J(ω)
            Since |A|² is dimensionless and probabilities must be positive,
            J(ω) should return non-negative values in rad/s.

            Mathematical formulation:
            The spectral density describes the bath correlation function
            in frequency domain:
            J(omega) = ∫ dt exp(i * omega * t) * <B(t)B(0)>

            For classical noise, J(omega) is real, even, and non-negative:
            J(-omega) = J(omega) >= 0

            For quantum baths,
            detailed balance can be enforced by _compute_density() depending of flag:
            J(-omega) = J(omega) * exp(-h * omega / (k_B * T))

        :param thermal_balance_mode: Strategy for enforcing thermal detailed balance.
            Accepts either a string or a `ThermalBalanceMode` enum value:

            - "skip" (or `ThermalBalanceMode.SKIP`): No modification. Use for
              high-temperature limits ($k_B T \gg \hbar\omega$) or non-thermal baths.
            - "symmetric" (or `ThermalBalanceMode.SYMMETRIC`): Symmetrizes the
              spectral density and enforces detailed balance on all transitions.
              Both upward and downward rates are adjusted to preserve the average
              coupling strength while satisfying
              $J(-\omega) = J(\omega)e^{-\hbar\omega/k_BT}$.
              *Default if None is provided.*
            - "complement" (or `ThermalBalanceMode.COMPLEMENT`): Fills only
              missing zero entries in the upward transition matrix ($\omega < 0$)
              using the corresponding downward transitions ($\omega > 0$) and the
              Boltzmann factor. Preserves user-computed downward rates exactly.

        :raises ValueError: If `thermal_balance_mode` is a string that does not
            match one of the valid options ("skip", "symmetric", "complement").
        :raises TypeError: If `thermal_balance_mode` is neither a string,
            `ThermalBalanceMode` enum, nor `None`.
        """
        super().__init__()
        if thermal_balance_mode is None:
            self.thermal_balance_mode = ThermalBalanceMode.SYMMETRIC
        elif isinstance(thermal_balance_mode, ThermalBalanceMode):
            self.thermal_balance_mode = thermal_balance_mode
        elif isinstance(thermal_balance_mode, str):
            try:
                self.thermal_balance_mode = ThermalBalanceMode(
                    thermal_balance_mode.lower()
                )
            except ValueError:
                valid_modes = [mode.value for mode in ThermalBalanceMode]
                raise ValueError(
                    f"Invalid thermal_balance_mode '{thermal_balance_mode}'. "
                    f"Must be one of: {valid_modes}"
                )
        else:
            raise TypeError(
                f"thermal_balance_mode must be str, ThermalBalanceMode, or None, "
                f"got {type(thermal_balance_mode)}"
            )
        self._initialize_components(operator_components)
        self.spectral_density_func = spectral_density_func
        self.eigen_basis_flag = None
        self.spectral_density_modifier = self._spectral_density_factory()

    def post_init(self, eigen_basis_flag: bool = False, secular: bool = False) -> None:
        """
        Post-Initialize the Redfield relaxation logic.
        :param eigen_basis_flag: The flag should the eigen basis be used. Or there is not None trasnformation matrix.
        If True then the basis will be None, and no transformation will be performed
        :param secular: Whether to apply the secular approximation.
        """
        self.secular = secular
        self.basis_manager = BasisRadfieldManager(eigen_basis_flag)
        self.eigen_basis_flag = eigen_basis_flag

    def _spectral_density_factory(self) ->\
            tp.Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Factory method to select the spectral density modification function.

        Returns the appropriate modifier function based on the thermal_balance_mode
        set during initialization. The modifier is applied to raw spectral density
        values to enforce thermal detailed balance.

        :return: Callable that modifies spectral density matrix.
            Signature: modifier(J_raw, omega_hz, temp) → J_modified
            Where:
            - J_raw: Raw spectral density matrix (..., N, N), units: rad/s
            - omega_hz: Transition frequencies (..., N, N), units: Hz
            - temp: Temperature (...), units: Kelvin
            - J_modified: Modified spectral density (..., N, N), units: rad/s
        """
        if self.thermal_balance_mode == ThermalBalanceMode.SKIP:
            return self._modify_density_skip
        elif self.thermal_balance_mode == ThermalBalanceMode.SYMMETRIC:
            return self._modify_density_symmetric
        else:  # ThermalBalanceMode.COMPLEMENT
            return self._modify_density_complement

    def _initialize_components(
            self,
            operator_components: OperatorComponentsType) -> None:
        """
        Register and preprocess the base operator components.

        Check that init take a list
        of tuples. Each tuple contains (field-independent, field-dependent) operators.
        Adds a singleton dimension at index -3 to each tensor. This dimension
        facilitates broadcasting when applying field-dependent coefficients to build
        the final coupling operators.

        self.operator_components will be a
        List[Tuple[Optional[Tensor], Optional[Tensor]]].

        :param operator_components: operator components to set the coupling operators
        """
        if not isinstance(operator_components, list):
            raise TypeError(
                "operator_components must return a list of tuples."
            )

        self.operator_components = []
        for raw_ops in operator_components:
            self._validate_register_components(raw_ops)
            static, dep = raw_ops
            s = static.unsqueeze(-3) if static is not None else None
            d = dep.unsqueeze(-3) if dep is not None else None
            self.operator_components.append((s, d))

    def _validate_register_components(
            self,
            raw_ops: tp.Tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]
    ):
        """
        Validate the input of init function

        Ensures that exactly two operator components are returned, corresponding
        to the linear coupling model: H_coupling = O_static + field * O_dependent.
        One or both operators may be None if that term is not required for the
        specific physical system, but the tuple structure must be preserved.

        :param raw_ops: A tuple containing exactly two elements (field-independent,
            field-dependent).
        :raises ValueError: If the tuple does not contain exactly two elements.
        """
        if len(raw_ops) != 2:
            raise ValueError(
                "In the Readfield channel you should pass the "
                "tuple of two operators "
                "(field-independent, field-dependent)."
                "If one term is not needed, set it to None."
            )

    def build_coupling_operator(
            self, fields: tp.Optional[torch.Tensor]) -> tp.List[torch.Tensor]:
        """
        Construct the single final system-bath coupling operator from base components.

        This method takes the components registered in 'register_operator_components'
        and combines them to form the single effective operator that couples the
        system to the bath.

        Mathematical formulation:
        The effective coupling operator O is constructed as a linear combination:
        O(fields) = O_static + sum_k(field_k * O_dependent_k)

        In this implementation, 'fields' is broadcast against the
        'field_dependent_component'.
        If 'field_dependent_component' represents a vector of operators
        (e.g. [Sx, Sy, Sz]),
        ensure 'fields' matches the dimensions for the dot product,
        or handle the summation
        externally. Here we assume element-wise multiplication broadcasting.

        :param fields: Tensor of external fields (e.g., magnetic fields).
            Shape: (..., K), where K is number of transition.

        :return: The list of coupling  operator. Shape of each (..., 1, N, N).
        """
        operators = []
        if fields is not None:
            field_expanded = fields.unsqueeze(-1).unsqueeze(-1)
        else:
            field_expanded = 0.0

        for static, dep in self.operator_components:
            static_term = static if static is not None else 0
            dep_term = dep if dep is not None else 0

            op = dep_term * field_expanded + static_term
            operators.append(op)
        return operators

    def expand_kronecker(
            self, left_dim: int, right_dim: int) -> None:
        """
       Compute I_L ⊗ A ⊗ I_R for the coupling operator.

       It is used when the system is part of a larger composite Hilbert space.
       Note: This modifies the stored 'operator_components' list, so it should be
       called before 'build_coupling_operator' or the components should be re-registered.
       Mathematical formulation

       Given an operator component O of size N x N, the expanded component O' is:
       O' = I_L ⊗ O ⊗ I_R
       The resulting dimension is D = L * N * R.

       :param left_dim: Size of the left identity matrix (L). If 0, skipped.
       :param right_dim: Size of the right identity matrix (R). If 0, skipped.
        """
        results = []
        for static, dep in self.operator_components:
            new_static = None
            new_dep = None

            if static is not None:
                new_static = _expand_op_kronecker(static, left_dim, right_dim)
            if dep is not None:
                new_dep = _expand_op_kronecker(dep, left_dim, right_dim)

            results.append((new_static, new_dep))
        self.operator_components = results

    def _compute_boltzmann_factor(self, omega_hz: torch.Tensor,
                                  temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute the Boltzmann factor for thermal balance.

        Mathematical formulation:
        factor = exp(-h * Δν / (k_B * T))
        where Δν is in Hz, h is Planck's constant, k_B is Boltzmann's constant.

        For detailed balance: J(-ω) = J(ω) * factor

        :param omega_hz: Transition frequencies in Hz. Shape: (..., N, N).
            omega_hz[i, j] = E_j - E_i (positive for upward transitions).
        :param temperature: temperature in Kelvin. Shape: (...) or scalar.
        :return: Boltzmann factor. Shape: (..., N, N).
        """
        return torch.exp(constants.unit_converter(omega_hz, "Hz_to_K") / temperature)

    def expand_zeros(
            self, left_dim: int, right_dim: int) -> None:
        """
         Embed each operator component into a larger zero-padded matrix.

         It is used for open quantum systems where the relaxation only acts on a
         subspace.

         Mathematical formulation

         O' = [ 0_L  0   0  ]
              [ 0    O   0  ]
              [ 0    0   0_R ]
         Resulting shape: (L + N + R) x (L + N + R).

         :param left_dim: Size of the top-left zero block (L).
         :param right_dim: Size of the bottom-right zero block (R).
         """
        results = []
        for static, dep in self.operator_components:
            new_static = None
            new_dep = None
            if static is not None:
                new_static = torch.nn.functional.pad(
                    static, (left_dim, right_dim, left_dim, right_dim)
                )
            if dep is not None:
                new_dep = torch.nn.functional.pad(
                    dep, (left_dim, right_dim, left_dim, right_dim)
                )
            results.append((new_static, new_dep))
        self.operator_components = results

    def _modify_density_skip(self, J_raw: torch.Tensor,
                             omega_hz: torch.Tensor,
                             temperature: torch.Tensor) -> torch.Tensor:
        """
        No modification to spectral density.

        Use this for:
        - High-temperature limit (k_B T >> ℏω)
        - Non-thermal baths
        - When detailed balance is already enforced in spectral_density()

        :param J_raw: Raw spectral density from spectral_density(). Units: rad/s.
        :param omega_hz: Transition frequencies in Hz.
        :param temperature: Temperature in Kelvin.
        :return: Unmodified spectral density. Units: rad/s.
        """
        return J_raw

    def _modify_density_symmetric(self, J_raw: torch.Tensor,
                                  omega_hz: torch.Tensor,
                                  temperature: torch.Tensor) -> torch.Tensor:
        """
        Symmetrize spectral density using Boltzmann factors.

        Mathematical formulation:
        Given raw J_raw(ω), compute symmetric mean:
        J_sym(ω) = (J_raw(ω) + J_raw(-ω)) / 2

        Then enforce detailed balance:
        J(ω) = 2 * J_sym(ω) / (1 + exp(-hω/k_B T))
        J(-ω) = J(ω) * exp(-hω/k_B T)

        This ensures thermal equilibrium
        while preserving the average coupling strength.

        Use this for:
        - Thermal baths where only symmetric coupling is known
        - When J(ω) should satisfy detailed balance exactly

        :param J_raw: Raw spectral density. Units: rad/s.
        :param omega_hz: Transition frequencies in Hz. omega[i,j] = E_j - E_i.
        :param temperature: Temperature in Kelvin.
        :return: Symmetrized spectral density. Units: rad/s.
        """
        J_raw_neg = J_raw.transpose(-2, -1)
        J_sym = (J_raw + J_raw_neg) / 2.0

        boltzmann = self._compute_boltzmann_factor(omega_hz, temperature)
        J_thermal = 2.0 * J_sym / (1.0 + boltzmann)

        return J_thermal

    def _modify_density_complement(self, J_raw: torch.Tensor,
                                   omega_hz: torch.Tensor,
                                   temperature: torch.Tensor) -> torch.Tensor:
        """
        Enforce quantum detailed balance: J(-ω) = J(ω) × exp(-ℏω / k_B T)

        Only modifies entries where omega < 0 (upward transitions / absorption).
        Downward transitions (omega > 0, emission) are preserved as computed.

        Matrix convention (matches EvolutionBase):
        ┌─────────────────────────────────────────────────────────────────────────┐
        │ J[i, j] = spectral density for transition j → i                         │
        │ omega_hz[i, j] = E_j - E_i  (transition frequency in Hz)                │
        │                                                                         │
        │ Sign convention:                                                        │
        │   omega_hz[i, j] > 0  →  E_j > E_i  →  downward transition (emission)   │
        │   omega_hz[i, j] < 0  →  E_j < E_i  →  upward transition (absorption)   │
        │                                                                         │
        │ Detailed balance:                                                       │
        │   J[j, i] = J[i, j] × exp(-(E_j - E_i) / k_B T)                         │
        │   J(-ω)   = J(ω)    × exp(-ℏω / k_B T)                                  │
        └─────────────────────────────────────────────────────────────────────────┘

        Implementation logic:
        ┌─────────────────────────────────────────────────────────────────────────┐
        │ WHERE omega_hz[i, j] < 0 (upward, absorption) AND J[i, j] == 0:         │
        │   → Fill J[i, j] = J[j, i] × exp((E_j - E_i) / k_B T)                   │
        │   → Uses downward transition J[j, i] (where omega > 0) as reference     │
        │                                                                         │
        │ WHERE omega_hz[i, j] >= 0 (downward, emission):                         │
        │   → Leave J[i, j] unchanged (user/computed value preserved)             │
        │                                                                         │
        │ WHERE J[i, j] > 0 (already computed):                                   │
        │   → Leave unchanged regardless of omega sign                            │
        └─────────────────────────────────────────────────────────────────────────┘

        :param J_raw: Raw spectral density matrix. Units: rad/s.
            J_raw[i, j] corresponds to transition j → i.
        :param omega_hz: Transition frequencies in Hz.
            omega_hz[i, j] = E_j - E_i.
            omega_hz > 0: downward (emission), omega_hz < 0: upward (absorption).
        :param temperature: Temperature in Kelvin.
        :return: Completed spectral density matrix. Units: rad/s.
            Only upward transitions (omega < 0) are modified.
        """
        J_result = J_raw.clone()
        energy_diff_K = constants.unit_converter(omega_hz, "Hz_to_K")
        temp_safe = torch.clamp(temperature, min=1e-10)
        boltzmann = torch.exp(energy_diff_K / temp_safe)
        J_transpose = J_raw.transpose(-2, -1)
        J_filled = J_transpose * boltzmann
        upward_and_missing = (omega_hz < 0) & (J_raw == 0)

        J_result = torch.where(upward_and_missing, J_filled, J_result)

        return J_result

    def _get_spectral_density_matrix(self, energies: torch.Tensor,
                                     temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute the matrix J_cd = J(omega_cd) for all pairs of eigenstates.

        :param energies: Tensor of eigenenergies in Hz. Shape: (..., N).
        :return: Tensor of shape (..., N, N) with J(omega_cd).
        """
        # omega_cd = E_up - E_down
        omega_hz = energies.unsqueeze(-2) - energies.unsqueeze(-1)
        return self.spectral_density_modifier(
            self.spectral_density_func(omega_hz / 2 * math.pi), omega_hz, temperature
        )

    def get_coupling_operators(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: torch.Tensor) -> tp.List[torch.Tensor]:
        """
        Wrapper to get the final field-dependent coupling operators in the eigenbasis.

        Passes the registered components to 'build_coupling_operator' to construct
        the list of effective operators, then transforms them to the eigenbasis
        using the provided unitary transformation matrix.

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External fields (e.g., magnetic field). Shape: (..., K).
        :return: A list of coupling operators in the eigenbasis.
            Each has Shape: (..., N, N).
        """
        ops_original = self.build_coupling_operator(fields)
        ops_eigen = self.basis_manager.apply_basis_transform(
            ops_original, transformation_unitary)
        return ops_eigen

    def _compute_transition_probs(self, operators: tp.List[torch.Tensor],
                                  spec_density: torch.Tensor) -> torch.Tensor:
        """
       Compute secular population transfer rates W_a<-c.

       Mathematical formulation

       The rate from state c to state a is given by Fermi's Golden Rule summed over
       all coupling operators k:
       W_a<-c = sum_k |(A_k)_ac|^2 * J(omega_ca)
       where omega_ca = E_c - E_a. Note that 'spec_density' is indexed as J_ca.

       :param operators: List of coupling operators in eigenbasis.
           Each Shape: (..., N, N).
       :param spec_density: Spectral density matrix J(omega_cd). Shape: (..., N, N).
       :return: Transition rate matrix W. Shape: (..., N, N).
           W[a, c] is the rate c -> a. Diagonal is zero.
       """
        W = torch.zeros_like(spec_density)
        J_trans = spec_density.transpose(-1, -2)

        for op in operators:
            sq = op.abs() ** 2
            W += sq * J_trans
        N = operators[0].shape[-1] if operators else spec_density.shape[-1]
        diag_mask = torch.eye(N, device=W.device, dtype=torch.bool)
        W = W.masked_fill(diag_mask, 0.0)
        return W

    def transition_probabilities(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: tp.Optional[torch.Tensor], energies: torch.Tensor,
            temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute the population transfer rate matrix.

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies in Hz. The shape [..., N]
        :param temperature: The system stationary temperature  in Kelvin.
        The shape is [] or [t], where t is number of time-steps.
        :return: Rate matrix W.
        """
        operators = self.get_coupling_operators(transformation_unitary, fields)
        spec_density = self._get_spectral_density_matrix(energies, temperature)
        return self._compute_transition_probs(operators, spec_density)

    def _compute_dephasing_matrix(
            self, operators: tp.List[torch.Tensor],
            spec_density: torch.Tensor) -> torch.Tensor:
        """
        Compute pure dephasing rates for coherences rho_ab (a != b).

        Mathematical formulation

        In the secular approximation, the decay rate of coherence rho_ab
        due to population relaxation is:
        Gamma_ab_pop = 0.5 * (W_sum_a + W_sum_b)
        where W_sum_i is the total rate out of state i.

        Additionally, pure dephasing (elastic scattering) contributes:
        Gamma_ab_pure = sum_k |A^k_aa - A^k_bb|^2 * J(0)

        This method computes the total dephasing rate.

        :param operators: List of coupling operators. Each Shape: (..., N, N).
        :param spec_density: Spectral density. Shape: (..., N, N).
        :return: Dephasing rate matrix. Shape: (..., N, N).
        """
        W = self._compute_transition_probs(operators, spec_density)

        rate_out = W.sum(dim=-2, keepdim=True)

        gamma_pop = 0.5 * (rate_out + rate_out.transpose(-1, -2))

        J0 = spec_density.diagonal(offset=0, dim1=-2, dim2=-1)

        pure_dephasing = torch.zeros_like(gamma_pop)

        for op in operators:
            op_diag = torch.diagonal(op, offset=0, dim1=-2, dim2=-1)
            diff = op_diag.unsqueeze(-1) - op_diag.unsqueeze(-2)
            sq_diff = diff.abs() ** 2
            pure_dephasing += sq_diff * J0.unsqueeze(-2)

        return gamma_pop + pure_dephasing

    def dephasing_matrix(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: tp.Optional[torch.Tensor],
            energies: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """
        Public wrapper for dephasing rate calculation.

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies in Hz. The shape [..., N]
        :param temperature: The system stationary temperature  in Kelvin.
        The shape is [] or [t], where t is number of time-steps.
        :return: Dephasing rates.
        """
        operators = self.get_coupling_operators(transformation_unitary, fields)
        spec_density = self._get_spectral_density_matrix(energies, temperature)
        return self._compute_dephasing_matrix(operators, spec_density)

    def _secular_superoperator_4d(
            self, operators: tp.List[torch.Tensor],
            spec_density: torch.Tensor) -> torch.Tensor:
        """
        Compute the Redfield tensor under the secular approximation.

        Mathematical formulation

        The secular approximation retains only terms where omega_ab approx omega_cd.
        For non-degenerate systems, this implies a=c and b=d for coherences,
        and a=b, c=d for populations.

        The 4D tensor R_abcd has three main contributions:
        1. Population Transfer (a=b, c=d): R_aacc = W_a<-c for a != c.
        2. Coherence Decay including population loss because of population (a!=b, c=a, d=b): R_abab = -Gamma_ab.
        All other terms are zero.

        :param operators: List of coupling operators.
        :param spec_density: Spectral density matrix.
        :return: 4D Relaxation tensor. Shape: (..., N, N, N, N).
        """
        W = self._compute_transition_probs(operators, spec_density)
        gamma = self._compute_dephasing_matrix(operators, spec_density)

        N = W.shape[-1]
        device = W.device
        R = torch.zeros(
            W.shape[:-2] + (N, N, N, N), dtype=W.dtype, device=device)

        idx = torch.arange(N, device=device)
        i, j = torch.meshgrid(idx, idx, indexing="ij")

        R[..., i, i, j, j] = W
        R[..., i, j, i, j] = -gamma

        return R

    def _non_secular_superoperator_4d(
            self, operators: tp.List[torch.Tensor],
            spec_density: torch.Tensor) -> torch.Tensor:
        """
        Compute the full Redfield tensor without secular approximation.

        Mathematical formulation (Section 3.4)

        The relaxation superoperator elements R_{abcd} are computed using the
        Bloch-Redfield formalism. The total Hamiltonian is partitioned into
        system, bath, and interaction terms, where the interaction is defined
        by coupling operators A^k and bath operators B^k.

        The intermediate tensor Gamma_{abcd}(omega) is defined as:
        Gamma_{abcd}(omega) = sum_k A^k_{ab} A^k_{cd}* J_k(omega)

        The full Redfield tensor R_{abcd} consists of four terms summed over
        intermediate states e:
        R_{abcd} = 0.5 * sum_e [
            Gamma_{aece}(omega_{ea}) * delta_{bd} +
            Gamma_{debe}*(omega_{ed}) * delta_{ac} -
            Gamma_{acbd}(omega_{db}) -
            Gamma_{cadb}*(omega_{db})
        ]

        This includes all terms allowing for coherence transfer between transitions
        with similar frequencies (non-secular).

        :param operators: List of coupling operators A^k. Shape: (..., N, N).
        :param spec_density: Spectral density matrix J_k(omega).
                             Shape: (..., N, N) corresponding to transition frequencies.
        :return: 4D Relaxation tensor R_{abcd}. Shape: (..., N, N, N, N).
        """
        ref_op = operators[0]
        spec_density = spec_density.to(ref_op.dtype)
        N = ref_op.shape[-1]
        device = ref_op.device

        # Compute sum_k A^k_{ab} A^k_{cd}*
        # This corresponds to the operator part of Gamma_{abcd}
        A_Aconj_sum = torch.zeros(
            ref_op.shape[:-2] + (N, N, N, N), dtype=ref_op.dtype, device=device)
        for op in operators:
            A_Aconj_sum += torch.einsum("...ab,...cd->...abcd", op, op.conj())

        R = torch.zeros_like(A_Aconj_sum)

        permute_indices = list(range(A_Aconj_sum.dim() - 4)) + [-1, -3, -4, -2]
        term3_part = A_Aconj_sum.permute(permute_indices)

        J_db = spec_density.unsqueeze(-3).unsqueeze(-3)
        R -= term3_part * J_db

        J_db_expanded = spec_density.unsqueeze(-2).unsqueeze(-1)
        R -= term3_part.conj() * J_db_expanded

        diag_sum = torch.einsum("...apcp,...pa->...ac", A_Aconj_sum, spec_density)
        I_bd = torch.eye(N, device=device, dtype=ref_op.dtype)
        term1 = torch.einsum("...ac,bd->...abcd", diag_sum, I_bd)
        R += term1

        diag_sum2 = torch.einsum("...dpbp,...pd->...db", A_Aconj_sum, spec_density)
        I_ac = torch.eye(N, device=device, dtype=ref_op.dtype)
        term2 = torch.einsum("...db,ac->...abcd", diag_sum2.conj(), I_ac)
        R += term2
        R *= 0.5

        return R

    def compute_relaxation_superoperator_4d(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: tp.Optional[torch.Tensor],
            energies: torch.Tensor,
            temperature: torch.Tensor) -> torch.Tensor:
        """
         Compute the 4D Redfield relaxation tensor.

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies in Hz. The shape [..., N]
        :param temperature: The system stationary temperature  in Kelvin.
        The shape is [] or [t], where t is number of time-steps.
        :return: Tensor R_abcd. Shape: (..., N, N, N, N).
        """
        operators = self.get_coupling_operators(transformation_unitary, fields)
        spec_density_matrix = self._get_spectral_density_matrix(
            energies, temperature
        )

        if self.secular:
            return self._secular_superoperator_4d(operators, spec_density_matrix)
        else:
            return self._non_secular_superoperator_4d(operators, spec_density_matrix)

    def compute_relaxation_superoperator(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: tp.Optional[torch.Tensor],
            energies: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute the 2D Liouvillian relaxation superoperator.

        Mathematical formulation

        The density matrix rho is vectorized in row-major order:
        |rho>> = [rho_00, rho_01, ..., rho_0N, rho_10, ...]^T
        The superoperator R acts as:
        d/dt |rho>> = R |rho>>
        where R_(ab),(cd) = R_abcd.

        The row-major index for element (i, j) is i * N + j.
        The superoperator maps index (c, d) to (a, b).

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies in Hz. The shape [..., N]
        :param temperature: The system temperature in K.
        The shape is [] or [t], where t is number of time-steps
        :return: Superoperator matrix. Shape: (..., N^2, N^2).
        """
        R4 = self.compute_relaxation_superoperator_4d(transformation_unitary, fields, energies, temperature)
        N = R4.shape[-1]
        return R4.reshape(R4.shape[:-4] + (N * N, N * N))

    def get_dephasing_vector(
            self, transformation_unitary: tp.Optional[torch.Tensor], fields: torch.Tensor,
            energies: torch.Tensor,
            temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute the level-specific dephasing vector gamma_i for Lindblad operators L_i = |i><i|.

        This vector satisfies Gamma_ij approx 0.5 * (gamma_i + gamma_j) for the population
        relaxation contribution. Note that the pure dephasing term (J(0)) cannot be exactly
        represented by diagonal Lindblad operators and is excluded from this vector to
        maintain mathematical consistency with the Lindblad form.

        Mathematical formulation

        The returned vector gamma corresponds to the total transition probability out of state i:
        gamma_i = sum_{k != i} W_{k<-i}

        This ensures that the coherence decay due to population transfer is correctly modeled
        by Lindblad operators L_i = sqrt(gamma_i) |i><i|.

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies in Hz. The shape [..., N]
        :param temperature: The system tempreature in K
        The shape is [] or [t], where t is number of time-steps
        :return: Dephasing vector gamma. Shape: (..., N).
        """
        operators = self.get_coupling_operators(transformation_unitary, fields)
        spec_density = self._get_spectral_density_matrix(energies, temperature)

        W = self._compute_transition_probs(operators, spec_density)
        gamma_vector = W.sum(dim=-2)
        return gamma_vector

    def close(self):
        """
        close redfield channel
        """
        self.basis_manager.transformation_unitary = None


class RedfieldManager:
    """
    The manager is used to manage the several redfield_channels.
    Each redfield_channel can have any number of coupling operators but only one spectral desnity
    """
    def __init__(self, redfield_channels: tp.List[RedfieldRelaxationChannel]):
        self.redfield_channels = redfield_channels

        self.eigen_basis_flag = self.redfield_channels[0].eigen_basis_flag
        if not all(channel.eigen_basis_flag == self.eigen_basis_flag
                   for channel in self.redfield_channels):
            raise ValueError(
                "All redfield channels must have the same eigen_basis_flag "
                "(all True or all False)"
            )

    def expand_zeros(
            self, left_dim: int, right_dim: int) -> None:
        """
         Embed each operator in each channel into a larger zero-padded matrix.

         It is used for open quantum systems where the relaxation only acts on a
         subspace.

         Mathematical formulation

         O' = [ 0_L  0   0  ]
              [ 0    O   0  ]
              [ 0    0   0_R ]
         Resulting shape: (L + N + R) x (L + N + R).

         :param left_dim: Size of the top-left zero block (L).
         :param right_dim: Size of the bottom-right zero block (R).
         """
        for channel in self.redfield_channels:
            channel.expand_zeros(left_dim, right_dim)

    def expand_kronecker(
            self, left_dim: int, right_dim: int) -> None:
        """
       Compute I_L ⊗ A ⊗ I_R for the coupling operators in each channel.

       It is used when the system is part of a larger composite Hilbert space.
       Note: This modifies the stored 'operator_components' list, so it should be
       called before 'build_coupling_operator' or the components should be re-registered.
       Mathematical formulation

       Given an operator component O of size N x N, the expanded component O' is:
       O' = I_L ⊗ O ⊗ I_R
       The resulting dimension is D = L * N * R.

       :param left_dim: Size of the left identity matrix (L). If 0, skipped.
       :param right_dim: Size of the right identity matrix (R). If 0, skipped.
        """
        for channel in self.redfield_channels:
            channel.expand_kronecker(left_dim, right_dim)

    def compute_transition_probabilities(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: tp.Optional[torch.Tensor], energies: torch.Tensor,
            temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute the population transfer rate matrix. The diagonal elements are zeros

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies In Hz. The shape [..., N]
        :param temperature: The system temperature in K
        The shape is [] or [t], where t is number of time-steps
        :return: Rate matrix W.
        """
        return sum(
            channel.transition_probabilities(
                transformation_unitary, fields, energies, temperature
            ) for channel in self.redfield_channels
        )

    def compute_relaxation_superoperator(
            self, transformation_unitary: tp.Optional[torch.Tensor], fields: tp.Optional[torch.Tensor],
            energies: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute the 2D Liouvillian relaxation superoperator.

        Mathematical formulation

        The density matrix rho is vectorized in row-major order:
        |rho>> = [rho_00, rho_01, ..., rho_0N, rho_10, ...]^T
        The superoperator R acts as:
        d/dt |rho>> = R |rho>>
        where R_(ab),(cd) = R_abcd.

        The row-major index for element (i, j) is i * N + j.
        The superoperator maps index (c, d) to (a, b).

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies in Hz. The shape [..., N]
        :param temperature: The system temperature in Kelvin.
        The shape is [] or [t], where t is number of time-steps
        :return: Superoperator matrix. Shape: (..., N^2, N^2).
        """
        return sum(
            channel.compute_relaxation_superoperator(
                transformation_unitary, fields, energies, temperature) for channel in
            self.redfield_channels
        )


def combine_redfield_managers(redfield_managers: tp.List[tp.Optional[RedfieldManager]]) -> RedfieldManager:
    """
    Combine multiple Redfield managers into a single manager.

    Conceptually, this merges all relaxation channels from different subsystems
    into one unified manager, safely ignoring any missing (None) entries.

    :param redfield_managers: List of managers to combine. Some items may be None.
    :return: A new RedfieldManager containing all valid channels.
    """
    channels = [
        channel
        for manager in redfield_managers
        if manager is not None
        for channel in manager.redfield_channels
    ]

    return RedfieldManager(channels) if channels else None

