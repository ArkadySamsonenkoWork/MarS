import math
from abc import ABC, abstractmethod
import torch
import typing as tp

from . import contexts
from .. import constants


class BaseGenerator(ABC):
    """
    Abstract base class for time-dependent generators of relaxation-related computations.

    Encapsulates a context that defines initial states, transition rates, and basis transformations,
    along with temperature and eigenvector data required to evaluate relaxation operators at arbitrary times.
    """
    def __init__(self,
                 context: contexts.BaseContext,
                 init_temperature: torch.Tensor,
                 res_fields: tp.Optional[torch.Tensor],
                 full_system_vectors: tp.Optional[torch.Tensor],
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        """
        :param context: Context object instance.

        :param init_temperature:  initial temperature of process.
        -It can be constant during the process
        -It can be skipped if the temperature defines from profile
        :param res_fields:
            Resonance fields of transitions.
            Shape: [..., M], where M is the number of resonance energies.
        :param full_system_vectors:
            Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
            where M is number of transitions, N is number of levels
            For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
            make the creator to compute these vectors.
            The default behavior, whether to calculate vectors or not,
            depends on the specific Spectra Manager and its settings.

        :param device: Computation device
        :param dtype:
        :param args:
        :param kwargs:
        """
        super().__init__()
        self.context = context
        self.init_temperature = init_temperature
        self.full_system_vectors = full_system_vectors
        self.res_fields = res_fields

    @abstractmethod
    def __call__(self, time: torch.Tensor):
        pass


class LevelBasedGenerator(BaseGenerator):
    """
    Abstract base class for generating transition probability matrices in a multi-level.

    system with populations and energy differences.
    The system of rate equations for two levels with populations n1, n2 and energies E1, E2 is:
        dn1/dt = -out_1 - k1 * n1 + k2 * n2
        dn2/dt = -out_2 + k1 * n1 - k2 * n2

    which can be written in matrix form:

        dN/dt = -OUT + K @ N
    where:
      - OUT is a vector of outgoing transitions from the system,
      - K is the relaxation matrix:
            K = [[-k1,  k2],
                 [ k1, -k2]]

    K itself can be rewritten via K' and driven transition DR
    K = K' + DR, where
        K'   – equilibrium relaxation (thermal),
        DR  – driven_probs transitions,

    At thermal equilibrium, transition rates satisfy detailed balance:
        k'1 / k'2 = n'2 / n'1 = exp(-(E2 - E1) / kT)
    Defining the average relaxation rate:

        k' = (k'1 + k'2) / 2

    we can compute:
        k'2 = 2k' / (1 + exp(-(E2 - E1) / kT))
        k'1 = 2k' * exp(-(E2 - E1) / kT) / (1 + exp(-(E2 - E1) / kT))

    In symmetric form, the "free probabilities" matrix (i.e. mean equilibrium transition probabilities) is:

        base_probs= [[0,  k'],
                    [k', 0]]

    DR matrix is matrix which probabilities are not connected by thermal equilibrium:
                     [[0,  dr_1],
                     [dr_2, 0]]
    """

    def __call__(self, time: torch.Tensor) -> tp.Tuple[
        tp.Optional[torch.Tensor],
        tp.Optional[torch.Tensor],
        tp.Optional[torch.Tensor],
        tp.Optional[torch.Tensor]
    ]:
        """
        Evaluate transition probabilities at given measurement times.

        Parameters
        :param time: torch.Tensor
        :return: tuple
            (temperature, base_probs, induced_probs, outgoing_probs)
            - temperature : torch.Tensor or None
                System temperature(s) at the given time(s).
            - free_probs : torch.Tensor [..., N, N]
                Thermal equilibrium (Boltzmann-weighted) transition probabilities.

            Example in symmetry form:
                free_probs = [[0,  k'],
                            [k', 0]]

            - induced_probs : torch.Tensor [..., N, N] or None
                Probabilities of driven transitions (e.g. due to external driving).

                Ind matrix is always symmetry: [[0,  i],
                                                [i, 0]]

            - out_probs : torch.Tensor [..., N]  or None
                Out-of-system transition probabilities (loss terms).
        """
        if self.context.time_dependant:
            time_dep_values = self.context.get_time_dependent_values(time)
        else:
            time_dep_values = None

        temp = self._temperature(time_dep_values)
        free_probs = self._base_transition_probs(time_dep_values)
        driven_probs = self._driven_transition_probs(time_dep_values)
        out_probs = self._outgoing_transition_probs(time_dep_values)
        return temp.unsqueeze(-1).unsqueeze(-1), free_probs, driven_probs, out_probs

    def _temperature(self, time_dep_values: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """Return temperature(s) at times t."""
        return self.init_temperature

    def _base_transition_probs(self, time_dep_values: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        """
        Retrieve spontaneous (free) transition probabilities transformed into the eigenbasis.

        These rates are subject to Boltzmann detailed balance.
        :param time_dep_values: Optional time-dependent scaling factors from the Context profile.
        :return: Tensor of shape [..., N, N] representing equilibrium transition rates.
        """
        return self.context.get_transformed_free_probs(self.full_system_vectors, time_dep_values)

    def _driven_transition_probs(self, time_dep_values: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        """
        Retrieve non-thermal (driven) transition probabilities in the eigenbasis.

        These rates are not modified by thermal constraints and represent external perturbations.
        :param time_dep_values: Optional time-dependent scaling factors from the Context profile.
        :return: Tensor of shape [..., N, N] or None if no driven processes are defined.
        """
        return self.context.get_transformed_driven_probs(self.full_system_vectors, time_dep_values)

    def _outgoing_transition_probs(self, time_dep_values: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        """
        Retrieve irreversible loss rates from each energy level in the eigenbasis.

        These represent population removal from the spin system (e.g., phosphorescence).
        :param time_dep_values: Optional time-dependent scaling factors from the Context profile.
        :return: Vector of shape [..., N] or None if no loss processes are defined.
        """
        return self.context.get_transformed_out_probs(self.full_system_vectors, time_dep_values)


class TempDepGenerator(LevelBasedGenerator):
    """
    Extension of LevelBasedGenerator where system temperature is time-dependent.

    This assumes the profile(t) returns absolute temperature in Kelvin at time t.
    All thermal transition rates are recomputed at each time step using the instantaneous temperature.
    """
    def _temperature(self, time_dep_values: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """
        Return time-dependent temperature from the Context profile.

        Assumes time_dep_values contains temperature values evaluated at 'time'.
        :param time_dep_values: Tensor of shape compatible with [..., 1, 1] containing temperatures.
        :return: Same as time_dep_values.
        """
        return time_dep_values


class DensityRWAGenerator(BaseGenerator):
    """
    Generator for Liouville-space superoperators under the rotating wave approximation (RWA).

    Returns the Hamiltonian H and two superoperators:
      - free_superop: spontaneous processes (thermal relaxation, losses, dephsing),
      - driven_superop: external non-equilibrium driving.

    The full Liouvillian is L = -i[H, ·] + R_free + R_driven, where [·,·] is the commutator,
    and R terms are relaxation superoperators.
    H is defined in rotating frame and equel to H =

    Unlike population-based models, this class preserves quantum coherences and operates on vectorized
    density matrices of size N^2 x N^2.
    """
    def __init__(self,
                 context: contexts.BaseContext,
                 init_temperature: torch.Tensor,
                 res_fields: torch.Tensor,
                 full_system_vectors: tp.Optional[torch.Tensor],
                 stationary_hamiltonian: torch.Tensor,
                 lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        """
        :param context: Context object instance.

        :param init_temperature:  initial temperature of process.
        -It can be constant during the process
        -It can be skipped if the temperature defines from profile

        :param res_fields:
            Resonance fields of transitions.
            Shape: [..., M], where M is the number of resonance energies.

        :param full_system_vectors:
            Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
            where M is number of transitions, N is number of levels
            For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
            make the creator to compute these vectors
            The default behavior, whether to calculate vectors or not,
            depends on the specific Spectra Manager and its settings.

        :param stationary_hamiltonian: The Hamiltonian in the given frame. The definition depends on approximations.
            -For RWA it uses full Hamiltonian in rotating frame.
            -For Propagator it uses real stationary Hamiltonian

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.
        :param device: Computation device
        :param dtype:
        :param args:
        :param kwargs:
        """
        super().__init__(context, init_temperature, res_fields, full_system_vectors, device, dtype)
        self.stationary_hamiltonian = stationary_hamiltonian
        self.level_down = lvl_down
        self.level_up = lvl_up

    def __call__(self, time: torch.Tensor) -> tp.Tuple[
        tp.Optional[torch.Tensor],
        torch.Tensor,
        tp.Optional[torch.Tensor],
        tp.Optional[torch.Tensor]
    ]:
        """
        Evaluate time-dependent relaxation superoperators for density-matrix evolution.

        :param time: Tensor of shape [T] containing measurement times.
        :return: A 4-tuple:
            - temperature: scalar tensor self.init_temperature (shape broadcastable to [..., 1, 1]).
            - stationary_hamiltonian: Hermitian operator in rotating frame H of shape [..., N, N].
            In dimensionless units, it is defined as:
                H = H0 + Sz + Ht(r), where H0 is stationary Hamiltonian, Sz is z-projection of total spin,
                Ht(r) is oscillating Hamiltonian in rotating frame (in this frame it doesn't depend on time)

            - free_superop: Liouville-space superoperator of shape [..., N^2, N^2] for thermal relaxation.
            - driven_superop: Liouville-space superoperator of shape [..., N^2, N^2] for non-thermal driving.
        """
        if self.context.time_dependant:
            time_dep_values = self.context.get_time_dependent_values(time)
        else:
            time_dep_values = None

        temp = self._temperature(time_dep_values)
        free_superop = self._base_superop(time_dep_values)
        driven_superop = self._driven_superop(time_dep_values)
        return temp, self.stationary_hamiltonian, free_superop, driven_superop

    def _temperature(self, time_dep_values: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        """Return temperature(s) at times t."""
        return self.init_temperature

    def _base_superop(self, time_dep_values: tp.Optional[torch.Tensor]) -> torch.Tensor:
        """
        Retrieve spontaneous relaxation superoperator in Liouville space.

        Includes spontaneous transitions, losses, and dephsing, and is corrected for detailed balance.
        :param time_dep_values: Optional time-dependent scaling from Context profile.
        :return: Superoperator tensor of shape [..., N^2, N^2].
        """
        return self.context.get_transformed_free_superop(self.full_system_vectors, time_dep_values)

    def _driven_superop(self, time_dep_values: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        """
        Retrieve non-thermal relaxation superoperator in Liouville space.

        Represents external driving not constrained by thermal equilibrium.
        :param time_dep_values: Optional time-dependent scaling from Context profile.
        :return: Superoperator tensor of shape [..., N^2, N^2] or None.
        """
        return self.context.get_transformed_driven_superop(self.full_system_vectors, time_dep_values)


class DensityPropagatorGenerator(DensityRWAGenerator):
    """
    Generator for full propagator-based density-matrix evolution without RWA.

    Assumes time-independent relaxation rates.
    Time dependence in the Context profile is explicitly disallowed.
    """
    def __call__(self, time: torch.Tensor) -> tp.Tuple[
        tp.Optional[torch.Tensor],
        torch.Tensor,
        tp.Optional[torch.Tensor],
        tp.Optional[torch.Tensor]
    ]:
        """
        Evaluate time-dependent relaxation superoperators for density-matrix evolution.

        :param time: Tensor of shape [T] containing measurement times.
        :return: A 4-tuple:
            - temperature: scalar tensor self.init_temperature (shape broadcastable to [..., 1, 1]).
            - stationary_hamiltonian: Hermitian stationary Hamiltonian H0
             of shape [..., N, N]. In dimensionless units,

            - free_superop: Liouville-space superoperator of shape [..., N^2, N^2] for thermal relaxation.
            - driven_superop: Liouville-space superoperator of shape [..., N^2, N^2] for non-thermal driving.
        """
        if self.context.time_dependant:
            raise NotImplementedError(
                "Propagator solution of evolution doesn't support time dependant relaxation rates"
            )
        time_dep_values = None
        temp = self._temperature(time_dep_values)
        free_superop = self._base_superop(time_dep_values)
        driven_superop = self._driven_superop(time_dep_values)
        return temp, self.stationary_hamiltonian, free_superop, driven_superop
