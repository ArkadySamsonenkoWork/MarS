import math
from abc import ABC, abstractmethod
import torch
import typing as tp

from . import contexts
from .. import constants


class BaseGenerator(ABC):
    def __init__(self,
                 context: contexts.BaseContext,
                 init_temperature: torch.Tensor,
                 full_system_vectors: tp.Optional[torch.Tensor] = None,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        """
        :param context: Context Describing the algorithm of relaxation processes
        :param args:
        :param kwargs:
        """
        super().__init__()
        self.context = context
        self.init_temperature = init_temperature
        self.full_system_vectors = full_system_vectors

    @abstractmethod
    def __call__(self, time: torch.Tensor):
        pass


class LevelBasedGenerator(BaseGenerator):
    """
    Abstract base class for generating transition probability matrices in a multi-level
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
    def __call__(self, time: torch.Tensor) ->\
            tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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
        induced_probs = self._driven_transition_probs(time_dep_values)
        out_probs = self._outgoing_transition_probs(time_dep_values)
        return temp.unsqueeze(-1).unsqueeze(-1), free_probs, induced_probs, out_probs

    def _temperature(self, time_dep_values: torch.Tensor) -> torch.Tensor | None:
        """Return temperature(s) at times t"""
        return self.init_temperature

    def _base_transition_probs(self, time_dep_values: torch.Tensor | None) -> torch.Tensor:
        """"""
        return self.context.get_transformed_free_probs(self.full_system_vectors, time_dep_values)

    def _driven_transition_probs(self, time_dep_values: torch.Tensor | None) -> torch.Tensor | None:
        """Optional driven transitions; default None"""
        return self.context.get_transformed_driven_probs(self.full_system_vectors, time_dep_values)

    def _outgoing_transition_probs(self, time_dep_values: torch.Tensor | None) -> torch.Tensor | None:
        """Optional outgoing transitions; default None"""
        return self.context.get_transformed_out_probs(self.full_system_vectors, time_dep_values)


class TempDepGenerator(LevelBasedGenerator):
    def _temperature(self, time_dep_values: torch.Tensor) -> torch.Tensor | None:
        """Return temperature(s) at times t"""
        return time_dep_values


class DensityRWAGenerator(BaseGenerator):
    """
    Abstract base class for generating density transition probability matrices in or multi-level
    using rotating wave approximation
    """
    def __init__(self,
                 context: contexts.BaseContext,
                 stationary_hamiltonian: torch.Tensor,
                 init_temperature: torch.Tensor,
                 lvl_down, lvl_up,
                 full_system_vectors: tp.Optional[torch.Tensor] = None,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        """
        :param context: Context Describing the algorithm of relaxation processes
        :param args:
        :param kwargs:
        """
        super().__init__(context, init_temperature, full_system_vectors, device, dtype)
        self.stationary_hamiltonian = stationary_hamiltonian
        self.level_down = lvl_down
        self.level_up = lvl_up
        """
        self.rwa_ham = self.stationary_hamiltonian.clone().to(self.oscillating_hamiltonian.dtype)
        indexes = torch.arange(self.rwa_ham.shape[-3])
        print(self.rwa_ham[0, 0])
        self.rwa_ham[..., indexes, lvl_down, lvl_down] += self.omega / 2
        self.rwa_ham[..., indexes, lvl_up, lvl_up] -= self.omega / 2
        print(self.rwa_ham[0, 0])

        self.rwa_ham[..., indexes, lvl_down, lvl_up] =\
            self.oscillating_hamiltonian[..., indexes, lvl_down, lvl_up]
        self.rwa_ham[..., indexes, lvl_up, lvl_down] =\
            self.oscillating_hamiltonian[
            ..., indexes, lvl_up, lvl_down]
        """

    def __call__(self, time: torch.Tensor) ->\
            tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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
        free_superop = self._base_superop(time_dep_values)
        driven_superop = self._driven_superop(time_dep_values)
        return temp, self.stationary_hamiltonian, free_superop, driven_superop

    def _temperature(self, time_dep_values: torch.Tensor | None) -> torch.Tensor | None:
        """Return temperature(s) at times t"""
        return self.init_temperature

    def _base_superop(self, time_dep_values: torch.Tensor | None) -> torch.Tensor:
        """"""
        return self.context.get_transformed_free_superop(self.full_system_vectors, time_dep_values)

    def _driven_superop(self, time_dep_values: torch.Tensor | None) -> torch.Tensor | None:
        """Optional driven transitions; default None"""
        return self.context.get_transformed_driven_superop(self.full_system_vectors, time_dep_values)


class DensityPropagatorGenerator(DensityRWAGenerator):
    def __call__(self, time: torch.Tensor) ->\
            tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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
            raise NotImplementedError(
                "Propagator solution of evolution doesn't support time dependant relaxation rates"
            )
        time_dep_values = None
        temp = self._temperature(time_dep_values)
        free_superop = self._base_superop(time_dep_values)
        driven_superop = self._driven_superop(time_dep_values)
        return temp, self.stationary_hamiltonian, free_superop, driven_superop
