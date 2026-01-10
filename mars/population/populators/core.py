from abc import ABC, abstractmethod

import torch
from torch import nn

from ... import constants
import typing as tp

from .. import tr_utils
from .. import matrix_generators
from .. import contexts


class BasePopulator(nn.Module):
    """
    Base class for populators.

    A populator is responsible for computing the part of the EPR transition intensity
    that depends on the populations of energy levels (or the full density matrix in more advanced cases).
    This includes:
      - Thermal equilibrium populations (Boltzmann distribution),
      - Context-defined initial populations,
      - Population differences between resonant upper and lower states.

    This class supports both stationary and time-dependent scenarios through inheritance.
    It handles initialization from temperature or from a relaxation Context,
    and provides unified access to population initialization logic.

    The actual intensity computation (including matrix elements, line shapes, etc.)
    is performed downstream in the spectra creator; the populator only supplies
    the population-dependent factor.
    """
    def __init__(self,
                 context: tp.Optional[contexts.BaseContext] = None,
                 init_temperature: tp.Union[float, torch.Tensor] = 293.0,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer(
            "init_temperature", torch.tensor(init_temperature, device=device, dtype=dtype)
        )
        self.context = context
        self._init_context_meta()

    def _precompute(self, res_fields, lvl_down, lvl_up, energies, vector_down, vector_up, *args, **kwargs):
        return res_fields, lvl_down, lvl_up, energies, vector_down, vector_up

    def _init_context_meta(self):
        if self.context is not None:
            if self.context.contexted_init_population:
                self.contexted = True
                self._getter_init_population = self._context_dependant_init_population
            else:
                self.contexted = False
                self._getter_init_population = self._temp_dependant_init_population
            self.time_dependant = self.context.time_dependant

        else:
            self.contexted = False
            self._getter_init_population = self._temp_dependant_init_population
            self.time_dependant = False

    def _initial_populations(
            self, energies: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            full_system_vectors: tp.Optional[torch.Tensor],
            *args, **kwargs
    ):
        """
        :param energies:
            The energies of spin states. The shape is [..., R, N], where R is number of resonance transitions
        :param lvl_down : array-like
            Indexes of energy levels of lower states from which transitions occur.
            Shape: [R], where R is number of resonance transitions
            N is the number of energy levels.
        :param lvl_up : array-like
            Indexes of energy levels of upper states to which transitions occur.
            Shape: [R], where R is number of resonance transitions
        :param full_system_vectors: Eigen vector of each level of a spin system. The shape os [..., N, N].
        For some cases it can be None
        :param args:
        :param kwargs:
        :return: initial populations
        """
        return self._getter_init_population(energies, lvl_down, lvl_up, full_system_vectors)

    def _temp_dependant_init_population(self,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs):
        return nn.functional.softmax(
            -constants.unit_converter(energies, "Hz_to_K") / self.init_temperature, dim=-1
        )

    def _context_dependant_init_population(self,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs):
        return self.context.get_transformed_init_populations(full_system_vectors, normalize=True)

    def _out_population_difference(self, populations: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor):
        """
        Calculate the population difference between transitioning energy levels.

        Parameters
        ----------
        :param populations:
             population values.
            Shape: [..., R, N] or [N], where N is the number of energy levels. R is number of resonance transitions

        :param lvl_down : array-like
            Indexes of energy levels of lower states from which transitions occur.
            Shape: [R], where R is number of resonance transitions
            N is the number of energy levels.

        :param lvl_up : array-like
            Indexes of energy levels of upper states to which transitions occur.
            Shape: [R], where R is number of resonance transitions

        :return:
        -------
            The population difference between transitioning energy levels.
        """
        if populations.dim() == 1:
            populations = populations.unsqueeze(-2)
        indexes = torch.arange(populations.shape[-2], device=populations.device)
        return populations[..., indexes, lvl_down] - populations[..., indexes, lvl_up]


class BaseTimeDepPopulator(BasePopulator):
    """
      Base class for time-dependent populators that model relaxation dynamics in time-resolved EPR.

      This class implements the common infrastructure for solving the kinetic or Liouville-von Neumann
      equations that govern the evolution of populations or the density matrix:
        dn/dt = K(t, n) · n        (population-based)
        dρ/dt = -i[H, ρ] + R[ρ]    (density matrix-based)

      Key components:
        1. **Populator**: Defines initial state and numerical strategy (this class and subclasses).
        2. **Context**: Encodes physical relaxation mechanisms (losses, spontaneous/induced transitions,
           decoherence) and their basis of definition.
        3. **Transition matrix generator**: Constructs the relaxation operator (K or R) from Context.
        4. **Solver**: Integrates the evolution equation (stationary, quasi-stationary, or adaptive ODE).

      Subclasses must implement:
        - `init_solver`: selects appropriate integrator based on time-dependence,
        - `_init_tr_matrix_generator`: builds generator for relaxation superoperator,
        - `forward`: orchestrates the full computation pipeline.
      """
    def __init__(self,
                 context: tp.Optional[contexts.BaseContext],
                 tr_matrix_generator_cls: tp.Type[matrix_generators.BaseGenerator],
                 solver: tp.Optional[tr_utils.EvolutionSolver] = None,
                 init_temperature: tp.Union[float, torch.Tensor] = 293.0,
                 difference_out: bool = False,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """
        :param context: context is a dataclass / Dict with any objects that are used to compute relaxation matrix.
        :param tr_matrix_generator_cls: class of Matrix Generator
            that will be used to compute probabilities of transitions
        :param solver: It solves the general equation dn/dt = A(n,t) @ n.

            The following solvers are available:
            - odeint_solver:  Default solver.
            It uses automatic control of time-steps. If you are not sure about the correct time-steps use it
            - stationary_rate_solver. When A does not depend on time use it.
            It just uses that in this case n(t) = exp(At) @ n0
            - exponential_solver. When A does depend on time but does not depend on n,
            It is possible to precompute A and exp(A) in all points.
            In this case the solution is n_i+1 = exp(A_idt) @ ni
            If solver is None than it will be initialized as odeint solver or stationary solver according to the context

        :param init_temperature: initial temperature. In default case it is used to find initial population

        :param difference_out: If True, the output intensity is expressed as the difference relative
               to the initial signal:
                       intensity(t) = intensity(t) - intensity(t=0).
                       This is useful for simulating differential or transient absorption spectra.

        :param device: device to compute (cpu / gpu)
        """
        super().__init__(context, init_temperature, device, dtype)
        self.solver = self.init_solver(solver)
        self.tr_matrix_generator_cls = tr_matrix_generator_cls
        self.difference_out = difference_out
        self.to(device)

    @abstractmethod
    def init_solver(self, solver: tp.Optional[tp.Callable]) -> tp.Callable:
        if solver is not None:
            return solver
        if self.time_dependant:
            return tr_utils.EvolutionSolver.odeint_solver
        else:
            return tr_utils.EvolutionSolver.stationary_rate_solver

    def _post_compute(self, time_intensities: torch.Tensor, *args, **kwargs):
        """
        :param time_intensities: The population difference between transitioning energy levels depending on time
        :return: intensity of transitions due to population difference
        """
        if self.difference_out:
            return time_intensities - time_intensities[0].unsqueeze(0)
        else:
            return time_intensities

    @abstractmethod
    def _init_tr_matrix_generator(self, *args, **kwargs) -> matrix_generators.BaseGenerator:
        """
        Function creates TransitionMatrixGenerator - it is object that can compute probabilities of transitions.

        :param args: tuple, optional.

        :param kwargs : dict, optional
        :param return:
        -------
        TransitionMatrixGenerator instance
        """
        tr_matrix_generator = self.tr_matrix_generator_cls(*args, **kwargs)
        return tr_matrix_generator