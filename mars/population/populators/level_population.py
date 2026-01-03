import torch
import typing as tp
import copy

from .. import tr_utils
from .. import matrix_generators
from .. import contexts
from . import core


class LevelBasedPopulator(core.BaseTimeDepPopulator):
    """
    LevelBasedPopulator which describes relaxation via relaxation between energy levels.
    1) Populator itself. Populator determines the population of initial states and computational logic.
    2) Context - describing the relaxation probabilities and their transformations
    3) Optionally it is possible to set matrix_generators which returns the relaxation matrices
    """
    def __init__(self,
                 context: tp.Optional[contexts.BaseContext],
                 tr_matrix_generator_cls: tp.Type[matrix_generators.BaseGenerator]
                 = matrix_generators.LevelBasedGenerator,
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

        :param device: device to compute (cpu / gpu)
        """
        super().__init__(context, tr_matrix_generator_cls, solver, init_temperature, difference_out, device, dtype)

    def init_solver(self, solver: tp.Optional[tp.Callable]) -> tp.Callable:
        if solver is not None:
            return solver
        if self.time_dependant:
            return tr_utils.EvolutionPopulationSolver.odeint_solver
        else:
            return tr_utils.EvolutionPopulationSolver.stationary_rate_solver

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
        pass

    def _init_tr_matrix_generator(self,
                                  time: torch.Tensor,
                                  res_fields: torch.Tensor,
                                  lvl_down: torch.Tensor,
                                  lvl_up: torch.Tensor, energies: torch.Tensor,
                                  vector_down: torch.Tensor,
                                  vector_up: torch.Tensor,
                                  full_system_vectors: tp.Optional[torch.Tensor],
                                  *args, **kwargs) -> matrix_generators.BaseGenerator:
        """
        Function creates TransitionMatrixGenerator - it is object that can compute probabilities of transitions.
        ----------
        :param time:
            Time points of measurements.

        :param res_fields:
            Resonance fields of transitions.
            Shape: [..., M], where M is the number of resonance energies.

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param energies:
            The energies of spin states. The shape is [..., N]

        :param vector_down:
            Eigenvectors of the lower energy states. The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param vector_up:
            Eigenvectors of the upper energy states.The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param full_system_vectors:
            Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
            where M is number of transitions, N is number of levels
            For some cases it can be None. The parameter of the creator 'full_system_vectors_flag == True'
            make the creator to compute these vectors

        :param args: tuple, optional.
        If the resfield algorithm returns full_system_vectors the full_system_vectors = args[0]

        :param kwargs : dict, optional

        :param return:
        -------
        TransitionMatrixGenerator instance
        """
        return self.tr_matrix_generator_cls(
            context=self.context,
            init_temperature=self.init_temperature,
            full_system_vectors=full_system_vectors
        )


class T1Populator(LevelBasedPopulator):
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
        population = self._getter_init_population(energies, lvl_down, lvl_up, full_system_vectors)
        out_populations = copy.deepcopy(population)
        indexes = torch.arange(energies.shape[-2], device=energies.device)
        out_populations[..., indexes, lvl_down] = population[..., indexes, lvl_up]
        out_populations[..., indexes, lvl_up] = population[..., indexes, lvl_down]
        return out_populations