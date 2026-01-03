from abc import ABC, abstractmethod
import math

import torch
import typing as tp

from .. import tr_utils
from .. import transform
from .. import matrix_generators
from ... import constants
from .. import contexts
from . import core
from .. import rk4

def transform_to_complex(vector):
    if vector.dtype == torch.float32:
        return vector.to(torch.complex64)
    elif vector.dtype == torch.float64:
        return vector.to(torch.complex128)
    else:
        return vector


class BaseDensityPopulator(core.BaseTimeDepPopulator):
    def __init__(self,
                 omega_intensity: tp.Optional[tp.Union[torch.Tensor, float]] = 1e2,
                 context: tp.Optional[contexts.BaseContext] = None,
                 tr_matrix_generator_cls: tp.Type[matrix_generators.BaseGenerator] =
                 matrix_generators.DensityRWAGenerator,
                 solver: tp.Optional[tr_utils.EvolutionSolver] = None,
                 init_temperature: tp.Union[float, torch.Tensor] = 293.0,
                 difference_out: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        super().__init__(context, tr_matrix_generator_cls, solver, init_temperature, difference_out, device, dtype)
        self.register_buffer(
            "two_pi", torch.tensor(math.pi * 2, device=device, dtype=dtype)
        )
        self.register_buffer("omega_intensity", torch.tensor(omega_intensity))
        self.liouvilleator = transform.Liouvilleator

    def init_solver(self, solver: tp.Optional[tp.Callable]) -> tp.Callable:
        if solver is not None:
            return solver
        if self.time_dependant:
            return tr_utils.EvolutionRWASolver.odeint_solver
        else:
            return tr_utils.EvolutionRWASolver.stationary_rate_solver

    def _init_context_meta(self):
        if self.context is not None:
            if self.context.contexted_init_population:
                self.contexted = True
                self._getter_init_density = self._context_dependant_init_density
            else:
                self.contexted = False
                self._getter_init_density = self._temp_dependant_init_density
            self.time_dependant = self.context.time_dependant

        else:
            self.contexted = False
            self._getter_init_density = self._temp_dependant_init_density
            self.time_dependant = False

    def _get_initial_Hamiltonian(self, energies: torch.Tensor):
        """
        :param energies: the eigen energies of energy levels at Hz. The shpae is [..., M, N],
        where N is number of levels, M is number of transition
        :return:
        """

        return torch.diag_embed(energies)

    def _initial_density(
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
        return self._getter_init_density(energies, lvl_down, lvl_up, full_system_vectors)

    def _temp_dependant_init_density(self,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs):
        populations = torch.nn.functional.softmax(
            -constants.unit_converter(energies, "Hz_to_K") / self.init_temperature, dim=-1
        )
        return transform_to_complex(torch.diag_embed(populations, dim1=-1, dim2=-2))

    def _context_dependant_init_density(self,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs):
        return self.context.get_transformed_init_density(full_system_vectors)

    def _transofrm_to_eigenbasis(self, full_basis, args_matrix):
        out_matrix = []
        for matrix in args_matrix:
            out_matrix.append(full_basis.conj().transpose(-1, -2) @ matrix @ full_basis)
        return out_matrix

    def _init_tr_matrix_generator(self,
                                  time: torch.Tensor,
                                  res_fields: torch.Tensor,
                                  lvl_down: torch.Tensor,
                                  lvl_up: torch.Tensor, energies: torch.Tensor,
                                  vector_down: torch.Tensor,
                                  vector_up: torch.Tensor,
                                  full_system_vectors: tp.Optional[torch.Tensor],
                                  resonance_frequency: torch.Tensor, H0: torch.Tensor, Gz: torch.Tensor, Ht: torch.Tensor,
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

        :param resonance_frequency: Resonance frequency of the spin transition, in hertz (Hz).
        Scalar value (shape: `[]`).
        :param H0: Static (time-independent) part of the spin Hamiltonian, expressed in hertz (Hz).
        :param Ht: Time-dependent (oscillating) component of the Hamiltonian, given in angular
        frequency units (Hz / 2π).



        :param args: tuple, optional.
        If the resfield algorithm returns full_system_vectors the full_system_vectors = args[0]

        :param kwargs : dict, optional

        :param return:
        -------
        TransitionMatrixGenerator instance
        """
        shift = - Gz * constants.unit_converter(self.two_pi * resonance_frequency, "Hz_to_T_e") / 2.000
        tr_matrix_generator = self.tr_matrix_generator_cls(context=self.context,
                                                           stationary_hamiltonian=H0 + shift + Ht,
                                                           lvl_down=lvl_down, lvl_up=lvl_up,
                                                           init_temperature=self.init_temperature,
                                                           full_system_vectors=full_system_vectors,
                                                           )
        return tr_matrix_generator

    def forward(self,
                time: torch.Tensor, res_fields: torch.Tensor,
                lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                energies: torch.Tensor, vector_down: torch.Tensor,
                vector_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                resonance_frequency: torch.Tensor,
                *args, **kwargs) -> torch.Tensor:
        """
        :param time:
            Time points of measurements. The shape is [T], where T is number of time-steps

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

        :param F: Magnetic free part of spin Hamiltonian H = F + B * G. The shape is [...., N, N]
        :param Gx: x-part of Hamiltonian Zeeman Term. The shape is [...., N, N]
        :param Gy: y-part of Hamiltonian Zeeman Term. The shape is [...., N, N]
        :param Gz: z-part of Hamiltonian Zeeman Term. The shape is [...., N, N]

        :param resonance_frequency: Resonance frequency of the spin transition, in hertz (Hz).
        Scalar value (shape: `[]`).

        :param args: additional args from spectra creator.

        :param kwargs:
        :return: Part of the transition intensity that depends on the population of the levels.
        The shape is [T, ...., Tr]
        """
        res_fields, lvl_down, lvl_up, energies, vector_down, vector_up = self._precompute(res_fields,
                                                                                          lvl_down, lvl_up,
                                                                                          energies, vector_down,
                                                                                          vector_up)
        H0 = self._get_initial_Hamiltonian(energies) * self.two_pi
        F, Gx, Gy, Gz = self._transofrm_to_eigenbasis(
            full_system_vectors,
            (F.unsqueeze(-3), Gx.unsqueeze(-3), Gy.unsqueeze(-3), Gz.unsqueeze(-3))
        )
        Ht = constants.unit_converter(self.omega_intensity, "Hz_to_T_e") * Gx
        initial_density = self._initial_density(energies, lvl_down, lvl_up, full_system_vectors)
        tr_matrix_generator = self._init_tr_matrix_generator(time, res_fields,
                                                             lvl_down, lvl_up, energies, vector_down,
                                                             vector_up, full_system_vectors, resonance_frequency,
                                                             H0, Gz, Ht,
                                                             *args, **kwargs)
        evo = tr_utils.EvolutionSuper(energies)
        desnities = self.solver(
            time, self.liouvilleator.vec(initial_density),
            evo, tr_matrix_generator, lvl_down, lvl_up, self.liouvilleator.vec(Gy.transpose(-2, -1))
        )
        return desnities


class PopagatorDensityPopulator(BaseDensityPopulator):
    def __init__(self,
                 omega_intensity: tp.Optional[tp.Union[torch.Tensor, float]] = 1e2,
                 context: tp.Optional[contexts.BaseContext] = None,
                 tr_matrix_generator_cls: tp.Type[matrix_generators.BaseGenerator] =
                 matrix_generators.DensityPropagatorGenerator,
                 solver: tp.Optional[tr_utils.EvolutionSolver] = tr_utils.EvolutionPropagatorSolver(),
                 init_temperature: tp.Union[float, torch.Tensor] = 293.0,
                 difference_out: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        super().__init__(
            omega_intensity, context, tr_matrix_generator_cls, solver, init_temperature, difference_out, device, dtype
        )
        self.measurement_time = torch.tensor(40 * 1e-9, dtype=dtype, device=device)
        self.n_steps = 16

    def _init_tr_matrix_generator(self,
                                  time: torch.Tensor,
                                  res_fields: torch.Tensor,
                                  lvl_down: torch.Tensor,
                                  lvl_up: torch.Tensor, energies: torch.Tensor,
                                  vector_down: torch.Tensor,
                                  vector_up: torch.Tensor,
                                  full_system_vectors: tp.Optional[torch.Tensor],
                                  resonance_frequency: torch.Tensor, H0: torch.Tensor,
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

        :param resonance_frequency: Resonance frequency of the spin transition, in hertz (Hz).
        Scalar value (shape: `[]`).
        :param H0: Static (time-independent) part of the spin Hamiltonian, expressed in hertz (Hz).
        :param Ht: Time-dependent (oscillating) component of the Hamiltonian, given in angular
        frequency units (Hz / 2π).



        :param args: tuple, optional.
        If the resfield algorithm returns full_system_vectors the full_system_vectors = args[0]

        :param kwargs : dict, optional

        :param return:
        -------
        TransitionMatrixGenerator instance
        """
        tr_matrix_generator = self.tr_matrix_generator_cls(context=self.context,
                                                           stationary_hamiltonian=H0,
                                                           lvl_down=lvl_down, lvl_up=lvl_up,
                                                           init_temperature=self.init_temperature,
                                                           full_system_vectors=full_system_vectors,
                                                           )
        return tr_matrix_generator


    def forward(self,
                time: torch.Tensor, res_fields: torch.Tensor,
                lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                energies: torch.Tensor, vector_down: torch.Tensor,
                vector_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                resonance_frequency: torch.Tensor,
                *args, **kwargs) -> torch.Tensor:
        """
        :param time:
            Time points of measurements. The shape is [T], where T is number of time-steps

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

        :param F: Magnetic free part of spin Hamiltonian H = F + B * G. The shape is [...., N, N]
        :param Gx: x-part of Hamiltonian Zeeman Term. The shape is [...., N, N]
        :param Gy: y-part of Hamiltonian Zeeman Term. The shape is [...., N, N]
        :param Gz: z-part of Hamiltonian Zeeman Term. The shape is [...., N, N]

        :param resonance_frequency: Resonance frequency of the spin transition, in hertz (Hz).
        Scalar value (shape: `[]`).

        :param args: additional args from spectra creator.

        :param kwargs:
        :return: Part of the transition intensity that depends on the population of the levels.
        The shape is [T, ...., Tr]
        """
        res_fields, lvl_down, lvl_up, energies, vector_down, vector_up = self._precompute(res_fields,
                                                                                          lvl_down, lvl_up,
                                                                                          energies, vector_down,
                                                                                          vector_up)
        H0 = self._get_initial_Hamiltonian(energies) * self.two_pi
        F, Gx, Gy, Gz = self._transofrm_to_eigenbasis(
            full_system_vectors,
            (F.unsqueeze(-3), Gx.unsqueeze(-3), Gy.unsqueeze(-3), Gz.unsqueeze(-3))
        )

        initial_density = self._initial_density(energies, lvl_down, lvl_up, full_system_vectors)
        tr_matrix_generator = self._init_tr_matrix_generator(time, res_fields,
                                                             lvl_down, lvl_up, energies, vector_down,
                                                             vector_up, full_system_vectors, resonance_frequency, H0,
                                                             *args, **kwargs)
        evo = tr_utils.EvolutionSuper(energies)
        superop_static = evo(*tr_matrix_generator(time))

        Gx = constants.unit_converter(self.omega_intensity, "Hz_to_T_e") * Gx
        Gy = constants.unit_converter(self.omega_intensity, "Hz_to_T_e") * Gy

        res_omega = self.two_pi * resonance_frequency
        tau = 1 / resonance_frequency
        delta_phi = self.two_pi / self.n_steps
        out = torch.zeros(*(time.shape[0], *Gx.shape[:-2]), dtype=res_fields.dtype, device=res_fields.device)

        for Gt in [Gx, Gy]:
            out += self.solver.stationary_rate_solver(
                time, initial_density, Gt, superop_static,
                res_omega, tau, delta_phi, self.measurement_time, self.n_steps
            )
        return out


