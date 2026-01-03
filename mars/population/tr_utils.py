import math

from abc import ABC, abstractmethod
import torch
from torchdiffeq import odeint

from .. import constants
import typing as tp

from . import contexts
from . import matrix_generators
from . import transform
from . import rk4


class EvolutionMatrix:
    """
    Construct full evolution matrix from energy differences and transition probabilities.
    """
    def __init__(self, res_energies: torch.Tensor, symmetry_probs: bool = True):
        """
        :param res_energies: The resonance energies. The shape is [..., N, N], where N is spin system dimension
        :param symmetry_probs: Is the probabilities of transitions are given in symmetric form. Default is True
        """
        self.energy_diff = res_energies.unsqueeze(-2) - res_energies.unsqueeze(-1)
        self.energy_diff = constants.unit_converter(self.energy_diff, "Hz_to_K")
        self.config_dim = self.energy_diff.shape[:-2]
        self._free_transform = self._prob_transform_factory(symmetry_probs)

    def _prob_transform_factory(self, symmetry_probs: bool) -> tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if symmetry_probs:
            return self._compute_boltzmann_symmetry
        else:
            return self._compute_boltzmann_complement

    def _compute_energy_factor(self, temp: torch.Tensor) -> torch.Tensor:
        denom = 1 + torch.exp(-self.energy_diff / temp)
        return torch.reciprocal(denom)

    def _compute_boltzmann_symmetry(self, temp: torch.tensor, free_probs: torch.Tensor) -> torch.Tensor:
        energy_factor = self._compute_energy_factor(temp)
        return (free_probs + free_probs.transpose(-2, -1)) * energy_factor

    def _compute_boltzmann_complement(self, temp: torch.tensor, free_probs: torch.Tensor) -> torch.Tensor:
        numerator = torch.exp(self.energy_diff / temp)
        return torch.where(free_probs == 0, free_probs.transpose(-1, -2) * numerator, free_probs)

    def __call__(self, temp: torch.tensor,
                 free_probs: tp.Optional[torch.Tensor] = None,
                 driven_probs: torch.Tensor | None = None,
                 out_probs: torch.Tensor | None = None) -> torch.Tensor:
        """
        Build full transition matrix.
        :param temp: Temperature(s).
        :param free_probs: Optional Free relaxation probabilities [..., N, N].
        :param driven_probs: Optional induced transitions [..., N, N].
        :param out_probs: Optional outgoing transition rates [..., N].
        :return: Transition matrix [..., N, N].

        Example (2-level system):

        Free relaxation (symmetric form):
            base_probs = [[0,  k'],
                          [k', 0]]

        Driven transitions:
            induced_probs = [[0,  dr1'],
                             [dr2', 0]]

        Outgoing rates:
            out_probs = [t, t]

        Resulting matrix:
            [[-2k' * exp(-(E2 - E1)/kT),   2k'],
             [ 2k' * exp(-(E2 - E1)/kT), -2k']] / (1 + exp(-(E2 - E1)/kT))

          + [[-i',  i'],
             [ i', -i']]

          - [[t, 0],
             [0, t]]
        """
        indices = torch.arange(self.energy_diff.shape[-1], device=self.energy_diff.device)
        if free_probs is not None:
            probs_matrix = self._free_transform(temp, free_probs)
            probs_matrix[..., indices, indices] = -probs_matrix.sum(dim=-2)
            transition_matrix = probs_matrix
        else:
            transition_matrix = 0

        if driven_probs is not None:
            driven_probs[..., indices, indices] = -driven_probs.sum(dim=-2)
            transition_matrix += driven_probs
        if out_probs is not None:
            transition_matrix -= torch.diag_embed(out_probs)
        return transition_matrix


class EvolutionSuper(EvolutionMatrix):
    """
    Construct full evolution superoperator from energy differences and transition probabilities.
    """
    def __init__(self, res_energies: torch.Tensor, symmetry_probs: bool = True):
        """
        :param res_energies: The resonance energies. The shape is [..., N, N], where N is spin system dimension
        :param symmetry_probs: Is the probabilities of transitions are given in symmetric form. Default is True
        """
        super().__init__(res_energies, symmetry_probs)
        self.N = res_energies.shape[-1]
        self.pop_indices = torch.arange(self.N, device=res_energies.device) * (self.N + 1)

    def _prob_transform_factory(self, symmetry_probs: bool) -> tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if symmetry_probs:
            return self._compute_boltzmann_symmetry
        else:
            return self._compute_boltzmann_complement

    def _compute_energy_factor(self, temp: torch.Tensor) -> torch.Tensor:
        denom = 1 + torch.exp(-self.energy_diff / temp)
        return torch.reciprocal(denom)

    def _compute_boltzmann_symmetry(self, temp: torch.Tensor, free_superop: torch.Tensor) -> torch.Tensor:
        """
        Apply Boltzmann scaling to population transfer rates in superoperator.

        For population transfer rates R[i,i],[j,j] (transitions between states i and j):
            R_mean[i,j] = (R[i,i],[j,j] + R[j,j],[i,i]) / 2
            R_new[i,i],[j,j] = R_mean[i,j] * energy_factor[i,j]
            R_new[j,j],[i,i] = R_mean[i,j] * energy_factor[i,j]
        Then it is corrected the decay rates because they can be changed. That is

        decay_i = R[i,i,i,i,] + sum{R[i,i,j,j]} - total depopulation of i energy level
        After correct we perform R[i,i,i,i,]_new =  -sum{R[i,i,j,j]_new} + decay_i.


        :param temp: Temperature tensor [...]
        :param free_superop: Superoperator [..., N², N²]
        :return: Scaled superoperator [..., N², N²]
        """
        N = self.N
        device = free_superop.device

        pop_indices = torch.arange(N, device=device) * (N + 1)
        pop_block = free_superop[..., pop_indices[:, None], pop_indices[None, :]]
        decay = pop_block.sum(dim=-2)  # It is decay rate from the level. It should be preserved after all correction

        pop_block = (pop_block + pop_block.transpose(-2, -1))
        energy_factor = self._compute_energy_factor(temp)

        new_superop = torch.zeros(
                (*energy_factor.shape[:-2], self.N**2, self.N**2), device=free_superop.device, dtype=free_superop.dtype
            ) + free_superop

        pop_block = pop_block * energy_factor

        diag_indices = torch.arange(N, device=device)
        mask = ~torch.eye(pop_block.shape[-1], device=pop_block.device, dtype=torch.bool)
        pop_block[..., diag_indices, diag_indices] = -(mask * pop_block).sum(dim=-2) + decay
        new_superop[..., pop_indices[:, None], pop_indices[None, :]] = pop_block
        return new_superop

    def _compute_boltzmann_complement(self, temp: torch.tensor, free_probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self,
                 temp: torch.tensor,
                 Ht: torch.Tensor,
                 free_superop: torch.Tensor | None = None,
                 driven_superop: torch.Tensor | None = None,
                 ) -> torch.Tensor:
        """
        Build full transition matrix.
        :param temp: Temperature(s).
        :param H0: Stationary Hamiltonian. The shape is  [..., N, N].
        :param Ht: Time Dependant Hamiltonian. The shape is [..., N, N].
        :param free_superop: Optional outgoing transition rates [..., N**2, N**2].
        :param driven_superop: Optional outgoing transition rates [..., N**2, N**2].
        :return: Transition matrix [..., N**2, N**2].
        """
        super_op = transform.Liouvilleator.hamiltonian_superop(Ht)
        if free_superop is not None:
            super_op = self._free_transform(temp, free_superop) + super_op
        if driven_superop is not None:
            super_op = driven_superop + super_op
        return super_op


class EvolutionSolver(ABC):
    @staticmethod
    @abstractmethod
    def odeint_solver(*args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def exponential_solver(*args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def stationary_rate_solver(*args, **kwargs):
        pass


class EvolutionPopulationSolver(EvolutionSolver):
    @staticmethod
    def odeint_solver(time: torch.Tensor, initial_populations: torch.Tensor,
                     evo: EvolutionMatrix, matrix_generator: matrix_generators.LevelBasedGenerator,
                     lvl_down: torch.Tensor, lvl_up: torch.Tensor):
        indexes = torch.arange(initial_populations.shape[-1], device=lvl_up.device)
        TIME_SCALE = 1.0
        time_scaled = time * TIME_SCALE
        def _rate_equation(t, n_flat, evo: EvolutionMatrix, matrix_generator: matrix_generators.LevelBasedGenerator):
            """
            RHS for dn/dt = M(t) n, where M depends on t through temperature.
            - t: scalar time
            - n_flat: flattened populations of shape (..., K)
            Returns dn_flat/dt of same shape.
            """
            t_seconds = t / TIME_SCALE
            M_t = evo(*matrix_generator(t_seconds))
            dn = torch.matmul(M_t, n_flat.unsqueeze(-1)).squeeze(-1)
            return dn
        sol = odeint(func=lambda t, y: _rate_equation(
                     t, y, evo, matrix_generator),
                     y0=initial_populations,
                     t=time_scaled
                     )
        return sol[..., indexes, lvl_down] - sol[..., indexes, lvl_up]

    @staticmethod
    def exponential_solver(time: torch.Tensor,
                          initial_populations: torch.Tensor,
                          evo: EvolutionMatrix, matrix_generator: matrix_generators.LevelBasedGenerator,
                          lvl_down: torch.Tensor, lvl_up: torch.Tensor):
        indexes = torch.arange(lvl_up.shape[0], device=lvl_up.device)

        dt = (time[1:] - time[:-1])
        M = evo(*matrix_generator(time))
        dt = dt[:, None, None, None, None]
        exp_M = torch.matrix_exp(M[:-1] * dt)
        size = time.size()[0]
        n = torch.zeros((size,) + initial_populations.shape, dtype=initial_populations.dtype)

        n[0] = initial_populations

        for i in range(len(time) - 1):
            current_n = n[i]  # Shape [..., K]
            next_n = torch.matmul(exp_M[i], current_n.unsqueeze(-1)).squeeze(-1)
            n[i + 1] = next_n
        return n[..., indexes, lvl_down] - n[..., indexes, lvl_up]

    @staticmethod
    def stationary_rate_solver(time: torch.Tensor,
                         initial_populations: torch.Tensor,
                         evo: EvolutionMatrix, matrix_generator: matrix_generators.LevelBasedGenerator,
                         lvl_down: torch.Tensor, lvl_up: torch.Tensor):
        M = evo(*matrix_generator(time[0]))
        eig_vals, eig_vecs = torch.linalg.eig(M)

        indexes = torch.arange(lvl_up.shape[0], device=lvl_up.device)

        intermediate = (torch.linalg.inv(eig_vecs) @ initial_populations.unsqueeze(-1).to(eig_vecs.dtype)).squeeze(-1)
        dims_to_add = M.dim() - 1
        reshape_dims = [len(time)] + [1] * dims_to_add
        time_reshaped = time.reshape(reshape_dims)
        exp_factors = torch.exp(time_reshaped * eig_vals)
        torch.mul(intermediate, exp_factors, out=exp_factors)
        eig_vecs = eig_vecs[..., indexes, lvl_down, :] - eig_vecs[..., indexes, lvl_up, :]
        return (eig_vecs.unsqueeze(0) * exp_factors).real.sum(-1)


class EvolutionRWASolver(EvolutionSolver):
    @staticmethod
    def odeint_solver(time: torch.Tensor, initial_density: torch.Tensor,
                     evo: EvolutionMatrix, matrix_generator: matrix_generators.LevelBasedGenerator,
                     lvl_down: torch.Tensor, lvl_up: torch.Tensor, detection_vector: torch.Tensor):
        indexes = torch.arange(lvl_up.shape[0], device=lvl_up.device)
        def _rate_equation(t, n_flat, evo: EvolutionMatrix, matrix_generator: matrix_generators.LevelBasedGenerator):
            """
            RHS for dn/dt = M(t) n, where M depends on t through temperature.
            - t: scalar time
            - n_flat: flattened populations of shape (..., K)
            Returns dn_flat/dt of same shape.
            """
            M_t = evo(*matrix_generator(t))
            dn = torch.matmul(M_t, n_flat.unsqueeze(-1)).squeeze(-1)
            return dn
        sol = odeint(func=lambda t, y: _rate_equation(
                     t, y, evo, matrix_generator),
                     y0=initial_density,
                     t=time
                     )
        return (detection_vector.unsqueeze(0) * sol).real.sum(dim=-1)

    @staticmethod
    def exponential_solver(time: torch.Tensor,
                          initial_density: torch.Tensor,
                          evo: EvolutionMatrix, matrix_generator: matrix_generators.LevelBasedGenerator,
                          lvl_down: torch.Tensor, lvl_up: torch.Tensor, detection_vector: torch.Tensor):

        dt = (time[1] - time[0])
        M = evo(*matrix_generator(time))
        exp_M = torch.matrix_exp(M * dt)

        size = time.size()[0]
        n = torch.zeros((size,) + initial_density.shape, dtype=initial_density.dtype)

        n[0] = initial_density

        for i in range(len(time) - 1):
            current_n = n[i]  # Shape [..., K]
            next_n = torch.matmul(exp_M[i], current_n.unsqueeze(-1)).squeeze(-1)
            n[i + 1] = next_n
        return (detection_vector.unsqueeze(0) * n).real.sum(dim=-1)

    @staticmethod
    def stationary_rate_solver(time: torch.Tensor,
                         initial_density: torch.Tensor,
                         evo: EvolutionMatrix, matrix_generator: matrix_generators.LevelBasedGenerator,
                         lvl_down: torch.Tensor, lvl_up: torch.Tensor, detection_vector: torch.Tensor):
        M = evo(*matrix_generator(time[0]))
        eig_vals, eig_vecs = torch.linalg.eig(M)

        intermediate = (torch.linalg.inv(eig_vecs) @ initial_density.unsqueeze(-1).to(eig_vecs.dtype)).squeeze(-1)

        dims_to_add = M.dim() - 1
        reshape_dims = [len(time)] + [1] * dims_to_add
        time_reshaped = time.reshape(reshape_dims)
        exp_factors = torch.exp(time_reshaped * eig_vals)
        torch.mul(intermediate, exp_factors, out=exp_factors)
        #eig_vecs = eig_vecs[..., indexes, lvl_down, :] - eig_vecs[..., indexes, lvl_up, :]
        out = torch.matmul(detection_vector.unsqueeze(-2), eig_vecs).squeeze(-2)
        return (out.unsqueeze(0) * exp_factors).real.sum(dim=-1)



class EvolutionPropagatorSolver(EvolutionSolver):
    def _get_resips(self, U_2pi: torch.Tensor, M_power: int):
        """
        :param U_2pi: torch.Tensor. Full-period evolution operator of shape [..., N^2, N^2].
        :param M_power: int Number of measurement periods.
        :return: tuple of the next data:
        -------
        resip_term_2pi : torch.Tensor
            Single-period residual term (I - U_2pi) of shape [..., N^2, N^2].
        resip_term_2pi_M : torch.Tensor
            Multi-period residual term (I - U_M) of shape [..., N^2, N^2].
        """
        U_M = torch.linalg.matrix_power(U_2pi, M_power)
        I = torch.eye(U_2pi.shape[-1], dtype=U_2pi.dtype, device=U_2pi.device)
        resip_term_2pi = I - U_2pi
        resip_term_2pi_M = I - U_M
        return resip_term_2pi, resip_term_2pi_M

    def _modify_integral_term(self, integral: torch.Tensor, U_2pi: torch.Tensor, M_power: int, d_phi: torch.Tensor):
        """
        :param integral: the integral of U(phi) * sin(phi) dphi over one period. The shape is [..., N^2, N^2]
        :param U_2pi: U(2pi). The shape is [..., N^2, N^2]
        :param M_power: int Number of measurement periods.
        :param d_phi: the integration step
        :return:
            Modified integral for the computation over M periods. That is U(phi) * sin(phi) dphi over M period s
            The shape is [..., N^2, N^2]
        """
        mean_over_measurement, resip_term_2pi_M = self._get_resips(U_2pi, M_power)
        torch.linalg.solve(mean_over_measurement, resip_term_2pi_M, out=mean_over_measurement)
        torch.matmul(integral, mean_over_measurement, out=integral)
        torch.add(integral, resip_term_2pi_M, alpha=d_phi/12, out=integral)
        integral.mul_(d_phi)
        return integral

    def _U_N_batched(self, U_2pi: torch.Tensor, powers: list[tp.Union[int, torch.Tensor]]):
        eiglev, eigbasis = torch.linalg.eig(U_2pi)
        embedings = torch.stack([torch.pow(eiglev, m) for m in powers], dim=-2)
        return eigbasis, torch.linalg.inv(eigbasis), embedings

    def _compute_out(self,
                     detective_vector: torch.Tensor,
                     integral: torch.Tensor,
                     eigen_basis: torch.Tensor,
                     time_dep_values: torch.Tensor,
                     eigen_basis_inv: torch.Tensor,
                     density_vector: torch.Tensor):
        """
        :param detective_vector: Vector form of Gx or Gy operators. The shape is [..., n^2]
        :param integral: Integral term from the equation rho(t) * sin(wt) dt. The shape is [..., n^2, n^2]
        :param eigen_basis: Eigen basis of U_2pi propagator. The shape is [..., n^2, n^2]
        :param time_dep_values: The eigen values of U_2pi propagator in time powers. The shape is [..., time_steps, n^2]
        :param eigen_basis_inv: Inversion of eigen basis of U_2pi propagator. The shape is [..., n^2, n^2]
        :param density_vector: The density at zero time in vector form. The shape is [..., n^2]
        :return:
            The output signal, the shape is [tau, ...]
        """
        temp = torch.einsum('...i,...ji->...j', density_vector, eigen_basis_inv).unsqueeze(-2)
        temp = time_dep_values * temp
        temp = torch.einsum('...ti,...ji->...tj', temp, eigen_basis)
        temp = torch.einsum('...ti,...ji->...tj', temp, integral)
        result = torch.einsum('...i,...ti->...t', detective_vector, temp)
        return -result.movedim(-1, 0).real

    def stationary_rate_solver(
            self, time: torch.Tensor, initial_density: torch.Tensor, hamiltonain_time_dep: torch.Tensor,
            superop_static: torch.Tensor, res_omega: torch.Tensor, period_time: torch.Tensor, delta_phi: torch.Tensor,
            measurement_time: torch.Tensor, n_steps: int
    ):
        """
        Solve for a signal rate under periodic driving res_omega.
        This method computes the time-dependent expectation value of an observable
        under a periodically driven quantum system using Floquet theory and
        Runge-Kutta integration.

        :param time: torch.Tensor. Time points for signal evaluation
        :param initial_density: Initial density matrix, The shape is [..., N, N]
        :param hamiltonain_time_dep: Time-dependent Hamiltonian operator, The shape is [..., N, N]
        :param superop_static: The static part of super operator. The shape is [..., N^2, N^2]
        :param res_omega: torch.Tensor, Resonance frequency at s-1
        :param period_time: Measurement period.
        :param delta_phi: Phase increment per period.
        :param measurement_time: Total measurement duration.
        :param n_steps: Number of RK4 integration steps.
        :return: Time-dependent expectation values of shape [time_steps, ...].
        """
        liouvilleator = transform.Liouvilleator

        superop_dynamic = liouvilleator.hamiltonian_superop(hamiltonain_time_dep)
        U_2pi, integral = rk4.solve_matrix_ode_rk4(
            superop_static / res_omega, superop_dynamic / res_omega, n_steps
        )

        M_power = int((measurement_time / period_time).item())
        integral = self._modify_integral_term(integral, U_2pi, M_power, delta_phi)
        powers = torch.ceil(time / period_time)
        direct, inverse, eigen_values = self._U_N_batched(U_2pi, powers)
        return self._compute_out(
            liouvilleator.vec(hamiltonain_time_dep.transpose(-2, -1)),
            integral, direct, eigen_values, inverse, liouvilleator.vec(initial_density)
        )

    @staticmethod
    def odeint_solver(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def exponential_solver(*args, **kwargs):
        raise NotImplementedError




