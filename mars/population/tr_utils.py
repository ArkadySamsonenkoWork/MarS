import math

from abc import ABC, abstractmethod
import torch
from torchdiffeq import odeint

from .. import constants
import typing as tp

from . import matrix_generators
from . import transform
from . import rk4


class EvolutionBase(ABC):
    """Base class for time evolution implementations.

    This class defines the common interface for time evolution strategies.
    Subclasses must implement the ``evolve`` method to describe how a state
    or operator evolves over time under a given Hamiltonian or generator.
    """

    def __init__(self, res_energies: torch.Tensor, symmetry_probs: bool = True):
        """
        :param res_energies: The resonance energies.

        The shape is [..., N, N], where N is spin system dimension
        :param symmetry_probs: Is the probabilities of transitions are given in symmetric form. Default is True
        """
        self.energy_diff = res_energies.unsqueeze(-2) - res_energies.unsqueeze(-1)
        self.energy_diff = constants.unit_converter(self.energy_diff, "Hz_to_K")
        self.config_dim = self.energy_diff.shape[:-2]
        self._free_transform = self._prob_transform_factory(symmetry_probs)
        self.spin_system_dim = res_energies.shape[-1]

    def _prob_transform_factory(self, symmetry_probs: bool) -> tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Selects the Boltzmann transformation function based on input convention.

        :param symmetry_probs: Boolean flag.
        :return: Callable implementing either symmetric or asymmetric thermal scaling.
        """
        if symmetry_probs:
            return self._compute_boltzmann_symmetry
        else:
            return self._compute_boltzmann_complement

    def _compute_energy_factor(self, temp: torch.Tensor) -> torch.Tensor:
        """
        Compute Boltzmann factor denominator: 1 / (1 + exp(-ΔE / k_B T)).

        :param temp: Temperature tensor.
        :return: Tensor of same shape as energy_diff '[..., N, N]'.
        """
        denom = 1 + torch.exp(-self.energy_diff / temp)
        return torch.reciprocal(denom)

    @abstractmethod
    def _compute_boltzmann_symmetry(self, temp: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        Modifies the matrix so that it complies with the principle of detailed balance.

        :param temp: Temperature in K
        :param matrix: Matrix which should be modified
        :return: The modified matrix
        """
        pass

    @abstractmethod
    def _compute_boltzmann_complement(self, temp: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        Modifies the matrix so that it complies with the principle of detailed balance.

        Use another approach compared to '_compute_boltzmann_symmetry'

        :param temp: Temperature in K
        :param matrix: Matrix which should be modified
        :return: The modified matrix
        """
        pass

    @abstractmethod
    def __ceil__(self, *args, **kwargs) -> torch.Tensor:
        """
        :param args:

        :param kwargs:
        :return: The matrix or superoperator which used for the time-equation solution
        """


class EvolutionMatrix(EvolutionBase):
    """
    Builds a transition rate matrix that describes how populations of energy levels change over time.

    due to relaxation processes.

    The matrix combines three types of contributions:
      - **Free (spontaneous) transitions**: bidirectional probabilities
        between pairs of levels that obey thermal equilibrium
        at a given temperature. These are adjusted using the Boltzmann factor based on energy differences.
      - **Driven transitions**: external or non-equilibrium transitions that are
        not constrained by thermal balance and are added as-is.
      - **Outgoing losses**: irreversible decay from individual levels (e.g., phosphorescence), which
        reduces total population and appears as negative diagonal terms.

    The input energies are expected to be eigenvalues of the spin Hamiltonian. Energy differences
    are automatically converted from frequency units (Hz) to temperature-compatible units.

    The resulting matrix has zero column sums, ensuring conservation of total probability when
    no outgoing losses are present.
    """

    def _compute_boltzmann_symmetry(self, temp: torch.tensor, free_probs: torch.Tensor) -> torch.Tensor:
        """
        Apply detailed balance to symmetric mean rates:

            K_ij = 2 * free_probs[i,j] / (1 + exp(-ΔE_ij / k_B T)),
            K_ji = 2 * free_probs[i,j] * exp(-ΔE_ij / k_B T) / (1 + exp(-ΔE_ij / k_B T)).

        :param temp: Temperature.
        :param free_probs: Symmetric matrix where free_probs[i,j] = free_probs[j,i] = k'_ij.
        :return: Asymmetric rate matrix satisfying K_ij / K_ji = exp(-ΔE_ij / k_B T).
        """
        energy_factor = self._compute_energy_factor(temp)
        return (free_probs + free_probs.transpose(-2, -1)) * energy_factor

    def _compute_boltzmann_complement(self, temp: torch.tensor, free_probs: torch.Tensor) -> torch.Tensor:
        """
        Apply detailed balance to symmetric mean rates:

            K_ij =  K_ji * exp(-ΔE_ij / k_B T)
            K_ji = K_ji

        :param temp: Temperature.
        :param free_probs: Symmetric matrix where free_probs[i,j] = free_probs[j,i] = k'_ij.
        :return: Asymmetric rate matrix satisfying K_ij / K_ji = exp(-ΔE_ij / k_B T).
        """
        numerator = torch.exp(self.energy_diff / temp)
        return torch.where(free_probs == 0, free_probs.transpose(-1, -2) * numerator, free_probs)

    def __call__(self, temp: torch.tensor,
                 free_probs: tp.Optional[torch.Tensor] = None,
                 driven_probs: tp.Optional[torch.Tensor] = None,
                 out_probs: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
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

        Outgoing transitions:
            out_probs = [o, o]

        Resulting matrix:
            [[-2k' * exp(-(E2 - E1)/kT),   2k'],
             [ 2k' * exp(-(E2 - E1)/kT), -2k']] / (1 + exp(-(E2 - E1)/kT))

          + [[-i',  i'],
             [ i', -i']]

          - [[o, 0],
             [0, o]]
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


class EvolutionSuper(EvolutionBase):
    """
    Builds a full evolution superoperator for density-matrix dynamics.

    The superoperator combines:
      - **Coherent evolution** from a given Hamiltonian (converted internally to Liouville form),
      - **Thermal relaxation**, which is corrected to satisfy detailed balance at the specified temperature,
      - **Driven relaxation**, which is added without thermal correction.

    Only the parts of the relaxation superoperator that correspond to population exchange
    between energy levels are modified to enforce thermal equilibrium. All other elements,
    including those affecting coherences, are left unchanged.

    The class assumes that all input superoperators (thermal and driven) are already expressed
    in the eigenbasis of the Hamiltonian.

    Diagonal decay rates (total depopulation from each level) are preserved during thermal correction
    to maintain physical consistency of the relaxation model.
    """
    def __init__(self, res_energies: torch.Tensor, symmetry_probs: bool = True):
        """
        :param res_energies: The resonance energies.

        The shape is [..., N, N], where N is spin system dimension
        :param symmetry_probs: Is the probabilities of transitions are given in symmetric form. Default is True
        """
        super().__init__(res_energies, symmetry_probs)
        self.pop_indices = torch.arange(
            self.spin_system_dim, device=res_energies.device) * (self.spin_system_dim + 1)

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
        :param free_superop: Superoperator [..., N^2, N^2]
        :return: Scaled superoperator [..., N^2, N^2]
        """
        device = free_superop.device

        pop_indices = torch.arange(self.spin_system_dim, device=device) * (self.spin_system_dim + 1)
        pop_block = free_superop[..., pop_indices[:, None], pop_indices[None, :]]
        decay = pop_block.sum(dim=-2)  # It is decay rate from the level. It should be preserved after all correction

        pop_block = (pop_block + pop_block.transpose(-2, -1))
        energy_factor = self._compute_energy_factor(temp)

        new_superop = torch.zeros(
                (*energy_factor.shape[:-2], self.spin_system_dim**2, self.spin_system_dim**2),
            device=free_superop.device, dtype=free_superop.dtype) + free_superop

        pop_block = pop_block * energy_factor

        diag_indices = torch.arange(self.spin_system_dim, device=device)
        mask = ~torch.eye(pop_block.shape[-1], device=pop_block.device, dtype=torch.bool)
        pop_block[..., diag_indices, diag_indices] = -(mask * pop_block).sum(dim=-2) + decay
        new_superop[..., pop_indices[:, None], pop_indices[None, :]] = pop_block
        return new_superop

    def _compute_boltzmann_complement(self, temp: torch.tensor, free_probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self,
                 temp: torch.tensor,
                 H: torch.Tensor,
                 free_superop: tp.Optional[torch.Tensor] = None,
                 driven_superop: tp.Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:
        """
        Build full superoperator from Hamiltonian and relaxation superoperator:

            Then 'drho / dt = R_full [rho]'

        :param temp: Temperature(s).
        :param H: Hamiltonian operator. The shape is [..., N, N].
        :param free_superop: The part of the superoperator that will be transformed to satisfy the detailed balance.
            The shape is '[..., N**2, N**2]'.
        :param driven_superop: The part of the superoperator that will be saved without transformation.
            The shape is '[..., N**2, N**2]'.
        :return: Transition matrix [..., N**2, N**2].
        """
        super_op = transform.Liouvilleator.hamiltonian_superop(H)

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
    """
    Solvers for population-only rate equations.

    These methods evolve a vector of level populations under a time-dependent or time-independent
    transition rate matrix and return the difference in populations between two specified levels,
    which corresponds to the observable EPR signal intensity.

    In general it solves equation:
    dN / dt = K(N, t) @ N
    """
    @staticmethod
    def odeint_solver(time: torch.Tensor, initial_populations: torch.Tensor,
                     evo: EvolutionMatrix, matrix_generator: matrix_generators.LevelBasedGenerator,
                     lvl_down: torch.Tensor, lvl_up: torch.Tensor):
        """
        Solve population dynamics using an adaptive ODE integrator (torchdiffeq.odeint).

        Suitable when the rate matrix changes with time (e.g., due to time-dependent temperature).
        Returns the signal as the difference between populations of upper and lower levels.
        :param time: Time points for evaluation, shape [T].
        :param initial_populations: Initial population vector, shape [..., N].
        :param evo: EvolutionMatrix instance that builds the full rate matrix from generator output.
        :param matrix_generator: Generator that provides thermal, driven, and loss rates at any time.
        :param lvl_down: Indices of lower energy levels involved in the observed transitions.
        :param lvl_up: Indices of upper energy levels involved in the observed transitions.
        :return: Signal intensity over time, shape [T, ..., R], where R is the number of transitions.
        """
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
        """
        Solve population dynamics by piecewise matrix exponentiation.

        It assumes the rate matrix can be considered as quasi-stationery in the interval [t_i, t_{i+1}].
        It can be quicker than ODE solution because allows to compute rate matrix in all time points simultaneously

        :param time: Time points, shape [T].
        :param initial_populations: Initial population vector, shape [..., N].
        :param evo: EvolutionMatrix instance.
        :param matrix_generator: Generator evaluated at each time point.
        :param lvl_down: Indices of lower levels.
        :param lvl_up: Indices of upper levels.
        :return: Signal intensity over time, shape [T, ..., R].
        """
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
        """
        Solve population dynamics analytically when the rate matrix is constant in time.

        Uses eigen-decomposition of the rate matrix to compute exp(M t) efficiently.
        Fastest method for time-independent relaxation.
        :param time: Time points, shape [T].
        :param initial_populations: Initial population vector, shape [..., N].
        :param evo: EvolutionMatrix instance.
        :param matrix_generator: Generator evaluated only at time[0].
        :param lvl_down: Indices of lower levels.
        :param lvl_up: Indices of upper levels.
        :return: Signal intensity over time, shape [T, ..., R].
        """
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
    """
    Solvers for density-matrix evolution under the Rotating Wave Approximation (RWA).

    These methods evolve the full density matrix but assume the Hamiltonian and relaxation
    are compatible with RWA constraints (isotropic g-tensor, circular polarization, etc.).
    The observable is computed as the expectation value of a detection operator (e.g., Gx or Gy).

    In general this solver is similar to the EvolutionPopulationSolver, but operates not with population vector
    but with denisty matrix
    """
    @staticmethod
    def odeint_solver(time: torch.Tensor, initial_density: torch.Tensor,
                     evo: EvolutionMatrix, matrix_generator: matrix_generators.LevelBasedGenerator,
                     detection_vector: torch.Tensor):
        """
        Solve density-matrix dynamics using an adaptive ODE integrator.

        The detection operator is provided in vectorized form (Liouville space).
        :param time: Time points, shape [T].
        :param initial_density: Initial density matrix flattened to vector, shape [..., N^2].
        :param evo: EvolutionMatrix
        :param matrix_generator: Generator that returns relaxation data at any time.
        :param detection_vector: Vectorized operator for signal detection, shape [..., N^2].
        :return: Signal intensity over time, shape [T, ..., R].
        """
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
                          detection_vector: torch.Tensor):
        """
        Solve density-matrix dynamics by piecewise matrix exponentiation.

        It assumes the rate matrix can be considered as quasi-stationery in the interval [t_i, t_{i+1}].
        It can be quicker than ODE solution because allows to compute rate matrix in all time points simultaneously

        :param time: Time points, shape [T].
        :param initial_density: Initial state in vectorized form, shape [..., N²].
        :param evo: EvolutionMatrix used to assemble the superoperator.
        :param matrix_generator: Generator evaluated at each time.
        :param detection_vector: Detection operator in vectorized form, shape [..., N²].
        :return: Signal intensity over time, shape [T, ..., R].
        """
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
                         evo: EvolutionMatrix,
                         matrix_generator: matrix_generators.LevelBasedGenerator,
                         detection_vector: torch.Tensor):
        """
        Analytical solution for time-independent Liouville superoperator under RWA.

        Uses eigen-decomposition of the superoperator for fast evaluation.
        :param time: Time points, shape [T].
        :param initial_density: Initial state in vectorized form, shape [..., N²].
        :param evo: EvolutionMatrix.
        :param matrix_generator: Evaluated only once at time[0].
        :param detection_vector: Detection operator in vectorized form, shape [..., N²].
        :return: Signal intensity over time, shape [T, ..., R].
        """
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
    """
    Solver for full time-resolved EPR signals using construction of Propagator.

    Uses Floquet theory and numerical integration over one microwave period to compute
    the stroboscopic evolution propagator. Supports arbitrary g-anisotropy and general
    relaxation superoperators. The signal is obtained by integrating the density matrix
    against a sinusoidal detection function over the measurement window.
    """
    def _get_resips(self, U_2pi: torch.Tensor, M_power: int):
        """
        :param U_2pi: torch.Tensor.

        Full-period evolution operator of shape [..., N^2, N^2].
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

    def _modify_integral_term(
            self, integral: torch.Tensor, U_2pi: torch.Tensor, M_power: int, d_phi: torch.Tensor):
        """
        :param integral: the integral of U(phi) * sin(phi) dphi over one period.

        The shape is [..., N^2, N^2]
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

    def _modify_integral_term_single_period(
            self, integral: torch.Tensor, U_2pi: torch.Tensor, M_power: None, d_phi: torch.Tensor):
        """
        If number of measurement periods equel to 1, then there are no need to solve some parts of computations.

        Then this function is used

        :param integral: the integral of U(phi) * sin(phi) dphi over one period. The shape is [..., N^2, N^2]
        :param U_2pi: U(2pi). The shape is [..., N^2, N^2]
        :param M_power: int Number of measurement periods. For this case it is None
        :param d_phi: the integration step
        :return:
            Modified integral for the computation over M periods. That is U(phi) * sin(phi) dphi over M period s
            The shape is [..., N^2, N^2]
        """
        I = torch.eye(U_2pi.shape[-1], dtype=U_2pi.dtype, device=U_2pi.device)
        resip_term_2pi_M = I - U_2pi
        torch.add(integral, resip_term_2pi_M, alpha=d_phi/12, out=integral)
        integral.mul_(d_phi)
        return integral

    def _U_N_batched(self, U_2pi: torch.Tensor, powers: tp.Union[list[int], torch.Tensor]):
        eigvel, eigbasis = torch.linalg.eig(U_2pi)
        embedings = torch.stack([torch.pow(eigvel, m) for m in powers], dim=-2)
        return eigbasis, torch.linalg.pinv(eigbasis), embedings

    def _compute_out(self,
                     detective_vector: torch.Tensor,
                     integral: torch.Tensor,
                     eigen_basis: torch.Tensor,
                     time_dep_values: torch.Tensor,
                     eigen_basis_inv: torch.Tensor,
                     density_vector: torch.Tensor):
        """
        Computes the time-domain signal using a spectral decomposition of the time evolution operator.

        The signal is computed as:
            signal = detective_vector @ (integral @ rho(t))
        where the time-evolved density vector rho(t) is expressed in the eigenbasis of the U_2pi propagator:
            rho(t) = eigen_basis @ diag(time_dep_values[t]) @ eigen_basis_inv @ density_vector

        This avoids explicit matrix exponentials by working directly with precomputed eigenvalues
        (time_dep_values) and eigenvectors (eigen_basis and its inverse).

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
        result = -result.movedim(-1, 0).real
        return result

    def stationary_rate_solver(
            self, time: torch.Tensor, initial_density: torch.Tensor, hamiltonain_time_dep: torch.Tensor,
            superop_static: torch.Tensor, res_omega: torch.Tensor, period_time: torch.Tensor, delta_phi: torch.Tensor,
            measurement_time: tp.Optional[torch.Tensor], n_steps: int
    ):
        """
        Solve for a signal rate under periodic driving res_omega.

        This method computes the time-dependent expectation value of an observable
        under a periodically driven system using Floquet theory and
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
        if measurement_time is not None:
            M_power = int(torch.ceil(measurement_time / period_time).item())
            integral = self._modify_integral_term(integral, U_2pi, M_power, delta_phi)

        else:
            integral = self._modify_integral_term_single_period(integral, U_2pi, None, delta_phi)
        powers = torch.ceil(time / period_time)
        direct, inverse, eigen_values = self._U_N_batched(U_2pi, powers)
        """
        Un = torch.stack([torch.matrix_power(U_2pi, int(m.item())) for m in powers], dim=-3).

        rho_t = Un @ liouvilleator.vec(initial_density).unsqueeze(-2).unsqueeze(-1)
        rho_t = rho_t.squeeze()
        sum_signal = integral.unsqueeze(-3) @ rho_t.unsqueeze(-1)
        sum_signal = sum_signal.squeeze()
        detect = torch.einsum(
            "...i,...ti->...t", liouvilleator.vec(hamiltonain_time_dep.transpose(-2, -1)), rho_t)
        detect = -detect.movedim(-1, 0).real
        return detect
        """
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




