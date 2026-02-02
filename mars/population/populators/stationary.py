import typing as tp

import torch
from ... import constants
from . import core
from .. import contexts


class StationaryPopulator(core.BasePopulator):
    """COmputes the population-dependent part of the transition intensity for
    stationary (CW) EPR spectra.

    The population difference between upper and lower resonant levels determines the net absorption
    (or emission) intensity. Two initialization strategies are supported:

    1. **Thermal equilibrium**: Populations follow the Boltzmann distribution at `init_temperature`:
        p_i ∝ exp(−E_i / k_B T),
       and the population difference for a transition i ← j is:
        Δp = p_j − p_i.
       This is used when no Context is provided, or when the Context does not define initial populations.

    2. **Context-defined populations**: If the Context specifies initial populations (e.g., photoexcited
       triplet sublevel polarization), these are used instead of thermal values. The populations are
       automatically transformed into the field-dependent eigenbasis using `full_system_vectors`.
    """
    def forward(self,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs) -> torch.Tensor:
        """Computes the population difference for each resonant EPR transition.

        :param energies:
            Eigenenergies of all spin states in Hz, shape [..., R, N],
            where R is the number of resonance conditions (e.g., orientations),
            and N is the number of energy levels.

        :param lvl_down:
            Indices of lower energy levels involved in transitions, shape [R].

        :param lvl_up:
            Indices of upper energy levels involved in transitions, shape [R].

        :param full_system_vectors:
            Eigenvectors of the full spin Hamiltonian, shape [..., N, N].
            Required only if initial populations are defined in a non-eigenbasis (e.g., ZFS basis)
            and Context provides them. Used to transform populations into the field-dependent eigenbasis.

        :return:
            Population differences Δp = p_upper − p_lower for each transition,
            shape [..., R], ready to be multiplied by transition matrix elements.
        """
        populations = self._initial_populations(energies, lvl_down, lvl_up, full_system_vectors)
        if self.context is not None:
            self.context.close_context()
        return self._out_population_difference(populations, lvl_down, lvl_up)
