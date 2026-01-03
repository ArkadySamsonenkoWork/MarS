import typing as tp

import torch
from ... import constants
from . import core
from .. import contexts

class StationaryPopulator(core.BasePopulator):
    """
    Compute the intensity of transition part depending ot population
    1) at some temperature (i -> j): (exp(-Ej/ kT) - exp(-Ei / kT)) / (sum (exp) ) if context or context.population is None
    2) at given context population algorithm if it is given
    """
    def forward(self,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs):
        """
        :param energies: energies in Hz
        :param full_system_vectors: the
        :return: population_differences
        """
        populations = self._initial_populations(energies, lvl_down, lvl_up, full_system_vectors)
        return self._out_population_difference(populations, lvl_down, lvl_up)
