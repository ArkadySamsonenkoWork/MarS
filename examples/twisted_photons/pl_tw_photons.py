import sys
import os
import typing as tp
from importlib import reload

import numpy as np
import scipy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

sys.path.append(os.path.abspath('D:\ITC\РНФ_Курганский_2024\pythonProject'))
import spin_system
import spectra_manager
import res_freq_algorithm
import constants

def wigner_term_square(helicity: int, num: int, theta: torch.Tensor):
    if helicity == num:
        return torch.pow(torch.cos(theta / 2), 4)

    elif helicity == -num:
        return torch.pow(torch.sin(theta / 2), 4)

    else:
        return torch.pow(torch.sin(theta), 2) / 2


def bessel_term_kron(num: int, kappa: tp.Optional[torch.Tensor] = None, radius: tp.Optional[torch.Tensor] = None):
    return int(not bool(num))


def bessel_term_aver(num: int, kappa: tp.Optional[torch.Tensor] = None, radius: tp.Optional[torch.Tensor] = None):
    denom = 2 * torch.pi
    arg = kappa * radius
    numerator = scipy.special.jv(num, arg) ** 2 - scipy.special.jv(num - 1, arg) * scipy.special.jv(num + 1, arg)
    numerator = numerator.to(dtype=torch.float32)
    return numerator / denom


def bessel_term_radius(num: int, kappa: tp.Optional[torch.Tensor] = None, radius: tp.Optional[torch.Tensor] = None):
    arg = kappa * radius
    numerator = scipy.special.jv(num, arg) ** 2
    numerator = numerator.to(dtype=torch.float32)
    return numerator


class WaveTwBaseTerms(nn.Module):
    def __init__(self, helicity: int, theta: float, radius: float, total_momentum: int):
        super().__init__()
        self.bessel_term = self._init_bessel_term()
        self.radius = torch.tensor(radius)
        self.helicity = helicity
        self.theta = torch.tensor(theta)
        self.total_momentum = total_momentum

    def _init_bessel_term(self):
        raise NotImplementedError

    def forward(self, wave_len: float):
        d_pl = wigner_term_square(self.helicity, 1, self.theta)
        d_zero = wigner_term_square(self.helicity, 0, self.theta)
        d_m = wigner_term_square(self.helicity, -1, self.theta)

        #wave_len = torch.tensor(wave_len)
        kappa = torch.sin(self.theta) * wave_len

        J_pl = self.bessel_term(self.total_momentum + 1, kappa, self.radius)
        J_zero = self.bessel_term(self.total_momentum, kappa, self.radius)
        J_m = self.bessel_term(self.total_momentum - 1, kappa, self.radius)

        wigners = (d_pl, d_zero, d_m)
        bessels = (J_pl, J_zero, J_m)
        return (
            self._xy_term(self.helicity, self.theta, wigners, bessels),
            self._z_term(self.helicity, self.theta, wigners, bessels),
            self._mixed_term(self.helicity, self.theta, wigners, bessels),
        )

    def _xy_term(self, helicity: int, theta: torch.Tensor,
                 wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 bessels: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        return (wigners[0] * bessels[2] + wigners[2] * bessels[0]) / 2

    def _z_term(self, helicity: int, theta: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                bessels: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        return wigners[1] * bessels[1]

    def _mixed_term(self, helicity: int, theta: torch.Tensor,
                    wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    bessels: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        return wigners[0] * bessels[2] - wigners[2] * bessels[0]


class PlaneWaveTerms(nn.Module):
    def __init__(self, helicity: int, theta: float):
        super().__init__()
        self.theta = torch.tensor(theta)
        self.helicity = helicity

    def forward(self, wave_len: tp.Optional[float] = None):
        d_pl = wigner_term_square(self.helicity, 1, self.theta)
        d_zero = wigner_term_square(self.helicity, 0, self.theta)
        d_m = wigner_term_square(self.helicity, -1, self.theta)
        wigners = (d_pl, d_zero, d_m)
        return (
            self._xy_term(self.helicity, self.theta, wigners),
            self._z_term(self.helicity, self.theta, wigners),
            self._mixed_term(self.helicity, self.theta, wigners),
        )

    def _xy_term(self, helicity: int, theta: torch.Tensor, wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        return (wigners[0] + wigners[2]) / 2

    def _z_term(self, helicity: int, theta: torch.Tensor, wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        return wigners[1]

    def _mixed_term(self, helicity: int, theta: torch.Tensor, wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        return wigners[0] - wigners[2]


class TwWaveKron(WaveTwBaseTerms):
    def _init_bessel_term(self):
        return bessel_term_kron


class TwWaveAver(WaveTwBaseTerms):
    def _init_bessel_term(self):
        return bessel_term_aver


class TwWaveRad(WaveTwBaseTerms):
    def _init_bessel_term(self):
        return bessel_term_radius


def compute_matrix_element(vector_down: torch.Tensor, vector_up: torch.Tensor, G: torch.Tensor):
    tmp = torch.matmul(G.unsqueeze(-3), vector_down.unsqueeze(-1))
    return (vector_up.conj() * tmp.squeeze(-1)).sum(dim=-1)


class TwPlIntensityCalculator(spectra_manager.StationaryIntensitiesCalculator):
    def __init__(self, spin_system_dim: int,  radius: tp.Optional[float],
                 total_momentum: tp.Optional[int], helicity: int, theta: float,
                 compute_type: str,
                 temperature: tp.Optional[float] = None,
                 populator: tp.Optional[tp.Callable] = None, tolerancy: float = 1e-14,
                 device: torch.device = torch.device("cpu")):
        super().__init__(spin_system_dim, temperature, populator, tolerancy, device=device)
        if compute_type == "plane":
            self.terms_computer = PlaneWaveTerms(theta=theta, helicity=helicity)
        elif compute_type == "tw_average":
            self.terms_computer = TwWaveAver(theta=theta, helicity=helicity, radius=radius,
                                             total_momentum=total_momentum)
        elif compute_type == "tw_center":
            self.terms_computer = TwWaveKron(theta=theta, helicity=helicity, radius=radius,
                                             total_momentum=total_momentum)
        elif compute_type == "tw_radius":
            self.terms_computer = TwWaveRad(theta=theta, helicity=helicity, radius=radius,
                                            total_momentum=total_momentum)
        else:
            raise NotImplementedError

    def _compute_magnitization_part(self, Gx, Gy, Gz, vector_down, vector_up,
                                    resonance_manifold, resonance_energies):
        mu_x = compute_matrix_element(vector_down, vector_up, -Gx)
        mu_y = compute_matrix_element(vector_down, vector_up, -Gy)
        mu_z = compute_matrix_element(vector_down, vector_up, -Gz)

        magnitization_xy = mu_x.square().abs() + mu_y.square().abs()
        magnitization_z = mu_z.square().abs()
        magnitization_mixed = (mu_x * mu_y.conj()).imag

        terms = self.terms_computer(constants.unit_converter(resonance_manifold, "Hz_to_cm-1"))
        out = magnitization_xy * terms[0] + magnitization_z * terms[1] + magnitization_mixed * terms[2]
        return out * (constants.PLANCK / constants.BOHR) ** 2

    def compute_intensity(self, Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies,
                          resonance_manifold, full_system_vectors: tp.Optional[torch.Tensor], *args, **kwargs):
        """Base method to compute intensity (to be overridden)."""
        intensity = self.populator(resonance_energies, lvl_down, lvl_up, *args, **kwargs) * (
                self._compute_magnitization_part(Gx, Gy, Gz, vector_down, vector_up,
                                                 resonance_manifold, resonance_energies)
        )
        return intensity
