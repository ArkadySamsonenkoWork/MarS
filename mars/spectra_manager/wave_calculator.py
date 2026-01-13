import math
import typing as tp

import torch
import torch.nn as nn

from ..population import contexts
from .spectra_manager import compute_matrix_element, StationaryIntensityCalculator
from .. import constants


def wigner_term_square(helicity: int, num: int, theta: torch.Tensor):
    """
    Compute squared Wigner d-matrix element for EPR transition probability.

    This function calculates orientation-dependent terms for different helicity projections
    in electron paramagnetic resonance simulations.

    :param helicity: Photon helicity (+1 or -1 for circular polarization)
    :param num: Spin quantum number projection onto z-axis (-1, 0, or +1)
    :param theta: Polar angle between radiation direction and quantization axis (radians)
    :return: Squared Wigner term as torch.Tensor
    """
    if helicity == num:
        return torch.pow(torch.cos(theta / 2), 4)

    elif helicity == -num:
        return torch.pow(torch.sin(theta / 2), 4)

    else:
        return torch.pow(torch.sin(theta), 2) / 2


class PlaneWaveTerms(nn.Module):
    """
    Base module for polarization-dependent term computation for plane waves
    """
    def __init__(self, polarization: str, theta: float,
                 phi: tp.Optional[float], device: torch.device, dtype: torch.dtype):
        """
        :param polarization: Radiation polarization state. Must be one of:
                1) '+1' or '-1' for circular polarization
                2) 'un' for unpolarized radiation
                3) 'lin' for linear polarization
        :param theta: Polar angle between radiation direction and static magnetic field (radians)
        :param phi: An angle between static magnetic field and
        magnetic field of radiation for linear polarization orientation. It is used only fot linear polarization
        :param device: torch.device
        :param dtype: torch.dtype

        """
        super().__init__()
        self.register_buffer("theta", torch.tensor(theta, device=device, dtype=dtype))
        self.register_buffer("phi", torch.tensor(phi, device=device, dtype=dtype))
        self.output_method = self._parse_polarization(polarization)

    def _parse_polarization(self, polarization: str):
        if polarization == "+1":
            self.helicity = 1
            return self._circle

        elif polarization == "-1":
            self.helicity = -1
            return self._circle

        elif polarization == "un":
            return self._unpolarized

        elif polarization == "lin":
            return self._linear

        else:
            raise ValueError(
                "polarization must be '+1' or '-1' for circular polarization, 'lin' for linear and 'un' for unpolarized"
            )

    def forward(self, wave_len: tp.Optional[float] = None):
        return self.output_method(wave_len)


class PowderPlaneWaveTerms(PlaneWaveTerms):
    """
    Polarization terms calculator for disordered (powder) samples for plane wave radiation
    """
    def _circle(self, wave_len: tp.Optional[float]):
        def _xy_term(
                helicity: int, theta: torch.Tensor, phi: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            return (wigners[0] + wigners[2]) / 2

        def _z_term(
                helicity: int, theta: torch.Tensor, phi: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            return wigners[1]

        def _mixed_term(
                helicity: int, theta: torch.Tensor, phi: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            return wigners[0] - wigners[2]

        d_pl = wigner_term_square(self.helicity, 1, self.theta)
        d_zero = wigner_term_square(self.helicity, 0, self.theta)
        d_m = wigner_term_square(self.helicity, -1, self.theta)
        wigners = (d_pl, d_zero, d_m)
        return (
            _xy_term(self.helicity, self.theta, self.phi, wigners),
            _z_term(self.helicity, self.theta, self.phi, wigners),
            _mixed_term(self.helicity, self.theta, self.phi, wigners),
        )

    def _unpolarized(self, wave_len: tp.Optional[float]):
        def _xy_term(
                helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ):
            return (wigners[0] + wigners[2]) / 4

        def _z_term(
                helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ):
            return wigners[1] / 2

        def _mixed_term(
                helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ):
            return 0.0

        helicity = 1  # It can be 1 or -1 for this case. It doesn't matter
        d_pl = wigner_term_square(1, 1, self.theta)
        d_zero = wigner_term_square(1, 0, self.theta)
        d_m = wigner_term_square(1, -1, self.theta)
        wigners = (d_pl, d_zero, d_m)
        return (
            _xy_term(helicity, self.theta, self.phi, wigners),
            _z_term(helicity, self.theta, self.phi, wigners),
            _mixed_term(helicity, self.theta, self.phi, wigners),
        )

    def _linear(self, wave_len: tp.Optional[float]):
        def _xy_term(helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor):
            return torch.sin(phi).square()

        def _z_term(helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor):
            return torch.cos(phi).square() / 2

        def _mixed_term(helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor):
            return 0.0

        return (
            _xy_term(None, self.theta, self.phi),
            _z_term(None, self.theta, self.phi),
            _mixed_term(None, self.theta, self.phi),
        )


class CrystalPlaneWaveTerms(PlaneWaveTerms):
    """
    Polarization terms calculator for single-crystal or many-crystal samples for plane wave radiation
    """
    def _circle(self, wave_len: tp.Optional[float]):
        def _xy_term(
                helicity: int, theta: torch.Tensor, phi: torch.Tensor):
            return 1/4

        def _z_term(
                helicity: int, theta: torch.Tensor, phi: torch.Tensor):
            return 0.0

        def _mixed_term(
                helicity: int, theta: torch.Tensor, phi: torch.Tensor):
            return 1 if helicity == 1 else -1

        return (
            _xy_term(self.helicity, self.theta, self.phi),
            _z_term(self.helicity, self.theta, self.phi),
            _mixed_term(self.helicity, self.theta, self.phi),
        )

    def _unpolarized(self, wave_len: tp.Optional[float]):
        def _xy_term(helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor):
            return 1/4

        def _z_term(helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor):
            return 0.0

        def _mixed_term(helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor):
            return 0.0

        return (
            _xy_term(None, self.theta, self.phi),
            _z_term(None, self.theta, self.phi),
            _mixed_term(None, self.theta, self.phi)
        )

    def _linear(self, wave_len: tp.Optional[float]):
        def _xy_term(helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor):
            return torch.sin(phi).square()

        def _z_term(helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor):
            return torch.cos(phi).square() / 2

        def _mixed_term(helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor):
            return 0.0

        return (
            _xy_term(None, self.theta, self.phi),
            _z_term(None, self.theta, self.phi),
            _mixed_term(None, self.theta, self.phi)
        )


class WaveIntensityCalculator(StationaryIntensityCalculator):
    """
    Computes the intensity of transitions for general type of radiation, when the radiation has different orientation
    with respect to static magnetic field.

    Handles calculation of transition intensities based on:
    - Transition matrix elements (magnetization).
    - Level populations. Uses Boltzmann thermal populations at specified temperature
      or predefined population given in context.
    """
    def __init__(self,
                 spin_system_dim: int, disordered: bool,
                 polarization: str, theta: float, phi: tp.Optional[float] = math.pi / 2,
                 terms_computer: tp.Optional[
                     tp.Callable[
                         [tp.Any],
                         tuple[
                             tp.Union[float, torch.Tensor],
                             tp.Union[float, torch.Tensor],
                             tp.Union[float, torch.Tensor]
                         ]
                     ]
                 ] = None,
                 temperature: tp.Optional[float] = 293.0,
                 populator: tp.Optional[tp.Callable] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32
                 ):
        """
        :param spin_system_dim: The dimension of a spin system.

        :param disordered: The flag is used for powder averaging

        :param polarization:  The polarization of radiation. It should be one of the variants:
               1) '+1' or '-1' for circular polarization
               2) 'un' for unpolarized radiation
               3) 'lin' for linear polarization

        :param theta: The angle between radiation direction and stationary magnetic field

        :param phi: The angle between oscillating magnetic field and static magnetic field.
        It is used only in linear polarization. Default is None

        :param terms_computer:
            Callable that computes the polarization-dependent weight factors for the three magnetization components:
                - xy-component (transverse in-plane),
                - z-component (longitudinal),
                - mixed xy-phase term (imaginary coherence).
            The callable must accept a single optional argument (e.g., transition energy or wavelength in cm⁻¹)
            and return a 3-tuple of scalars or tensors: (w_xy, w_z, w_mixed).
            If None, a default plane-wave-based implementation is used:
                • PowderPlaneWaveTerms for disordered (powder) samples,
                • CrystalPlaneWaveTerms for single-crystal samples.

        :param temperature: The temperature of an experiment. If populator is not None it takes from it
        :param populator:
            Class that is used to compute part intensity due to population of levels. Default is None
            If it is None then it will be initialized as default calculator specific to given intensity_calculator

        :param context: Optional[context]
            The instance of BaseContext which describes the relaxation mechanism.
            It can have the initial population logic, transition between energy levels, decoherences, driven transition,
            out system transitions. For more complicated scenario the full relaxation superoperator can be used.

        :param device: torch.device
        :param dtype: torch.dtype
        """
        super().__init__(
            spin_system_dim, temperature, populator, context, disordered, device=device, dtype=dtype)
        self.terms_computer = self._init_terms_computer(terms_computer=terms_computer, disordered=disordered,
                                                        theta=theta, phi=phi, polarization=polarization,
                                                        device=device, dtype=dtype)

    def _init_terms_computer(self,
                             polarization: str, theta: float, phi: tp.Optional[float],
                             terms_computer: tp.Optional[tp.Callable], disordered: bool,
                             device: torch.device, dtype: torch.dtype
                             ):
        if terms_computer is None:
            if disordered:
                return PowderPlaneWaveTerms(
                    theta=theta, phi=phi, polarization=polarization, device=device, dtype=dtype
            )
            return CrystalPlaneWaveTerms(
                theta=theta, phi=phi, polarization=polarization, device=device, dtype=dtype
            )

        else:
            return terms_computer

    def _compute_magnitization_crystal(
            self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
            vector_down: torch.Tensor, vector_up: torch.Tensor,
            resonance_manifold: torch.Tensor, resonance_energies: torch.Tensor):
        mu_x = compute_matrix_element(vector_down, vector_up, -Gx)
        mu_y = compute_matrix_element(vector_down, vector_up, -Gy)
        mu_z = compute_matrix_element(vector_down, vector_up, -Gz)

        magnitization_xy = mu_x.square().abs() + mu_y.square().abs()
        magnitization_z = mu_z.square().abs()
        magnitization_mixed = (mu_x * mu_y.conj()).imag

        terms = self.terms_computer(constants.unit_converter(resonance_manifold, "Hz_to_cm-1"))
        out = magnitization_xy * terms[0] + magnitization_z * terms[1] + magnitization_mixed * terms[2]
        return out * (constants.PLANCK / constants.BOHR) ** 2

    def _compute_magnitization_powder(
            self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
            vector_down: torch.Tensor, vector_up: torch.Tensor,
            resonance_manifold: torch.Tensor, resonance_energies: torch.Tensor) -> torch.Tensor:
        mu_x = compute_matrix_element(vector_down, vector_up, -Gx)
        mu_y = compute_matrix_element(vector_down, vector_up, -Gy)
        mu_z = compute_matrix_element(vector_down, vector_up, -Gz)

        magnitization_xy = mu_x.square().abs() + mu_y.square().abs()
        magnitization_z = mu_z.square().abs()
        magnitization_mixed = (mu_x * mu_y.conj()).imag

        terms = self.terms_computer(resonance_manifold)
        out = magnitization_xy * terms[0] + magnitization_z * terms[1] + magnitization_mixed * terms[2]
        return out * (constants.PLANCK / constants.BOHR) ** 2

    def compute_intensity(
            self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
            vector_down: torch.Tensor, vector_up: torch.Tensor,
            lvl_down: torch.Tensor, lvl_up: torch.Tensor, resonance_energies: torch.Tensor,
            resonance_manifold: torch.Tensor, full_system_vectors: tp.Optional[torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        """
        Compute  EPR intensities under polarized radiation
        :param Gx, Gy, Gz: Zeeman operator components
        :param vector_down, vector_up: Eigenvectors of lower/upper states
        :param ...: Other parameters (unused here but kept for interface consistency)
        :return: Magnetization-squared term, shape [...]
        """

        intensity = self.populator(resonance_energies, lvl_down, lvl_up, full_system_vectors, *args, **kwargs) * (
                self._compute_magnitization(Gx, Gy, Gz, vector_down, vector_up,
                                            resonance_manifold, resonance_energies)
        )
        return intensity
