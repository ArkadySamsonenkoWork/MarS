import sys
import random
import typing as tp
import math

import torch.nn as nn
import torch

sys.path.append("..")
import constants
import spectra_manager


def vector_to_de(vector: torch.Tensor):
    mean = vector.mean(dim=-1, keepdim=True)
    deviations = vector - mean
    D = (3 / 2) * deviations[..., 2:3]
    E = (deviations[..., 0:1] - deviations[..., 1:2]) / 2
    return torch.cat([mean, D, E], dim=-1)


def de_to_vector(de_vector: torch.Tensor):
    Dx = -1 / 3 * de_vector[..., -2] + de_vector[..., -1]  + de_vector[..., -3]
    Dy = -1 / 3 * de_vector[..., -2] - de_vector[..., -1] + de_vector[..., -3]
    Dz = 2 / 3 * de_vector[..., -2] + de_vector[..., -3]
    return torch.stack([Dx, Dy, Dz], dim=-1)


class ComponentsTransform:
    def __init__(self, freq_factor: float = 1.0, temp_factor: float = 5.0):
        self.freq_shift = torch.tensor([3.8375e+08,  2.8735e+10,  3.7846e+09])
        self.freq_std = torch.tensor([1.0216e+12, 1.6933e+12, 2.8727e+11])
        #self.freq_std = torch.tensor([1.0, 1.0, 1.0])

        self.temp_shift = torch.tensor([3.7256e+07,  4.5698e+08, 7.5622e+07])
        self.temp_std = torch.tensor([2.5768e+10, 4.4681e+10, 7.3604e+09])
        #self.temp_std = torch.tensor([1.0, 1.0, 1.0])

    def __call__(self, components, temperature):
        components = vector_to_de(components)
        components_feat_freq = components
        components_feat_temp = components / temperature.unsqueeze(0)

        components_feat_freq = components_feat_freq - self.freq_shift
        components_feat_freq = components_feat_freq / self.freq_std

        components_feat_temp = components_feat_temp - self.temp_shift
        components_feat_temp = components_feat_temp / self.temp_std

        return components_feat_freq, components_feat_temp

    def unbatch(self, components: torch.Tensor):
        components_feat_freq = components[..., :3] * self.freq_std + self.freq_shift
        components_feat_temp = components[..., 3:] * self.temp_std + self.temp_shift
        temp = components_feat_freq / components_feat_temp
        return de_to_vector(components_feat_freq), temp
        #return components_feat_freq, temp


class AnglesTransform:
    def __call__(self, angles: torch.Tensor):
        #angles = utils.get_canonical_orientations(angles.transpose(0, -2)).transpose(0, -2)
        angles[..., 0] = angles[..., 0] / (2 * torch.pi)
        angles[..., 1] = angles[..., 1] / torch.pi
        angles[..., 2] = angles[..., 2] / (2 * torch.pi)
        return angles

    def unbatch(self, angles: torch.Tensor):
        angles[..., 0] = angles[..., 0] * (2 * torch.pi)
        angles[..., 1] = angles[..., 1] * torch.pi
        angles[..., 2] = angles[..., 2] * (2 * torch.pi)
        return angles


class SpinTransform:
    def __init__(self, shift: float = 1.0, std: float = 1.0):
        self.shift = torch.tensor(shift)
        self.std = torch.tensor(std)

    def __call__(self, spins: torch.Tensor, types: torch.Tensor):
        spins_feature = torch.zeros_like(types, dtype=torch.float32)

        types_batched = types[(slice(None),) + (0,) * (types.ndim - 1)]
        mask_particles = (types_batched != 2)
        spins_feature[mask_particles, ...] = spins
        spins_feature = (spins_feature - self.shift) / self.std
        return spins_feature

    def unbatch(self, spins):
        return spins * self.std + self.shift


class ComponentsAnglesTransform:
    def __init__(self):
        self.angles_transform = AnglesTransform()
        self.components_transform = ComponentsTransform()
        self.spin_transform = SpinTransform()

    def __call__(
            self, components: torch.Tensor, temperature: torch.Tensor,
            angles: torch.Tensor, types: torch.Tensor, spins: torch.Tensor
    ):
        components_feat_freq, components_feat_temp = self.components_transform(components, temperature)
        angles = self.angles_transform(angles)
        spins = self.spin_transform(spins, types).unsqueeze(-1)
        return torch.cat((components_feat_freq, components_feat_temp, angles, spins), dim=-1)

    def unbatch(self, node_embed: torch.Tensor):
        components, temp = self.components_transform.unbatch(node_embed[..., :6])
        angles = self.angles_transform.unbatch(node_embed[..., 6:9])
        spins = self.spin_transform.unbatch(node_embed[..., 9])
        return components, temp, angles, spins




class SpecTransformField:
    def __init__(self, g_tensor_shift: float = 2.0, freq_shift: float = 20.0 * 1e9, freq_deriv: float = 20.0 * 1e9):
        self.g_tensor_shift = torch.tensor(g_tensor_shift)
        self.freq_shift = freq_shift
        self.freq_deriv = freq_deriv
        self.eps = 1e-3
        self.cut_off = 1e-2

    def __call__(self, field: torch.Tensor, freq: torch.Tensor):
        deriv_field = torch.where(field+self.eps >= self.cut_off, field, field+self.eps)
        g_tensors = (constants.PLANCK * freq.unsqueeze(-1)) / (constants.BOHR * (deriv_field+self.eps))
        # g_tensors = torch.flip(g_tensors, dims=(-1,))
        g_feature = g_tensors - self.g_tensor_shift
        freq_feature = (freq - self.freq_shift) / self.freq_deriv
        return g_feature, freq_feature

    def unbatch(self, freq_feature: torch.Tensor):
        freq = (freq_feature * self.freq_deriv + self.freq_shift)
        return freq


class SpecTransformSpecIntensity:
    def __init__(self):
        self.eps = 1e-7
    def __call__(self, spec: torch.Tensor):
        spec = spec / (torch.max(spec, dim=-1, keepdim=True)[0] + self.eps)
        return spec


class BroadTransform:
    def __init__(self, shift: float = constants.unit_converter(0.5e-1, "T_to_Hz_e"),
                 std: float = constants.unit_converter(1e-1, "T_to_Hz_e")):
        self.shift = torch.tensor([3.6182e+08, -5.1052e+04, 2.2607e+04, 1.7761e+08, 1.7803e+08])
        self.std = torch.tensor([2.0223e+08, 4.4778e+06, 2.5379e+06, 2.0138e+08, 2.0170e+08])

    def __call__(self, ham_strain: torch.Tensor, lorentz: torch.Tensor, gauss: torch.Tensor):
        ham_strain = vector_to_de(ham_strain)
        lorentz = lorentz * constants.BOHR / constants.PLANCK
        gauss = gauss * constants.BOHR / constants.PLANCK

        return (torch.cat((ham_strain, lorentz.unsqueeze(-1), gauss.unsqueeze(-1)), dim=-1) - self.shift) / self.std

    def unbatch(self, features: torch.Tensor):
        out = features * self.std + self.shift
        ham_strain = de_to_vector(out[..., :3])
        lorentz = out[..., 3] * constants.PLANCK / constants.BOHR
        gauss = out[..., 4] * constants.PLANCK / constants.BOHR

        return ham_strain, lorentz, gauss


class SpecFieldPrepare(nn.Module):
    def __init__(self,
                 min_width: float = 1e-4,
                 max_width: float = 1e-1,
                 init_interpolation_points: int = 3000,
                 max_add_points: int = 4000,
                 out_points: int = 1000,
                 spectral_width_factor: float = 5,
                 rng_generator: tp.Optional[random.Random] = None):
        """
        Prepare magnetic field, spectra, Gaussian and Lorentzian tensors using the following procedure:

        1) Each spectrum is interpolated to init_interpolation_points number of points
        2) Additional points are added to each side of the spectrum. The maximum number of points
           added per side is equal to max_add_points
        3) Gaussian and Lorentzian widths are generated using the formula:
           min_width < gauss_width + lorentz_width < min(max_width,
                                                          mean(ham_strain),
                                                          spectral_width / spectral_width_factor)
        4) Final interpolation to out_points number of points is performed

        :param min_width: Minimal linewidth in Tesla (T).
            If ham_strain in the magnetic field is greater than min_width, it is ignored
        :param max_width: Maximal linewidth in Tesla (T)
        :param init_interpolation_points: Initial number of points for spectrum interpolation
        :param max_add_points: Maximum number of points to add to each side of the spectrum
        :param out_points: Final number of output points after interpolation
        :param spectral_width_factor: Factor used to determine the maximum spectral width constraint
        :param rng_generator: Optional random number generator
        """
        super().__init__()
        self.min_width = min_width
        self.max_width = max_width
        self.out_points = out_points
        self.max_add_points = max_add_points
        self.init_interpolation_points = init_interpolation_points
        self.spectral_width_factor = spectral_width_factor

        self.post_processor = spectra_manager.PostSpectraProcessing()
        self.eps = 1e-7

        if rng_generator is None:
            self.rng = random.Random(None)
        else:
            self.rng = rng_generator

    def _generate_random_widths(self, mean_ham_strain: torch.Tensor, spectral_width: torch.Tensor):
        """
        Generate random Gauss and Lorentz widths.
        :param mean_ham_strain: Shape [...]
        :return: gauss, lorentz tensors of shape [...]
        """
        batch_shape = mean_ham_strain.shape
        device = mean_ham_strain.device

        max_total = torch.min(self.max_width - mean_ham_strain, spectral_width / self.spectral_width_factor)
        min_total = self.min_width - mean_ham_strain

        max_total = torch.clamp(max_total, min=0.0)
        min_total = torch.clamp(min_total, min=0.0)

        total_width = torch.rand(batch_shape, device=device) * (max_total - min_total) + min_total
        gauss_fraction = torch.rand(batch_shape, device=device)

        gauss = total_width * gauss_fraction
        lorentz = total_width * (1 - gauss_fraction)

        return gauss, lorentz

    def _interpolate_data_after_conv(self, fields: torch.Tensor, spec: torch.Tensor):
        batch_shape = spec.shape[:-1]
        N = spec.shape[-1]
        min_field_pos = fields[..., 0]
        max_field_pos = fields[..., -1]

        spec = torch.nn.functional.interpolate(
            spec.reshape(-1, N).unsqueeze(1),
            size=self.out_points,
            mode='linear',
            align_corners=True
        ).squeeze(1).reshape(*batch_shape, self.out_points)

        steps = torch.linspace(0, 1, self.out_points, device=spec.device, dtype=spec.dtype)
        fields = steps * (max_field_pos - min_field_pos).unsqueeze(-1) + min_field_pos.unsqueeze(-1)
        return fields, spec

    def _init_data_to_covolution(self, min_field_pos: torch.Tensor, max_field_pos: torch.Tensor, spec: torch.Tensor):
        batch_shape = spec.shape[:-1]
        N = spec.shape[-1]
        add_points_right = self.rng.randint(0, self.max_add_points)
        spectral_width = max_field_pos - min_field_pos
        field_step = spectral_width / (N - 1)
        max_points_left = torch.min(min_field_pos / field_step)


        max_points_left = max(int(max_points_left.item()) - 1, 0)
        add_points_left = min(self.rng.randint(0, self.max_add_points), max_points_left)

        target_points = self.init_interpolation_points + add_points_left + add_points_right

        spec_flat = torch.nn.functional.interpolate(
            spec.reshape(-1, N).unsqueeze(1),
            size=self.init_interpolation_points,
            mode='linear',
            align_corners=True
        ).squeeze(1)

        field_step = spectral_width / (self.init_interpolation_points - 1)
        min_field_pos = min_field_pos - field_step * add_points_left
        max_field_pos = max_field_pos + field_step * add_points_right
        new_spec = torch.zeros((math.prod(batch_shape), target_points), device=spec.device, dtype=spec.dtype)

        new_spec[..., add_points_left: target_points - add_points_right] = spec_flat
        new_spec = new_spec.reshape(*batch_shape, target_points)

        steps = torch.linspace(0, 1, target_points, device=spec.device, dtype=spec.dtype)
        fields = steps * (max_field_pos - min_field_pos).unsqueeze(-1) + min_field_pos.unsqueeze(-1)

        return fields, new_spec

    def forward(
            self,
            min_field_pos: torch.Tensor,
            max_field_pos: torch.Tensor,
            spec: torch.Tensor,
            ham_strain: torch.Tensor
    ) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param min_field_pos: Minimum position of the magnetic field. Shape: [...]
        :param max_field_pos: Maximum position of the magnetic field. Shape: [...]
        :param spec: Magnetic resonance spectra. Shape: [..., N], where N is the number of initial points
        :param ham_strain: Hamiltonian strain. Shape: [...]
        :return: Tuple containing:
            - field (torch.Tensor): Output magnetic field in Tesla (T). Shape: [..., out_points]
            - spec (torch.Tensor): Output EPR spectrum in arbitrary units. Shape: [..., out_points]
            - gauss (torch.Tensor): Gaussian linewidth in Tesla (T). Shape: [...]
            - lorentz (torch.Tensor): Lorentzian linewidth in Tesla (T). Shape: [...]
        """
        mean_ham_strain = torch.mean(ham_strain, dim=-1) * constants.PLANCK / constants.BOHR
        spectral_width = max_field_pos - min_field_pos
        gauss, lorentz = self._generate_random_widths(mean_ham_strain, spectral_width)

        field, spec = self._init_data_to_covolution(min_field_pos, max_field_pos, spec)
        spec = self.post_processor(gauss, lorentz, field, spec)
        field, spec = self._interpolate_data_after_conv(field, spec)
        spec = spec / (torch.max(abs(spec), dim=-1, keepdim=True)[0] + self.eps)
        return field, spec, gauss, lorentz


class SpectraDistortion(nn.Module):
    def __init__(self,
                 noise_max_level: float = 0.1,
                 baseline_quadratic: float = 0.05,
                 baseline_linear: float = 0.05,
                 baseline_constant: float = 0.05,
                 correct_baseline: bool = True,
                 baseline_points: int = 20
                 ):
        """
        Applies distortions to EPR spectra including baseline drift and noise.

        This module simulates experimental artifacts commonly found in magnetic resonance spectroscopy:
        - Quadratic baseline drift (field-dependent instrumental offset)
        - Gaussian noise with variable amplitude

        The baseline is modeled as a second-order polynomial in normalized field coordinates:
            baseline(x) = a*x² + b*x + c
        where x is the normalized magnetic field position [0, 1].

        :param noise_max_level: Maximum noise amplitude as a fraction of signal.
            Actual noise level is randomly sampled from [0, noise_max_level] for each spectrum

        :param baseline_quadratic: Maximum coefficient for quadratic baseline term (x²).
            Applied to normalized field coordinate

        :param baseline_linear: Maximum coefficient for linear baseline term (x).
            Applied to normalized field coordinate

        :param baseline_constant: Maximum coefficient for constant baseline offset.
            Applied to normalized field coordinate

        :param correct_baseline: If True, applies baseline correction by subtracting the mean
            of edge points from the spectrum
        :param baseline_points: Number of points at each edge of the spectrum used for
            baseline correction (only used if correct_baseline=True)
        """
        super().__init__()
        self.noise_max_level = noise_max_level
        self.baseline_quadratic = baseline_quadratic
        self.baseline_linear = baseline_linear
        self.baseline_constant = baseline_constant
        self.correct_baseline = correct_baseline
        self.baseline_points = baseline_points

    def forward(self, magnetic_field: torch.Tensor, spec: torch.Tensor):
        """
        Apply baseline distortion and noise to EPR spectra.

        The distortion process:
        1) Normalize magnetic field to [0, 1] range
        2) Generate quadratic baseline: a*x² + b*x + c
        3) Add Gaussian noise with random amplitude ∈ [0, noise_max_level]
        4) Optionally correct baseline by subtracting mean of edge points

        :param magnetic_field: Magnetic field positions in Tesla (T). Shape: [..., N]
        :param spec: EPR spectrum intensities. Shape: [..., N]
        :return: Distorted EPR spectrum with same shape as input. Shape: [..., N]

        Note: The noise level is sampled independently for each spectrum in the batch,
              but remains constant across all points within a single spectrum.
        """
        field_min = magnetic_field.min(dim=-1, keepdim=True)[0]
        field_max = magnetic_field.max(dim=-1, keepdim=True)[0]
        field_norm = (magnetic_field - field_min) / (field_max - field_min + 1e-8)

        baseline = (self.baseline_quadratic * field_norm ** 2 +
                    self.baseline_linear * field_norm +
                    self.baseline_constant)

        noise_max_level = torch.rand((spec.shape[:-1]), dtype=spec.dtype, device=spec.device) * self.noise_max_level
        noise = torch.randn_like(spec) * noise_max_level.unsqueeze(-1)
        distorted_spec = spec + baseline + noise
        if self.correct_baseline:
            baseline = (torch.mean(distorted_spec[..., :self.baseline_points], dim=-1) + torch.mean(
                distorted_spec[..., -self.baseline_points:], dim=-1)) / 2
            distorted_spec = distorted_spec - baseline.unsqueeze(-1)
        return distorted_spec


class SpectraModifier(nn.Module):
    def __init__(self, rng_generator=random.Random(None)):
        """
        Initialize the spectrum modifier pipeline.
        :param rng_generator: Random number generator
        """
        super().__init__()
        self.spec_field_prepare = SpecFieldPrepare(rng_generator=rng_generator)
        self.spec_field_distorter = SpectraDistortion()

    def forward(self,
                min_field_pos: torch.Tensor,
                max_field_pos: torch.Tensor,
                spec: torch.Tensor,
                ham_strain: torch.Tensor):
        """
        Prepare and distort EPR spectra for training.

        Process:
        1. Interpolate spectrum and generate magnetic field grid
        2. Generate Gaussian and Lorentzian linewidth parameters
        3. Apply baseline distortion and noise to simulate experimental artifacts

        :param min_field_pos: Minimum magnetic field position in Tesla (T).
            Shape: [...]
        :param max_field_pos: Maximum magnetic field position in Tesla (T).
            Shape: [...]
        :param spec: Initial magnetic resonance spectrum intensities.
            Shape: [..., num_initial_points]
        :param ham_strain: Hamiltonian strain tensor affecting linewidth constraints measured in Hz.
            Shape: [..., 3]
        :return: Dictionary containing:
            - field (torch.Tensor): Interpolated magnetic field grid in Tesla (T).
                Shape: [..., out_points]
            - spec (torch.Tensor): Clean interpolated spectrum.
                Shape: [..., out_points]
            - spec_distorted (torch.Tensor): Distorted spectrum with baseline and noise.
                Shape: [..., out_points]
            - gauss (torch.Tensor): Generated Gaussian linewidth in Tesla (T).
                Shape: [...]
            - lorentz (torch.Tensor): Generated Lorentzian linewidth in Tesla (T).
                Shape: [...]
        """
        field, spec, gauss, lorentz = self.spec_field_prepare(min_field_pos, max_field_pos, spec, ham_strain)
        spec_distorted = self.spec_field_distorter(field, spec)
        return {
            "field": field,
            "spec": spec,
            "spec_distorted": spec_distorted,
            "gauss": gauss,
            "lorentz": lorentz
        }