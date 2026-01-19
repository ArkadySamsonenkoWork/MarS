import math
import typing as tp
from functools import wraps
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.fft as fft
import torch.nn as nn

from .. import constants
from .. import mesher
from .. import res_field_algorithm, secular_approximation, spin_system, res_freq_algorithm

from ..spectral_integration import BaseSpectraIntegrator,\
    SpectraIntegratorStationary, MeanIntegrator, AxialSpectraIntegratorStationary
from ..population import BaseTimeDepPopulator, StationaryPopulator, LevelBasedPopulator,\
    RWADensityPopulator, PropagatorDensityPopulator, BasePopulator
from ..population import contexts


def compute_matrix_element(vector_down: torch.Tensor, vector_up: torch.Tensor, G: torch.Tensor):
    """Compute transition matrix element <ψ_up| G |ψ_down>.

    :param vector_down: Lower-state eigenvector. Shape [..., N]
    :param vector_up: Upper-state eigenvector. Shape [..., N]
    :param G: Operator matrix (e.g., g-tensor component). Shape [..., N, N]
    :return: Complex-valued transition amplitude. Shape [...]
    """
    tmp = torch.matmul(G.unsqueeze(-3), vector_down.unsqueeze(-1))
    return (vector_up.conj() * tmp.squeeze(-1)).sum(dim=-1)


class PostSpectraProcessing(nn.Module):
    """Apply line-broadening (Gaussian, Lorentzian, or Voigt) to raw stick
    spectra.

    Supports batched and non-batched inputs. Automatically selects
    broadening method based on non-zero FWHM parameters. Convolution
    performed in Fourier domain.

    :param gauss: Gaussian FWHM (in same units as magnetic_field). Shape
        [] or [*batch_dims]
    :param lorentz: Lorentzian FWHM (in same units as magnetic_field).
        Shape [] or [*batch_dims]
    """
    def __init__(self, *args, **kwargs):
        """
        :param gauss: The gauss parameter.

        The shape is [batch_size] or []
        :param lorentz: The lorentz parameter. The shape is [batch_size] or []
        """
        super().__init__()
        pass

    def _skip_broader(self, gauss: torch.Tensor, lorentz: torch.Tensor,
                      magnetic_fields: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        return spec

    def _broading_fabric(self, gauss: torch.Tensor, lorentz: torch.Tensor) -> torch.Tensor:
        # Check if all values are zero (not just any)
        gauss_zero = (gauss == 0).all()
        lorentz_zero = (lorentz == 0).all()

        if gauss_zero and lorentz_zero:
            return self._skip_broader
        elif not gauss_zero and lorentz_zero:
            return self._gauss_broader
        elif gauss_zero and not lorentz_zero:
            return self._lorentz_broader
        else:
            return self._voigt_broader

    def forward(self, gauss: torch.Tensor, lorentz: torch.Tensor,
                magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param gauss: Tensor of shape [] or [*batch_dims].

        :param lorentz: Tensor of shape [] or [*batch_dims]
        :param magnetic_field: Tensor of shape [N] or [*batch_dims, N]
        :param spec: Spectrum tensor of shape [N] or [*batch_dims, N]
        :return: Broadened spectrum, same shape as spec
        """
        squeeze_output = False
        if gauss.dim() == 0:
            gauss = gauss.unsqueeze(0)
        if lorentz.dim() == 0:
            lorentz = lorentz.unsqueeze(0)
        if magnetic_field.dim() == 1:
            magnetic_field = magnetic_field.unsqueeze(0)
            squeeze_output = True
        if spec.dim() == 1:
            spec = spec.unsqueeze(0)

        _broading_method = self._broading_fabric(gauss, lorentz)
        result = _broading_method(gauss, lorentz, magnetic_field, spec)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _build_lorentz_kernel(self, magnetic_field: torch.Tensor, fwhm_lorentz: torch.Tensor) -> torch.Tensor:
        """
        :param magnetic_field: Shape [*batch_dims, N].

        :param fwhm_lorentz: Shape [*batch_dims]
        :return: Kernel of shape [*batch_dims, N]
        """
        dH = magnetic_field[..., 1] - magnetic_field[..., 0]
        N = magnetic_field.shape[-1]
        idx = torch.arange(N, device=magnetic_field.device) - N // 2

        # Reshape for broadcasting: idx -> [1, ..., 1, N]
        batch_dims = magnetic_field.dim() - 1
        idx_shape = [1] * batch_dims + [N]
        idx = idx.view(*idx_shape)

        # dH and fwhm_lorentz -> [*batch_dims, 1]
        dH = dH.unsqueeze(-1)
        gamma = (fwhm_lorentz.unsqueeze(-1) / 2)

        x = idx * dH
        L = (gamma / torch.pi) / (x ** 2 + gamma ** 2)
        return L

    def _build_gauss_kernel(self, magnetic_field: torch.Tensor, fwhm_gauss: torch.Tensor) -> torch.Tensor:
        """
        :param magnetic_field: Shape [*batch_dims, N].

        :param fwhm_gauss: Shape [*batch_dims]
        :return: Kernel of shape [*batch_dims, N]
        """
        dH = magnetic_field[..., 1] - magnetic_field[..., 0]
        N = magnetic_field.shape[-1]
        idx = torch.arange(N, device=magnetic_field.device) - N // 2

        # Reshape for broadcasting: idx -> [1, ..., 1, N]
        batch_dims = magnetic_field.dim() - 1
        idx_shape = [1] * batch_dims + [N]
        idx = idx.view(*idx_shape)

        # dH and fwhm_gauss -> [*batch_dims, 1]
        dH = dH.unsqueeze(-1)
        sigma = fwhm_gauss.unsqueeze(-1) / (2 * (2 * torch.log(torch.tensor(2.0, device=magnetic_field.device))) ** 0.5)

        x = idx * dH
        G = torch.exp(-0.5 * (x / sigma) ** 2) / (sigma * (2 * torch.pi) ** 0.5)
        return G

    def _build_voigt_kernel(self,
                            magnetic_field: torch.Tensor,
                            fwhm_gauss: torch.Tensor,
                            fwhm_lorentz: torch.Tensor) -> torch.Tensor:
        """
        :param magnetic_field: Shape [*batch_dims, N].

        :param fwhm_gauss: Shape [*batch_dims]
        :param fwhm_lorentz: Shape [*batch_dims]
        :return: Kernel of shape [*batch_dims, N]
        """
        N = magnetic_field.shape[-1]
        G = self._build_gauss_kernel(magnetic_field, fwhm_gauss)
        L = self._build_lorentz_kernel(magnetic_field, fwhm_lorentz)

        Gf = fft.rfft(torch.fft.ifftshift(G, dim=-1), dim=-1)
        Lf = fft.rfft(torch.fft.ifftshift(L, dim=-1), dim=-1)

        Vf = Gf * Lf
        V = torch.fft.fftshift(fft.irfft(Vf, n=N, dim=-1), dim=-1)
        V = V / V.sum(dim=-1, keepdim=True)
        return V

    def _apply_convolution(self, spec: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply convolution via FFT.

        :param spec: Shape [*batch_dims, N]
        :param kernel: Shape [*batch_dims, N]
        :return: Convolved spectrum of shape [*batch_dims, N]
        """
        S = fft.rfft(spec, dim=-1)
        K = fft.rfft(torch.fft.ifftshift(kernel, dim=-1), dim=-1)
        out = fft.irfft(S * K, n=spec.shape[-1], dim=-1)
        return out

    def _gauss_broader(self,
                       gauss: torch.Tensor, lorentz: torch.Tensor,
                       magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param gauss: Shape [*batch_dims].

        :param magnetic_field: Shape [*batch_dims, N]
        :param spec: Shape [*batch_dims, N]
        """
        kernel = self._build_gauss_kernel(magnetic_field, gauss)
        return self._apply_convolution(spec, kernel)

    def _lorentz_broader(self,
                         gauss: torch.Tensor, lorentz: torch.Tensor,
                         magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param lorentz: Shape [*batch_dims].

        :param magnetic_field: Shape [*batch_dims, N]
        :param spec: Shape [*batch_dims, N]
        """
        kernel = self._build_lorentz_kernel(magnetic_field, lorentz)
        return self._apply_convolution(spec, kernel)

    def _voigt_broader(self, gauss: torch.Tensor, lorentz: torch.Tensor,
                       magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param gauss: Shape [*batch_dims].

        :param lorentz: Shape [*batch_dims]
        :param magnetic_field: Shape [*batch_dims, N]
        :param spec: Shape [*batch_dims, N]
        """
        kernel = self._build_voigt_kernel(magnetic_field, gauss, lorentz)
        return self._apply_convolution(spec, kernel)


class BaseProcessing(nn.Module, ABC):
    """Base class for spectral integration and spectral post-processing over
    orientation meshes.

    This abstract class provides the framework for transforming resonance field data
    (fields, intensities, widths) into integrated spectra. It handles mesh-based orientation
    averaging for powder samples or single-crystal processing.

    The processing pipeline consists of:
    1. Transform resonance data to mesh format (interpolation, triangulation)
    2. Apply intensity masking based on threshold
    3. Integrate spectral contributions using the spectra integrator
    4. Apply post-processing (line broadening via convolution)
    """
    def __init__(self,
                 mesh: mesher.BaseMesh,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 chunk_size: int = 128,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param mesh: Mesh object defining orientation sampling grid.

        :param spectra_integrator: Integrator for computing spectra from resonance lines.
        Default is None and initialized with respect to class
        :param harmonic: Spectral harmonic (0 for absorption, 1 for first derivative). Default is 1
        :param post_spectra_processor: Processor for line broadening. Default is PostSpectraProcessing()
        :param chunk_size: Number of field points during integration. Default is 128
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """

        super().__init__()
        self.register_buffer("threshold", torch.tensor(1e-4, device=device))
        self.mesh = mesh
        self.post_spectra_processor = post_spectra_processor
        self.spectra_integrator = self._init_spectra_integrator(spectra_integrator, harmonic,
                                                                chunk_size=chunk_size, device=device, dtype=dtype)
        self.to(device)

    @abstractmethod
    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int,
                                 chunk_size: int, device: torch.device, dtype: torch.dtype):
        pass

    @abstractmethod
    def _compute_areas(self, expanded_size: torch.Tensor, device: torch.device):
        pass

    @abstractmethod
    def _transform_data_to_mesh_format(
            self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param res_fields: the tensor of resonance fields.

        The shape is [..., num_resonance fields]
        :param intensities: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :param width: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :return:
        res_fields tensor with the resonance field at each triangle vertices. The shape is [..., 3] or [...]
        width tensor with the resonance field at each triangle vertices. The shape is [...]
        intensities tensor with the resonance field at each triangle vertices. The shape is [...]
        areas tensor with the resonance field at each triangle vertices. The shape is [...]
        """
        pass

    def _final_mask(self, res_fields: torch.Tensor, width: torch.Tensor,
                    intensities: torch.Tensor, areas: torch.Tensor) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply intensity-based masking to discard negligible transitions.

        Transitions are retained only if their normalized intensity exceeds
        the internal threshold (default 1e-4). This reduces computational load
        during integration by excluding insignificant contributions.

        :param res_fields: Resonance fields at triangle vertices. Shape [..., M, 3] or [..., M]
        :param width: Linewidths associated with each transition. Shape [..., M]
        :param intensities: Transition intensities. Shape [..., M]
        :param areas: Integration weights (e.g., spherical triangle areas). Shape [..., M]
        :return: Filtered tensors (res_fields, width, intensities, areas), all with reduced last dimension
        """
        max_intensity = torch.amax(abs(intensities), dim=-1, keepdim=True)
        mask = ((intensities / max_intensity).abs() > self.threshold).any(dim=tuple(range(intensities.dim() - 1)))
        intensities = intensities[..., mask]
        width = width[..., mask]
        res_fields = res_fields[..., mask, :]
        areas = areas[..., mask]
        return res_fields, width, intensities, areas

    def _integration_precompute(self, res_fields, width, intensities, areas, fields):
        return res_fields, width, intensities, areas, fields

    def forward(self,
                res_fields: torch.Tensor,
                intensities: torch.Tensor,
                width: torch.Tensor,
                gauss: torch.Tensor,
                lorentz: torch.Tensor,
                fields: torch.Tensor):
        """
        Execute the full spectral processing pipeline:

        1. Map resonance data onto mesh geometry
        2. Apply intensity-based masking
        3. Precompute integration inputs
        4. Integrate spectrum using the configured integrator
        5. Apply line broadening via PostSpectraProcessing

        :param res_fields: Resonance magnetic fields. Shape [..., num_transitions]
        :param intensities: Transition intensities. Shape [..., num_transitions]
        :param width: Inhomogeneous linewidths (Gaussian FWHM). Shape [..., num_transitions]
        :param gauss: Gaussian broadening FWHM. Scalar or batched tensor.
        :param lorentz: Lorentzian broadening FWHM. Scalar or batched tensor.
        :param fields: Field axis for output spectrum. Shape [N] or [..., N]
        :return: Broadened spectrum matching shape of `fields`
        """

        res_fields, width, intensities, areas = (
            self._transform_data_to_mesh_format(
                res_fields, intensities, width
            )
        )
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)
        res_fields, width, intensities, areas, fields = self._integration_precompute(
            res_fields, width, intensities, areas, fields
        )
        spec = self.spectra_integrator(
            res_fields, width, intensities, areas, fields
        )
        return self.post_spectra_processor(gauss, lorentz, fields, spec)


class PowderStationaryProcessing(BaseProcessing):
    """Integrate stationary EPR spectra over spherical powder orientation mesh.

    This class provides the complete pipeline for transforming resonance field data
    (fields, intensities, widths) into integrated powder-averaged spectra for stationary
    (continuous-wave) EPR experiments.

    The processing pipeline consists of:
    1. Transform resonance data to mesh format (interpolation, triangulation)
    2. Apply intensity masking based on threshold
    3. Integrate spectral contributions using the spectra integrator
    4. Apply post-processing (line broadening via convolution)
    """
    def __init__(self,
                 mesh: mesher.BaseMeshPowder,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 chunk_size: int = 128,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32
                 ):
        """
        :param mesh: Powder mesh object (BaseMeshPowder) defining spherical grid.

        :param spectra_integrator: Custom integrator. Default is None (auto-initialized based on mesh parameters)
        :param harmonic: Spectral harmonic (0 for absorption, 1 for first derivative). Default is 1
        :param post_spectra_processor: Processor for line broadening. Default is PostSpectraProcessing()
        :param chunk_size: Number of field points during integration. Default is 128
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         chunk_size=chunk_size, device=device, dtype=dtype)

    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                 harmonic: int, chunk_size: int, device: torch.device, dtype: torch.dtype)\
            -> BaseSpectraIntegrator:
        """Initialize the appropriate spectra integrator based on mesh
        symmetry.

        Uses AxialSpectraIntegratorStationary for axial powder meshes;
        otherwise uses general SpectraIntegratorStationary.

        :param spectra_integrator: Optional pre-defined integrator
        :param harmonic: Spectral harmonic (0 = absorption, 1 = first
            derivative)
        :param chunk_size: Number of field points per integration chunk
        :param device: Computation device
        :param dtype: Floating-point precision
        :return: Instantiated integrator object
        """
        if spectra_integrator is None:
            if self.mesh.axial:
                return AxialSpectraIntegratorStationary(harmonic, chunk_size=chunk_size, device=device, dtype=dtype)
            return SpectraIntegratorStationary(harmonic, chunk_size=chunk_size, device=device, dtype=dtype)
        return spectra_integrator

    def _compute_areas(self, expanded_size: int, device: torch.device) -> torch.Tensor:
        """Compute spherical triangle areas from the powder mesh and expand.

        to match batch dimensions required for integration.

        :param expanded_size: Leading batch size before flattening
            (e.g., number of orientations)
        :param device: Target computation device
        :return: Flattened area tensor of shape [expanded_size *
            num_triangles]
        """
        grid, simplices = self.mesh.post_mesh
        areas = self.mesh.spherical_triangle_areas(grid, simplices)
        areas = areas.reshape(1, -1).expand(expanded_size, -1).flatten()
        return areas

    def _process_tensor(self, data_tensor: torch.Tensor) -> torch.Tensor:
        """Interpolate input resonance data (fields, intensities, widths) onto.

        the Delaunay triangulation defined by the powder mesh.

        :param data_tensor: Input tensor of shape [..., num_orientations, num_transitions]
        :return: Remapped tensor aligned with mesh simplices
        """
        _, simplices = self.mesh.post_mesh
        processed = self.mesh(data_tensor.transpose(-1, -2))
        return self.mesh.to_delaunay(processed, simplices)

    def _compute_batched_tensors(self, *args) -> torch.Tensor:
        """Stack multiple resonance-related tensors (e.g., fields, intensities,
        widths),.

        then remap them jointly onto the orientation mesh using `_process_tensor`.

        :param args: Tensors of identical shape [..., num_orientations, num_transitions]
        :return: Batched and mesh-aligned tensor of shape [..., 3, num_simplices, num_transitions]
        """
        batched_matrix = torch.stack(args, dim=-3)
        batched_matrix = self._process_tensor(batched_matrix)
        return batched_matrix

    def _transform_data_to_mesh_format(
            self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param res_fields: the tensor of resonance fields.

        The shape is [..., num_resonance fields]
        :param intensities: the tensor of resonance fields. The shape is [time, ..., num_resonance fields]
        :param width: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :return:
        res_fields tensor with the resonance field at each triangle vertices. The shape is [..., 3]
        width tensor with the resonance field at each triangle vertices. The shape is [...]
        intensities tensor with the resonance field at each triangle vertices. The shape is [...]
        areas tensor with the resonance field at each triangle vertices. The shape is [...]
        """
        batched_matrix = self._compute_batched_tensors(res_fields, intensities, width)
        expanded_size = batched_matrix.shape[-3]
        batched_matrix = batched_matrix.flatten(-3, -2)
        res_fields, intensities, width = torch.unbind(batched_matrix, dim=-3)

        width = width.mean(dim=-1)
        intensities = intensities.mean(dim=-1)
        areas = self._compute_areas(expanded_size, device=res_fields.device)
        return res_fields, width, intensities, areas


class CrystalStationaryProcessing(BaseProcessing):
    """Integrate stationary spectra for single-crystal or many-crystal oriented
    sample.

    This class provides the pipeline for transforming resonance field data into spectra
    for single-crystal samples or specific crystal orientations where no orientation
    averaging is required.

    The processing pipeline consists of:
    1. Transform resonance data to mesh format (interpolation, triangulation)
    2. Apply intensity masking based on threshold
    3. Integrate spectral contributions using mean contribution of each given orientation
    4. Apply post-processing (line broadening via convolution)
    """
    def __init__(self,
                 mesh: mesher.CrystalMesh,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 chunk_size: int = 128,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param mesh: Crystal mesh object defining single or discrete orientations.

        :param spectra_integrator: Custom integrator. Default is None (MeanIntegrator initialized)
        :param harmonic: Spectral harmonic (0 for absorption, 1 for first derivative). Default is 1
        :param post_spectra_processor: Processor for line broadening. Default is PostSpectraProcessing()
        :param chunk_size: Number of field points during integration. Default is 128
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         chunk_size=chunk_size, device=device, dtype=dtype)

    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int,
                                 chunk_size: int, device: torch.device, dtype: torch.dtype):
        if spectra_integrator is None:
            return MeanIntegrator(harmonic, chunk_size=chunk_size, device=device)
        else:
            return spectra_integrator

    def _compute_areas(self, expanded_size: torch.Size, device: torch.device):
        areas = torch.ones(expanded_size, dtype=torch.float32, device=device)
        return areas

    def _final_mask(self, res_fields: torch.Tensor, width: torch.Tensor,
                    intensities: torch.Tensor, areas: torch.Tensor):
        """Apply intensity thresholding for crystal (non-powder) samples.

        Unlike powder case, res_fields lacks triangle vertex dimension (no [..., 3]).
        Masking logic remains consistent: retain transitions above relative intensity threshold.

        :param res_fields: Resonance fields. Shape [..., M]
        :param width: Linewidths. Shape [..., M]
        :param intensities: Intensities. Shape [..., M]
        :param areas: Uniform weights (typically ones). Shape [..., M]
        :return: Filtered tensors with pruned last dimension
        """
        max_intensity = torch.amax(abs(intensities), dim=-1, keepdim=True)
        mask = ((intensities / max_intensity).abs() > self.threshold).any(dim=tuple(range(intensities.dim() - 1)))

        intensities = intensities[..., mask]
        width = width[..., mask]
        res_fields = res_fields[..., mask]
        areas = areas[..., mask]
        return res_fields, width, intensities, areas

    def _transform_data_to_mesh_format(
            self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param res_fields: the tensor of resonance fields.

        The shape is [..., num_resonance fields]
        :param intensities: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :param width: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :return:
        res_fields tensor with the resonance field at each triangle vertices. The shape is [...]
        width tensor with the resonance field at each triangle vertices. The shape is [...]
        intensities tensor with the resonance field at each triangle vertices. The shape is [...]
        areas tensor with the resonance field at each triangle vertices. The shape is [...]
        """
        res_fields = res_fields.flatten(-2, -1)
        intensities = intensities.flatten(-2, -1)
        width = width.flatten(-2, -1)

        expanded_size = res_fields.shape
        areas = self._compute_areas(expanded_size, res_fields.device)
        return res_fields, width, intensities, areas


class PowderTimeProcessing(PowderStationaryProcessing):
    """Integrate time-resolved EPR spectra over spherical powder orientation
    mesh.

    This class extends PowderStationaryProcessing to handle time-dependent intensities
    while keeping resonance fields and widths time-independent

    The processing pipeline consists of:
    1. Transform resonance data to mesh format (interpolation, triangulation)
    2. Apply intensity masking based on threshold
    3. Integrate spectral contributions.
    4. Apply post-processing (line broadening via convolution)
    """

    def _integration_precompute(self, res_fields: torch.Tensor, width: torch.Tensor,
                                intensities: torch.Tensor, areas: torch.Tensor, fields: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return res_fields.unsqueeze(-3),\
            width.unsqueeze(-2), intensities, areas.unsqueeze(-2), fields.unsqueeze(-2)

    def _transform_data_to_mesh_format(self, res_fields: torch.Tensor,
                                       intensities: torch.Tensor,
                                       width: torch.Tensor):
        """
        :param res_fields: the tensor of resonance fields.

        The shape is [..., num_resonance fields]
        :param intensities: the tensor of resonance fields. The shape is [time_dim, ..., num_resonance fields]
        :param width: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :return:
        res_fields tensor with the resonance field at each triangle vertices. The shape is [..., 3]
        width tensor with the resonance field at each triangle vertices. The shape is [...]
        intensities tensor with the resonance field at each triangle vertices. The shape is [time_dim, ...]
        areas tensor with the resonance field at each triangle vertices. The shape is [...]
        """
        batched_matrix = self._compute_batched_tensors(res_fields, width)
        expanded_size = batched_matrix.shape[-3]
        batched_matrix = batched_matrix.flatten(-3, -2)
        intensities = self._process_tensor(intensities)
        intensities = intensities.flatten(-3, -2)

        res_fields, width = torch.unbind(batched_matrix, dim=-3)
        width = width.mean(dim=-1)
        intensities = intensities.mean(dim=-1)
        areas = self._compute_areas(expanded_size, device=res_fields.device)
        return res_fields, width, intensities, areas


class CrystalTimeProcessing(CrystalStationaryProcessing):
    """Integrate time-resolved EPR spectra over single-crystal or many-crystal
    sample.

    This class extends PowderStationaryProcessing to handle time-dependent intensities
    while keeping resonance fields and widths time-independent

    The processing pipeline consists of:
    1. Transform resonance data to mesh format (interpolation, triangulation)
    2. Apply intensity masking based on threshold
    3. Integrate spectral contributions.
    4. Apply post-processing (line broadening via convolution)
    """
    def _integration_precompute(self, res_fields: torch.Tensor, width: torch.Tensor,
                                intensities: torch.Tensor, areas: torch.Tensor, fields: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return res_fields.unsqueeze(-3),\
            width.unsqueeze(-2), intensities, areas.unsqueeze(-2), fields.unsqueeze(-2)


class Broadener(nn.Module):
    """Compute inhomogeneous linewidths from spin Hamiltonian strain tensors.

    Evaluates field-dependent and field-independent contributions to
    transition width using perturbation theory on strained Hamiltonian
    components. Output is FWHM of Gaussian profile.

    Final width = sqrt(Hamiltonian_strain² + Σ(strain_contributions)²) ×
    (1/√(2 ln 2))
    """
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.register_buffer("_width_conversion", torch.tensor(1 / math.sqrt(2 * math.log(2)), device=device))
        self.to(device)

    def _compute_element_field_free(self, vector: torch.Tensor,
                          tensor_components_A: torch.Tensor, tensor_components_B: torch.Tensor,
                          transformation_matrix: torch.Tensor, correlation_matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            '...pij,jkl,ikl,...bk,...bl,ph->...hb',
            transformation_matrix, tensor_components_A, tensor_components_B, torch.conj(vector), vector,
            correlation_matrix
        ).real

    def _compute_element_field_dep(self, vector: torch.Tensor,
                          tensor_components: torch.Tensor,
                          transformation_matrix: torch.Tensor, correlation_matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            '...pi, ikl,...bk,...bl,ph->...hb',
            transformation_matrix, tensor_components, torch.conj(vector), vector, correlation_matrix
        ).real

    def _compute_field_strain_square(self, strained_data: tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                     vector_down: torch.Tensor, vector_up: torch.Tensor,
                                     B_trans: torch.Tensor) -> torch.Tensor:
        correlation_matrix, tensor_components, transformation_matrix = strained_data
        return (B_trans.unsqueeze(-2) * (
                self._compute_element_field_dep(vector_up, tensor_components, transformation_matrix,
                                                correlation_matrix) -
                self._compute_element_field_dep(vector_down, tensor_components, transformation_matrix,
                                                correlation_matrix)
        )).square().sum(dim=-2)

    def _compute_field_free_strain_square(self,
                                          strained_data: tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                                          vector_down: torch.Tensor, vector_up: torch.Tensor)\
            -> torch.Tensor:
        correlation_matrix, tensor_components_A, tensor_components_B, transformation_matrix = strained_data
        return (
                self._compute_element_field_free(
                    vector_up, tensor_components_A, tensor_components_B, transformation_matrix, correlation_matrix
                ) -
                self._compute_element_field_free(
                    vector_down, tensor_components_A, tensor_components_B, transformation_matrix, correlation_matrix
                )
        ).square().sum(dim=-2)

    def add_hamiltonian_straine(self, sample: spin_system.MultiOrientedSample, squared_width: torch.Tensor) ->\
            torch.Tensor:
        """Adds residual broadening due to unresolved interactions.

        :param sample: The MultiOrientedSample object
        :param squared_width: The square of gaussian broadening
        :return: Total gaussian broadening
        """
        hamiltonian_width = sample.build_ham_strain().unsqueeze(-1).square()
        return (squared_width + hamiltonian_width).sqrt()

    def forward(self, sample: spin_system.MultiOrientedSample,
                 vector_down: torch.Tensor, vector_up: torch.Tensor, B_trans: torch.Tensor) -> torch.Tensor:
        """Compute total Gaussian linewidth (FWHM) for each transition by
        combining:

        - Field-dependent strain contributions (from g-, D-tensor distributions)
        - Field-independent zero-field strain terms
        - Residual Hamiltonian strain (e.g., unresolved hyperfine)
        Result is scaled to FWHM using standard Gaussian conversion factor.

        :param sample: The MultiOrientedSample object
        :param vector_down: Lower-state eigenvector. Shape [..., N]
        :param vector_up: Upper-state eigenvector. Shape [..., N]
        :param B_trans: Magnetic fields of transitions
        :return: Return Total gaussian broadening due to the
            1) Unresolved interactions
            2) Hamiltonian parameters distribution
        """
        target_shape = vector_down.shape[:-1]
        result = torch.zeros(target_shape, dtype=B_trans.dtype, device=vector_down.device)

        for strained_data in sample.build_field_dep_strain():
            result += self._compute_field_strain_square(strained_data, vector_down, vector_up, B_trans)

        for strained_data in sample.build_zero_field_strain():
            result += self._compute_field_free_strain_square(strained_data, vector_down, vector_up)

        return self.add_hamiltonian_straine(sample, result) * self._width_conversion


class BaseIntensityCalculator(nn.Module):
    """Base class for computing EPR transition intensities.

    Handles calculation of transition intensities based on:
    - Transition matrix elements (magnetization)
    - Level populations (thermal, time-dependent, or custom)
    """
    def __init__(self,
                 spin_system_dim: int | list[int],
                 temperature: tp.Optional[float] = None,
                 populator: tp.Optional[tp.Union[BasePopulator, str]] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 disordered: bool = True,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param spin_system_dim: Dimension of spin system Hilbert space.

        :param temperature: Temperature in Kelvin of a sample.
        :param populator: BasePopulator object. Default is None
        (auto-initialized based on specific calculator).
        Also, can be set as string object for some cases (for example, for density computations)

        :param context: Relaxation/population context defining relaxation and initial population. Default is None
        :param disordered: If True, use powder averaging; if False, use crystal geometry. Default is True
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__()
        self.populator = self._init_populator(temperature, populator, context, disordered, device, dtype)
        self.spin_system_dim = spin_system_dim
        self.temperature = temperature
        self._compute_magnitization =\
            self._compute_magnitization_powder if disordered else self._compute_magnitization_crystal

        self.to(device)

    def _init_populator(self,  temperature: tp.Optional[float],
                        populator: tp.Optional[tp.Union[BasePopulator, str]],
                        context: tp.Optional[contexts.BaseContext], disordered: bool,
                        device: torch.device, dtype: torch.dtype) -> BasePopulator:
        """
        :param temperature: Sample temperature in Kelvin.

        :param populator: Optional custom population function
        :param context: Relaxation/population dynamics context
        :param disordered: True for powder averaging, False for single-crystal
        :param device: Computation device
        :param dtype: Floating-point type
        :return: BasePopulator object
        """
        return populator

    def _compute_magnitization_powder(self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                                      vector_down: torch.Tensor, vector_up: torch.Tensor):
        """Compute powder-averaged transition intensity.

        :param Gx, Gy, Gz: Cartesian components of Zeeman operator. Shape [..., N, N]
        :param vector_down: Lower-state eigenvector. Shape [..., N]
        :param vector_up: Upper-state eigenvector. Shape [..., N]
        :return: Intensity proportional to |<up|Gx|down>|² + |<up|Gy|down>|², in (J·s/μ_B)²
        """
        magnitization = compute_matrix_element(vector_down, vector_up, Gx).square().abs() +\
                        compute_matrix_element(vector_down, vector_up, Gy).square().abs()
        return magnitization * (constants.PLANCK / constants.BOHR) ** 2

    def _compute_magnitization_crystal(self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                                       vector_down: torch.Tensor, vector_up: torch.Tensor):
        """Compute crystal transition intensity.

        The orientation of the wave magnetic field is along the x-axis.
        :param Gx, Gy, Gz: Cartesian components of Zeeman operator. Shape [..., N, N]
        :param vector_down: Lower-state eigenvector. Shape [..., N]
        :param vector_up: Upper-state eigenvector. Shape [..., N]
        :return: Intensity proportional to |<up|Gx|down>|² + |<up|Gy|down>|², in (J·s/μ_B)²
        """
        magnitization = compute_matrix_element(vector_down, vector_up, Gx).square().abs()
        return magnitization * (constants.PLANCK / constants.BOHR) ** 2

    def compute_intensity(self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                vector_down: torch.Tensor, vector_up: torch.Tensor, lvl_down: torch.Tensor,
                lvl_up: torch.Tensor, resonance_energies: torch.Tensor, resonance_manifold,
                full_system_vectors: tp.Optional[torch.Tensor], *args, **kwargs):
        """Compute intensity of transitions.

        :param Gx, Gy, Gz: Zeeman operator components :param
        vector_down, vector_up: Transition eigenvectors :param lvl_down,
        lvl_up: Energy level indices
        :param resonance_energies: Hamiltonian eigenvalues
        :param resonance_manifold: Resonance condition values
            (fields/frequencies)
        :param full_system_vectors: Optional full eigenbasis
        :return: Transition intensities
        """
        raise NotImplementedError

    def forward(self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                vector_down: torch.Tensor, vector_up: torch.Tensor, lvl_down: torch.Tensor,
                lvl_up: torch.Tensor, resonance_energies: torch.Tensor, resonance_manifold,
                full_system_vectors: tp.Optional[torch.Tensor], *args, **kwargs):
        """:param Gx, Gy, Gz: Zeeman operator components.

        :param vector_down, vector_up: Transition eigenvectors :param
        lvl_down, lvl_up: Energy level indices
        :param resonance_energies: Hamiltonian eigenvalues
        :param resonance_manifold: Resonance condition values
            (fields/frequencies)
        :param full_system_vectors: Optional full eigenbasis
        :return: Transition intensities
        """
        return self.compute_intensity(Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies,
                                      resonance_manifold, full_system_vectors)


class StationaryIntensityCalculator(BaseIntensityCalculator):
    """Calculate transition intensities for stationary (CW) EPR experiments.

    Handles calculation of transition intensities based on:
    - Transition matrix elements (magnetization)
    - Level populations. Uses Boltzmann thermal populations at specified temperature
      or predefined population given in context.
    """
    def __init__(self, spin_system_dim: int, temperature: tp.Optional[float],
                 populator: tp.Optional[BasePopulator] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 disordered: bool = True,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """
        :param spin_system_dim: Dimension of spin system Hilbert space.

        :param temperature: Temperature in Kelvin of a sample.
        :param populator: BasePopulator object. Default is None
        (auto-initialized based as stationary populator)
        :param context: Relaxation/population context defining relaxation and initial population. Default is None
        :param disordered: If True, use powder averaging; if False, use crystal geometry. Default is True
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(spin_system_dim, temperature, populator, context, disordered, device=device, dtype=dtype)

    def _init_populator(self,
                        temperature: tp.Optional[float], populator: tp.Optional[BasePopulator],
                        context: tp.Optional[contexts.BaseContext],
                        disordered: bool, device: torch.device, dtype: torch.dtype) -> BasePopulator:
        """
        :param temperature: Sample temperature in Kelvin.

        :param populator: Optional population computation instance of BasePopulator
        :param context: Relaxation/population dynamics context
        :param disordered: True for powder averaging, False for single-crystal
        :param device: Computation device
        :param dtype: Floating-point type
        :return: BasePopulator object
        """
        if populator is None:
            return StationaryPopulator(context=context, init_temperature=temperature, device=device, dtype=dtype)
        else:
            return populator

    def compute_intensity(self,
                          Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                          vector_down: torch.Tensor, vector_up: torch.Tensor,
                          lvl_down: torch.Tensor, lvl_up: torch.Tensor, resonance_energies: torch.Tensor,
                          resonance_manifold: torch.Tensor,
                          full_system_vectors: tp.Optional[torch.Tensor], *args, **kwargs):
        """
        Compute CW-EPR transition intensities as the product of:

        :param Gx, Gy, Gz: Zeeman operator components. Shape [..., N, N]
        :param vector_down: Lower-state eigenvector. Shape [..., N]
        :param vector_up: Upper-state eigenvector. Shape [..., N]
        :param lvl_down, lvl_up: Energy level indices involved in transition
        :param resonance_energies: Eigenvalues of spin Hamiltonian. Shape [..., N]
        :param resonance_manifold: Resonance fields or frequencies. Shape [...]
        :param full_system_vectors: Full eigenbasis (optional, for advanced population models)
        :return: Intensity tensor matching transition dimension [...]
        """
        intensity = self.populator(resonance_energies, lvl_down, lvl_up, full_system_vectors, *args, **kwargs) * (
                self._compute_magnitization(Gx, Gy, Gz, vector_down, vector_up)
        )
        return intensity


class TimeIntensityCalculator(BaseIntensityCalculator):
    """Calculate time-dependent transition intensities for time-resolved EPR
    experiments based on relxation of.

    populations.

    Handles calculation of transition intensities based on:
    - Transition matrix elements (magnetization)
    - Level populations. Uses relaxation parameters and initial populations given in context
    """
    def __init__(self, spin_system_dim: int, temperature: tp.Optional[float],
                 populator: tp.Optional[tp.Union[BaseTimeDepPopulator, str]],
                 context: tp.Optional[contexts.BaseContext],
                 disordered: bool = True,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32,
                 ):
        """
        :param spin_system_dim: Dimension of spin system Hilbert space.

        :param temperature: Temperature in Kelvin of a sample.
        :param populator:
            Specifies the population calculator to use.
            If None (default), a LevelBasedPopulator or RWADensityPopulator  is automatically initialized
            depending on class.
            Alternatively, a string may be provided to select a density-based method:
            - rwa - uses the rotating-wave approximation
            - propagator - uses full time-propagator dynamics

        :param context: Relaxation/population context defining relaxation and initial population.
        :param disordered: If True, use powder averaging; if False, use crystal geometry. Default is True
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(
            spin_system_dim, temperature, populator, context, disordered, device=device, dtype=dtype
        )

    def _init_populator(self,
                        temperature: tp.Optional[float], populator: tp.Optional[tp.Union[BaseTimeDepPopulator, str]],
                        context: tp.Optional[contexts.BaseContext],
                        disordered: bool, device: torch.device, dtype: torch.dtype) -> BaseTimeDepPopulator:
        """
        :param temperature: Sample temperature in Kelvin.

        :param populator: Optional BaseTimeDepPopulator object
        :param context: Relaxation/population dynamics context
        :param disordered: True for powder averaging, False for single-crystal
        :param device: Computation device
        :param dtype: Floating-point type
        :return: BasePopulator object
        """
        if populator is None:
            return LevelBasedPopulator(context=context, init_temperature=temperature, device=device, dtype=dtype)
        else:
            return populator

    def compute_intensity(self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                          vector_down: torch.Tensor, vector_up: torch.Tensor,
                          lvl_down: torch.Tensor, lvl_up: torch.Tensor, resonance_energies: torch.Tensor,
                          resonance_manifold: torch.Tensor, full_system_vectors: tp.Optional[torch.Tensor],
                          *args, **kwargs):
        """Compute time-resolved EPR intensities based solely on transition
        matrix elements.

        Population dynamics are handled separately via `calculate_population`.
        This method returns the "geometric" part of intensity (|<ψ_up|G|ψ_down>|²).

        :param Gx, Gy, Gz: Zeeman operator components
        :param vector_down, vector_up: Eigenvectors of lower/upper states
        :param ...: Other parameters (unused here but kept for interface consistency)
        :return: Magnetization-squared term, shape [...]
        """
        intensity = (
                self._compute_magnitization(Gx, Gy, Gz, vector_down, vector_up)
        )
        return intensity

    def calculate_population(self, time: torch.Tensor,
                                    res_fields: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                                    resonance_energies: torch.Tensor,
                                    vector_down: torch.Tensor, vector_up: torch.Tensor,
                                    full_system_vectors: tp.Optional[torch.Tensor],
                                    *args, **kwargs
                             ):
        return self.populator(time, res_fields, lvl_down,
                              lvl_up, resonance_energies,
                              vector_down, vector_up,
                              full_system_vectors, *args, **kwargs)


class TimeDensityCalculator(TimeIntensityCalculator):
    """Calculate time-dependent transition intensities for time-resolved EPR
    experiments based on.

    matrix density relaxation formalism

    Default RWADensityPopulator populator is used
    """
    def _init_populator(self, temperature: torch.Tensor,
                        populator: tp.Optional[tp.Union[BaseTimeDepPopulator, str]],
                        context: tp.Optional[contexts.BaseContext], disordered: bool,
                        device: torch.device, dtype: torch.dtype):
        if populator is None:
            return RWADensityPopulator(
                context=context, init_temperature=temperature, disordered=disordered, device=device, dtype=dtype)
        elif isinstance(populator, str):
            if populator == "rwa":
                return RWADensityPopulator(
                    context=context, init_temperature=temperature, disordered=disordered, device=device, dtype=dtype)
            elif populator == "propagator":
                return PropagatorDensityPopulator(
                    context=context, init_temperature=temperature, disordered=disordered, device=device, dtype=dtype)
            else:
                raise ValueError("populator can be None, user-defined or sting 'rwa' or 'propagator'")
        else:
            setattr(populator, "disordered", disordered)
            return populator


@dataclass
class ParamSpec:
    """Let's consider the Hamiltonian with shape [..., N, N], where N is spin
    system size.

    Its resonance fields have dimension [...., K]. Let's call it 'scalar'
    Its eigen values have dimension [..., K, N], where K is number of resonance transitions. Let's call it 'vector'
    Its eigen vectors have dimension [..., K, N, N], where K is number of resonance transitions. Let's call it 'matrix'

    For some purposes it is necessary to get not only intensities, res-fields and width at resonance points
    but other parameters. To generalize the approach of making these parameters it is necessary to te
    """
    category: str
    dtype: torch.dtype

    def __post_init__(self):
        assert self.category in (
            "scalar", "vector", "matrix"), f"Category must be one of 'scalar', 'vector', 'matrix', got {self.category}"


class BaseSpectra(nn.Module, ABC):
    """Base class for EPR spectral simulation.

    Provides the complete pipeline for computing EPR spectra from spin Hamiltonian:
    1. Compute resonance fields/frequencies by diagonalizing Hamiltonian
    2. Calculate transition intensities from matrix elements and populations
    3. Compute linewidths from strain tensors
    4. Integrate over orientation mesh (for powder samples)
    5. Apply line broadening (Gaussian/Lorentzian/Voigt)

    Supports both stationary (CW) and time-resolved experiments, powder and
    single-crystal samples, field-swept and frequency-swept modes.
    """
    def __init__(self,
                 resonance_parameter: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_system.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[tp.Union[int, tuple]] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[BaseIntensityCalculator] = None,
                 populator: tp.Optional[tp.Union[BasePopulator, str]] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 integration_chunk_size: int = 128,
                 inference_mode: bool = True,
                 output_eigenvector: tp.Optional[bool] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 secular: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):
        """
        :param resonance_parameter: Resonance parameter of experiment: frequency or field.

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as default calculator specific to given spectra_creator

        :param populator:
            Class that is used to compute part intensity due to population of levels. Default is None
            If it is None then it is initialized as default populator specific to given (default) intensity_calculator

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance
        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it
        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For stationary creator is True, for time resolves is False

        :param integration_chunk_size:
            Chunk Size of integration process. Current implementation of powder integration is iterative.
            For whole set of resonance lines chunk size of spectral freq/field is computed.
            Increasing the size increases the integration speed, but also increases the required memory allocation.

        :param inference_mode: bool
            If inference_mode is True, then forward method will be performed under with torch.inference_mode():

        :param output_eigenvector: Optional[bool]
            If True, computes and returns the full system eigenvector. If False, returns None.
            For stationary computations, the default is False if context is None;
            for time-resolved simulations, the default is True.
            If set to None, the value is inferred automatically based on the population dynamics logic.

        :param context: Optional[context]
            The instance of BaseContext which describes the relaxation mechanism.
            It can have the initial population logic, transition between energy levels, dephasings, driven transition,
            out system transitions. For more complicated scenario the full relaxation superoperator can be used.

        :param secular: bool
            If parameter is True, then secular approximation will be used to compute resonance parameters.
            For time-resolved rotating wave approximation is necessary to use only True
            In general it is qute quicker than precise computation

        :param device: cpu / cuda. Base device for computations.

        :param dtype: float32 / float64
        Base dtype for all types of operations. If complex parameters is used,
        they will be converted in complex64, complex128
        """
        super().__init__()
        self.register_buffer("resonance_parameter", torch.tensor(resonance_parameter, device=device, dtype=dtype))
        self.register_buffer("threshold", torch.tensor(1e-2, device=device, dtype=dtype))
        self.register_buffer("tolerance", torch.tensor(1e-10, device=device, dtype=dtype))
        self.register_buffer("intensity_std", torch.tensor(1e-7, device=device, dtype=dtype))

        self.spin_system_dim, self.batch_dims, self.mesh =\
            self._init_sample_parameters(sample, spin_system_dim, batch_dims, mesh)
        self.mesh_size = self.mesh.initial_size
        self.broader = Broadener(device=device)

        self.output_eigenvector = self._init_output_eigenvector(output_eigenvector, context)
        self.res_algorithm = self._init_res_algorithm(
            output_eigenvector=self.output_eigenvector, secular=secular, device=device, dtype=dtype)
        if secular:
            self._hamiltonian_getter = lambda s: s.get_hamiltonian_terms_secular()
        else:
            self._hamiltonian_getter = lambda s: s.get_hamiltonian_terms()


        self.intensity_calculator = self._get_intensity_calculator(intensity_calculator,
                                                                   temperature, populator, context,
                                                                   device=device, dtype=dtype)
        self._param_specs = self._get_param_specs()
        self.spectra_processor = self._init_spectra_processor(spectra_integrator,
                                                              harmonic,
                                                              post_spectra_processor,
                                                              chunk_size=integration_chunk_size,
                                                              device=device, dtype=dtype)
        self.recompute_spin_parameters = recompute_spin_parameters
        self._init_cached_parameters()

        if inference_mode:
            self.forward = self._wrap_with_inference_mode(self.forward)

        self.to(device)
        self.to(dtype)

    def _init_cached_parameters(self):
        if not self.recompute_spin_parameters:
            self._cashed_flag = False
            self.vectors_u = None
            self.vectors_v = None
            self.valid_lvl_down = None
            self.valid_lvl_up = None
            self.res_fields = None
            self.resonance_energies = None
            self.full_eigen_vectors = None
            self._resfield_method = self._cashed_resfield

        else:
            self._resfield_method = self._recomputed_resfield

    def _wrap_with_inference_mode(self, forward_fn: tp.Callable[[tp.Any], tp.Any]):
        @wraps(forward_fn)
        def wrapper(*args, **kwargs):
            with torch.inference_mode():
                return forward_fn(*args, **kwargs)
        return wrapper

    def _init_res_algorithm(self, output_eigenvector: bool, secular: bool, device: torch.device, dtype: torch.dtype):
        return res_field_algorithm.ResField(
            spin_system_dim=self.spin_system_dim,
            mesh_size=self.mesh_size,
            batch_dims=self.batch_dims,
            output_full_eigenvector=output_eigenvector,
            device=device,
            dtype=dtype
        )

    @abstractmethod
    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing,
                                chunk_size: int,
                                device: torch.device,
                                dtype: torch.dtype) -> BaseProcessing:
        pass

    def _init_sample_parameters(self,
                                sample: tp.Optional[spin_system.MultiOrientedSample],
                                spin_system_dim: tp.Optional[int],
                                batch_dims: tp.Optional[tp.Union[int, tuple]],
                                mesh: tp.Optional[mesher.BaseMesh]):
        """
        Initilize the sample hyperparameters:

        -spin_system dimension
        -batch dimension
        -mesh size
        """
        if sample is None:
            if (spin_system_dim is not None) and (batch_dims is not None) and (mesh is not None):
                return spin_system_dim, batch_dims, mesh
            else:
                raise TypeError("You should pass sample or spin_system_dim, batch_dims, mesh arguments")
        else:
            spin_system_dim = sample.base_spin_system.spin_system_dim
            batch_dims = sample.config_shape[:-1]
            mesh = sample.mesh

        return spin_system_dim, batch_dims, mesh

    def _get_intensity_calculator(self,
                                  intensity_calculator: tp.Optional[BaseIntensityCalculator],
                                  temperature: float,
                                  populator: tp.Optional[tp.Union[BasePopulator, str]],
                                  context: tp.Optional[contexts.BaseContext],
                                  device: torch.device, dtype: torch.dtype):
        if intensity_calculator is None:
            return StationaryIntensityCalculator(
                self.spin_system_dim, temperature, populator, context, device=device, dtype=dtype
            )
        else:
            return intensity_calculator

    def _freq_to_field(self, vector_down: torch.Tensor, vector_up: torch.Tensor, Gz: torch.Tensor):
        """Compute frequency-to-field contribution."""
        factor_1 = compute_matrix_element(vector_up, vector_up, Gz)
        factor_2 = compute_matrix_element(vector_down, vector_down, Gz)

        diff = (factor_1 - factor_2).abs()
        safe_diff = torch.where(diff < self.tolerance, self.tolerance, diff)
        return safe_diff.reciprocal()

    def _init_output_eigenvector(
            self, output_eigenvector: tp.Optional[bool], context: tp.Optional[contexts.BaseContext]
    ) -> bool:
        if output_eigenvector is not None:
            return output_eigenvector
        else:
            return context is not None

    def _get_param_specs(self) -> list[ParamSpec]:
        """
        :return: list[ParamSpec].

        The number of parameters and
        their order must coincide with output of method _add_to_mask_additional
        """
        return []

    def _cashed_resfield(self, sample: spin_system.MultiOrientedSample,
                                B_low: torch.Tensor, B_high: torch.Tensor,
                                F: torch.Tensor, Gz: torch.Tensor):
        if not self._cashed_flag:
            (self.vectors_u, self.vectors_v), (self.valid_lvl_down, self.valid_lvl_up), self.res_fields,\
                self.resonance_energies, self.full_eigen_vectors =\
                self._recomputed_resfield(sample, B_low, B_high, F, Gz)

            self._cashed_flag = True

        return (self.vectors_u, self.vectors_v), (self.valid_lvl_down, self.valid_lvl_up), self.res_fields,\
            self.resonance_energies, self.full_eigen_vectors

    def _recomputed_resfield(self, sample: spin_system.MultiOrientedSample,
                                B_low: torch.Tensor, B_high: torch.Tensor,
                                F: torch.Tensor, Gz: torch.Tensor):
        (vectors_u, vectors_v), (valid_lvl_down, valid_lvl_up), res_fields, resonance_energies, full_eigen_vectors =\
                self.res_algorithm(sample, self.resonance_parameter, B_low, B_high, F, Gz)

        return (vectors_u, vectors_v), (valid_lvl_down, valid_lvl_up), res_fields,\
            resonance_energies, full_eigen_vectors

    def forward(self,
                 sample: spin_system.MultiOrientedSample,
                 fields: torch.Tensor, time: tp.Optional[torch.Tensor] = None, **kwargs):
        """
        :param sample: MultiOrientedSample object.

        :param fields: The magnetic fields in Tesla units
        :param time: It is used only for time resolved spectra
        :param kwargs:
        :return: spectra in 1D or 2D. Batched or un batched
        """
        B_low = fields[..., 0]
        B_high = fields[..., -1]
        B_low = B_low.unsqueeze(-1).repeat(*([1] * B_low.ndim), *self.mesh_size)
        B_high = B_high.unsqueeze(-1).repeat(*([1] * B_high.ndim), *self.mesh_size)

        F, Gx, Gy, Gz = self._hamiltonian_getter(sample)

        (vector_down, vector_up), (lvl_down, lvl_up), res_fields,\
            resonance_energies, full_system_vectors = self._resfield_method(sample, B_low, B_high, F, Gz)

        if (vector_down.shape[-2] == 0):
            return torch.zeros_like(fields)

        res_fields, intensities, width, full_system_vectors, *extras =\
            self.compute_parameters(sample, F, Gx, Gy, Gz,
                                    vector_down, vector_up,
                                    lvl_down, lvl_up,
                                    res_fields,
                                    resonance_energies,
                                    full_system_vectors)

        res_fields, intensities, width = self._postcompute_batch_data(
            sample, res_fields, intensities, width, F, Gx, Gy, Gz, full_system_vectors, time, *extras, **kwargs
        )

        gauss = sample.gauss
        lorentz = sample.lorentz

        return self._finalize(res_fields, intensities, width, gauss, lorentz, fields)

    def _postcompute_batch_data(self, sample: spin_system.BaseSample, res_fields: torch.Tensor,
                                intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                                Gz: torch.Tensor,
                                full_system_vectors: tp.Optional[torch.Tensor],
                                time: tp.Optional[torch.Tensor], *extras,  **kwargs):
        return res_fields, intensities, width

    def _finalize(self,
                  res_fields: torch.Tensor,
                  intensities: torch.Tensor,
                  width: torch.Tensor,
                  gauss: torch.Tensor,
                  lorentz: torch.Tensor,
                  fields: torch.Tensor):
        return self.spectra_processor(res_fields, intensities, width, gauss, lorentz, fields)

    def _mask_components(self, intensities_mask: torch.Tensor, *extras) -> list[tp.Any]:
        updated_extras = []
        for idx, param_spec in enumerate(self._param_specs):
            if param_spec.category == "scalar":
                updated_extras.append(extras[idx][..., intensities_mask])

            elif param_spec.category == "vector":
                updated_extras.append(extras[idx][..., intensities_mask, :])

            elif param_spec.category == "matrix":
                updated_extras.append(extras[idx][..., intensities_mask, :, :])
        return updated_extras

    def _add_to_mask_additional(self, vector_down: torch.Tensor, vector_up: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           resonance_energies: torch.Tensor):
        return ()

    def _mask_full_system_eigenvectors(
            self,
            mask: torch.Tensor,
            full_system_vectors: tp.Optional[torch.Tensor]
    ):
        if full_system_vectors is not None:
            return full_system_vectors[..., mask, :, :]
        else:
            return full_system_vectors

    def _compute_additional(self,
                           sample: spin_system.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor], *extras):
        return extras

    def compute_parameters(self, sample: spin_system.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           vector_down: torch.Tensor, vector_up: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           res_fields: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor]) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, tp.Optional[torch.Tensor], tuple[tp.Any]]:
        """
        :param sample: The sample which transitions must be found.

        :param F: Magnetic free part of spin Hamiltonian H = F + B * G
        :param Gx: x-part of Hamiltonian Zeeman Term
        :param Gy: y-part of Hamiltonian Zeeman Term
        :param Gz: z-part of Hamiltonian Zeeman Term

        :param vector_down:
            Eigenvectors of the lower energy states. The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param vector_up:
            Eigenvectors of the upper energy states.The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param resonance_energies:
            Energies of spin states. The shape is [..., N]

        :param res_fields: Resonance fields. The shape os [..., N]

        :param full_system_vectors: Eigen vector of each level of a spin system. The shape os [..., N, N]. If
        output_eigen_vectors == False, then it will be None

        :return: tuple of the next data
         - Resonance fields
         - Intensities of transitions
         - Width of transition lines
         - Full system eigen vectors or None
         - extras parameters computed in _compute_additional
        """
        intensities = self.intensity_calculator.compute_intensity(
            Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies, res_fields, full_system_vectors
        )
        lines_dimension = tuple(range(intensities.ndim - 1))
        intensities_mask = (intensities.abs() / intensities.abs().max() > self.threshold).any(dim=lines_dimension)
        intensities = intensities[..., intensities_mask]

        extras = self._add_to_mask_additional(vector_down,
            vector_up, lvl_down, lvl_up, resonance_energies)
        extras = self._mask_components(intensities_mask, *extras)
        full_system_vectors = self._mask_full_system_eigenvectors(intensities_mask, full_system_vectors)

        res_fields = res_fields[..., intensities_mask]
        vector_u = vector_down[..., intensities_mask, :]
        vector_v = vector_up[..., intensities_mask, :]

        freq_to_field = self._freq_to_field(vector_u, vector_v, Gz)
        intensities *= freq_to_field
        intensities = intensities / self.intensity_std
        width = self.broader(sample, vector_u, vector_v, res_fields) * freq_to_field

        extras = self._compute_additional(
            sample, F, Gx, Gy, Gz, full_system_vectors, *extras
        )
        return res_fields, intensities, width, full_system_vectors, *extras


class StationarySpectra(BaseSpectra):
    """Simulates standard EPR experiments where microwave frequency is fixed
    and.

    magnetic field is swept. Computes absorption or first-derivative spectra
    with proper orientation averaging for powder samples.

    Provides the complete pipeline for computing EPR spectra from spin Hamiltonian:
    1. Compute resonance fields/frequencies by diagonalizing Hamiltonian
    2. Calculate transition intensities from matrix elements and populations
    3. Compute linewidths from strain tensors
    4. Integrate over orientation mesh (for powder samples)
    5. Apply line broadening (Gaussian/Lorentzian/Voigt)

    Example usage:
        spectra = StationarySpectra(freq=9.8e9, sample=sample)
        fields = torch.linspace(0.2, 0.4, 500)
        spectrum = spectra(sample, fields)
    """
    def __init__(self,
                 freq: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_system.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[tp.Union[int, tuple]] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[BaseIntensityCalculator] = None,
                 populator: tp.Optional[StationaryPopulator] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 integration_chunk_size: int = 128,
                 inference_mode: bool = True,
                 output_eigenvector: tp.Optional[bool] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 secular: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):
        """
        :param freq: Resonance frequency of experiment at Hz.

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as StationaryIntensityCalculator

        :param populator:
            Class that is used to compute part intensity due to population of levels. Default is None
            If intensity_calculator is None or StationaryIntensityCalculator
            then it will be initialized as StationaryPopulator
            In this case the population is given as Boltzmann population

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum.

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance. Default is 1.

        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it

        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For stationary creator is True.

        :param integration_chunk_size:
            Chunk Size of integration process. Current implementation of powder integration is iterative.
            For whole set of resonance lines chunk size of spectral freq/field is computed.
            Increasing the size increases the integration speed, but also increases the required memory allocation.

        :param inference_mode: bool
            If inference_mode is True, then forward method will be performed under with torch.inference_mode():

        :param output_eigenvector: Optional[bool]
            If True, computes and returns the full system eigenvector. If False, returns None.
            For stationary computations, the default is False; for time-resolved simulations, the default is True.
            If set to None, the value is inferred automatically based on the population dynamics logic.

        :param context: Optional[context]
            The instance of BaseContext which describes the relaxation mechanism.
            It can have the initial population logic, transition between energy levels, dephasings, driven transition,
            out system transitions. For more complicated scenario the full relaxation superoperator can be used.

        :param secular: bool
            If parameter is True, then secular approximation will be used to compute resonance parameters.
            For time-resolved rotating wave approximation is necessary to use only True
            In general it is qute quicker than precise computation. Default is False

        :param device: cpu / cuda. Base device for computations.

        :param dtype: float32 / float64
        Base dtype for all types of operations. If complex parameters is used,
        they will be converted in complex64, complex128
        """
        super().__init__(freq, sample, spin_system_dim, batch_dims, mesh, intensity_calculator,
                         populator, spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters,
                         integration_chunk_size,
                         inference_mode, output_eigenvector, context, secular,
                         device=device, dtype=dtype)

    def _postcompute_batch_data(self, sample: spin_system.BaseSample,
                                res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                                full_system_vectors: tp.Optional[torch.Tensor],
                                time: tp.Optional[torch.Tensor],  *extras, **kwargs):
        return res_fields, intensities, width

    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing,
                                chunk_size: int,
                                device: torch.device,
                                dtype: torch.dtype) -> BaseProcessing:
        if self.mesh.disordered:
            return PowderStationaryProcessing(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                              chunk_size=chunk_size, device=device, dtype=dtype)
        else:
            return CrystalStationaryProcessing(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                               chunk_size=chunk_size, device=device, dtype=dtype)

    def __call__(self,
                sample: spin_system.MultiOrientedSample,
                fields: torch.Tensor, time: tp.Optional[torch.Tensor] = None, **kwargs):
        """
        :param sample: MultiOrientedSample object.

        :param fields: The magnetic fields in Tesla units
        :param time: It is used only for time resolved spectra
        :param kwargs:
        :return:
        """
        return super().__call__(sample, fields, time)


class TruncTimeSpectra(BaseSpectra):
    """Compute time-resolved EPR spectra for populations relaxation formalism.

    Uses truncated eigen vectors computation. For the general case use CoupledTimeSpectra

    Unlike CoupledTimeSpectra, only computes eigenvectors for resonant transitions
    (not full system), which improves computational efficiency.

    Provides the complete pipeline for computing EPR spectra from spin Hamiltonian:
    1. Compute resonance fields/frequencies by diagonalizing Hamiltonian
    2. Calculate transition intensities from matrix elements and populations
    3. Compute linewidths from strain tensors
    4. Integrate over orientation mesh (for powder samples)
    5. Apply line broadening (Gaussian/Lorentzian/Voigt)
    """
    def __init__(self,
                 freq: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_system.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[tp.Union[int, tuple]] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[tp.Callable] = None,
                 populator: tp.Optional[BaseTimeDepPopulator] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 0,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 integration_chunk_size: int = 128,
                 inference_mode: bool = True,
                 output_eigenvector: tp.Optional[bool] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 secular: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):

        """Note that by default these spin systems (energies, vectors, etc.)
        are calculated once and then cached.

        Default harmoinc is None

        :param freq: Resonance frequency of experiment

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as TimeIntensityCalculator

        :param populator:
            Object that is used to compute part intensity due to the difference of population between levels.
            By default, it is initialized as LevelBasedPopulator and uses solution of kinetic equation for populations

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum.

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance. Default is 0.

        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it

        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For time resolved spectra creator is False

        :param integration_chunk_size:
            Chunk Size of integration process. Current implementation of powder integration is iterative.
            For whole set of resonance lines chunk size of spectral freq/field is computed.
            Increasing the size increases the integration speed, but also increases the required memory allocation.

        :param inference_mode: bool
            If inference_mode is True, then forward method will be performed under with torch.inference_mode():

        :param output_eigenvector: Optional[bool]
            If True, computes and returns the full system eigenvector. If False, returns None.
            For stationary computations, the default is False; for time-resolved simulations, the default is True.
            If set to None, the value is inferred automatically based on the population dynamics logic.

        :param context: Optional[context]
            The instance of BaseContext which describes the relaxation mechanism.
            It can have the initial population logic, transition between energy levels, dephasings, driven transition,
            out system transitions. For more complicated scenario the full relaxation superoperator can be used.

        :param secular: bool
            If parameter is True, then secular approximation will be used to compute resonance parameters.
            For time-resolved rotating wave approximation is necessary to use only True
            In general it is qute quicker than precise computation. Default is False

        :param device: cpu / cuda. Base device for computations.

        :param dtype: float32 / float64
        Base dtype for all types of operations. If complex parameters is used,
        they will be converted in complex64, complex128
        """
        super().__init__(freq, sample, spin_system_dim, batch_dims, mesh, intensity_calculator, populator,
                         spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters,
                         integration_chunk_size,
                         inference_mode, output_eigenvector, context, secular,
                         device=device, dtype=dtype)

    def __call__(self, sample: spin_system.MultiOrientedSample, field: torch.Tensor, time: torch.Tensor, **kwargs) ->\
            torch.Tensor:
        """
        :param sample: MultiOrientedSample object.

        :param fields: The magnetic fields in Tesla units
        :param time: Time to compute time resolved spectra
        :param kwargs:
        :return: EPR spectra
        """
        return super().__call__(sample, field, time, **kwargs)

    def _get_intensity_calculator(self, intensity_calculator: tp.Optional[BaseIntensityCalculator],
                                  temperature: float,
                                  populator: tp.Optional[BaseTimeDepPopulator],
                                  context: tp.Optional[contexts.BaseContext],
                                  device: torch.device, dtype: torch.dtype):
        if intensity_calculator is None:
            return TimeIntensityCalculator(
                self.spin_system_dim, temperature, populator, context, device=device, dtype=dtype
            )
        else:
            return intensity_calculator

    def _get_param_specs(self) -> list[ParamSpec]:
        params = [
            ParamSpec("scalar", torch.long),
            ParamSpec("scalar", torch.long),
            ParamSpec("vector", torch.float32),
            ParamSpec("vector", torch.complex64),
            ParamSpec("vector", torch.complex64)
            ]
        return params

    def _add_to_mask_additional(self, vector_down: torch.Tensor, vector_up: torch.Tensor,
                        lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                        resonance_energies: torch.Tensor):

        return lvl_down, lvl_up, resonance_energies, vector_down, vector_up

    def _postcompute_batch_data(self, sample: spin_system.BaseSample,
                                res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                                Gz: torch.Tensor, full_system_vectors: tp.Optional[torch.Tensor],
                                time: torch.Tensor, *extras, **kwargs):
        lvl_down, lvl_up, resonance_energies, vector_down, vectors_up, *extras = extras

        population = self.intensity_calculator.calculate_population(
            time, res_fields, lvl_down, lvl_up,
            resonance_energies, vector_down, vectors_up, full_system_vectors, *extras
        )
        intensities = (intensities.unsqueeze(0) * population)
        return res_fields, intensities, width

    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing,
                                chunk_size: int,
                                device: torch.device,
                                dtype: torch.dtype) -> BaseProcessing:
        if self.mesh.disordered:
            return PowderTimeProcessing(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                        chunk_size=chunk_size, device=device, dtype=dtype)
        else:
            return CrystalTimeProcessing(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                        chunk_size=chunk_size, device=device, dtype=dtype)

    def _init_recompute_spin_flag(self) -> bool:
        """
        If flag is False: resfield data is cached.

        If flag is True: resfield recomputes every time
        :return:
        """
        return False

    def update_context(self, new_context: contexts.BaseContext) -> None:
        """Update context.

        :param new_context: New context object with updated parameters
        :return:
        """
        self.intensity_calculator.populator.context = new_context


class CoupledTimeSpectra(TruncTimeSpectra):
    """Compute time-resolved EPR spectra for populations relaxation formalism.

    Provides the complete pipeline for computing EPR spectra from spin Hamiltonian:
    1. Compute resonance fields/frequencies by diagonalizing Hamiltonian
    2. Calculate transition intensities from matrix elements and populations
    3. Compute linewidths from strain tensors
    4. Integrate over orientation mesh (for powder samples)
    5. Apply line broadening (Gaussian/Lorentzian/Voigt)
    """
    def _init_output_eigenvector(self, output_eigenvector: tp.Optional[bool],
                                 context: tp.Optional[contexts.BaseContext]) -> bool:
        if output_eigenvector is not None:
            return output_eigenvector
        else:
            return True


class DensityTimeSpectra(CoupledTimeSpectra):
    """Compute time-resolved EPR spectra for density matrix relaxation
    formalism.

    Default the rotating wave approximation is used

    Provides the complete pipeline for computing EPR spectra from spin Hamiltonian:
    1. Compute resonance fields/frequencies by diagonalizing Hamiltonian
    2. Calculate transition intensities from matrix elements and populations
    3. Compute linewidths from strain tensors
    4. Integrate over orientation mesh (for powder samples)
    5. Apply line broadening (Gaussian/Lorentzian/Voigt)
    """

    def __init__(self,
                 freq: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_system.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[tp.Union[int, tuple]] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[tp.Callable] = None,
                 populator: tp.Optional[tp.Union[BaseTimeDepPopulator, str]] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 0,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 integration_chunk_size: int = 128,
                 inference_mode: bool = True,
                 output_eigenvector: tp.Optional[bool] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 secular: bool = True,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):

        """Note that by default these spin systems (energies, vectors, etc.)
        are calculated once and then cached.

        Default harmoinc is None

        :param freq: Resonance frequency of experiment

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as TimeIntensityCalculator

        :param populator:
            Object used to compute the solution of the Liouville–von Neumann equation.
            If None`(the default), the solver uses the rotating-wave approximation (RWA)
            with RWADensityPopulator to evolve the density matrix.
            To use full propagator-based dynamics instead, pass the string 'propagator',
            which selects PropagatorDensityPopulator.

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum.

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance. Default is 0.

        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it

        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For time resolved spectra creator is False

        :param integration_chunk_size:
            Chunk Size of integration process. Current implementation of powder integration is iterative.
            For whole set of resonance lines chunk size of spectral freq/field is computed.
            Increasing the size increases the integration speed, but also increases the required memory allocation.

        :param inference_mode: bool
            If inference_mode is True, then forward method will be performed under with torch.inference_mode():

        :param output_eigenvector: Optional[bool]
            If True, computes and returns the full system eigenvector. If False, returns None.
            For stationary computations, the default is False; for time-resolved simulations, the default is True.
            If set to None, the value is inferred automatically based on the population dynamics logic.

        :param context: Optional[context]
            The instance of BaseContext which describes the relaxation mechanism.
            It can have the initial population logic, transition between energy levels, dephasings, driven transition,
            out system transitions. For more complicated scenario the full relaxation superoperator can be used.

        :param secular: bool
            If parameter is True, then secular approximation will be used to compute resonance parameters.
            For time-resolved rotating wave approximation (RWA) is necessary to use only True
            In general it is qute quicker than precise computation.
            For this class default is True bacause RWA is default method.

        :param device: cpu / cuda. Base device for computations.

        :param dtype: float32 / float64
        Base dtype for all types of operations. If complex parameters is used,
        they will be converted in complex64, complex128
        """
        super().__init__(freq, sample, spin_system_dim, batch_dims, mesh, intensity_calculator, populator,
                         spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters,
                         integration_chunk_size,
                         inference_mode, output_eigenvector, context, secular,
                         device=device, dtype=dtype)

    def _postcompute_batch_data(self, sample: spin_system.BaseSample,
                                res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                                Gz: torch.Tensor, full_system_vectors: tp.Optional[torch.Tensor],
                                time: torch.Tensor, *extras, **kwargs):
        lvl_down, lvl_up, resonance_energies, vector_down, vectors_up, *extras = extras
        Sz = sample.base_spin_system.get_electron_z_operator()
        population = self.intensity_calculator.calculate_population(
            time, res_fields, lvl_down, lvl_up,
            resonance_energies, vector_down, vectors_up,
            full_system_vectors,
            F, Gx, Gy, Gz, Sz,
            self.resonance_parameter, *extras
        )
        intensities = population
        return res_fields, intensities, width

    def _get_intensity_calculator(self, intensity_calculator: tp.Optional[BaseIntensityCalculator],
                                  temperature: float,
                                  populator: tp.Optional[tp.Union[BaseTimeDepPopulator, str]],
                                  context: tp.Optional[contexts.BaseContext],
                                  device: torch.device, dtype: torch.dtype):
        if intensity_calculator is None:
            return TimeDensityCalculator(
                self.spin_system_dim, temperature, populator, context, device=device, dtype=dtype
            )
        else:
            return intensity_calculator


class StationaryFreqSpectra(StationarySpectra):
    """Compute stationary EPR spectra at frequency domain.

    Default the rotating wave approximation is used

    Provides the complete pipeline for computing EPR spectra from spin Hamiltonian:
    1. Compute resonance fields/frequencies by diagonalizing Hamiltonian
    2. Calculate transition intensities from matrix elements and populations
    3. Compute linewidths from strain tensors
    4. Integrate over orientation mesh (for powder samples)
    5. Apply line broadening (Gaussian/Lorentzian/Voigt)
    """

    def __init__(self,
                 field: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_system.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[tp.Union[int, tuple]] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[BaseIntensityCalculator] = None,
                 populator: tp.Optional[StationaryPopulator] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 integration_chunk_size: int = 128,
                 inference_mode: bool = True,
                 output_eigenvector: tp.Optional[bool] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32
                 ):
        """
        :param field: Resonance field of experiment.

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as StationaryIntensityCalculator

        :param populator:
            Class that is used to compute part intensity due to population of levels. Default is None
            If intensity_calculator is None or StationaryIntensityCalculator
            then it will be initialized as StationaryPopulator
            In this case the population is given as Boltzmann population

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum.

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance. Default is 1.

        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it

        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For stationary creator is True.

        :param integration_chunk_size:
            Chunk Size of integration process. Current implementation of powder integration is iterative.
            For whole set of resonance lines chunk size of spectral freq/field is computed.
            Increasing the size increases the integration speed, but also increases the required memory allocation.

        :param inference_mode: bool
            If inference_mode is True, then forward method will be performed under with torch.inference_mode():

        :param output_eigenvector: Optional[bool]
            If True, computes and returns the full system eigenvector. If False, returns None.
            For stationary computations, the default is False; for time-resolved simulations, the default is True.
            If set to None, the value is inferred automatically based on the population dynamics logic.

        :param context: Optional[context]
            The instance of BaseContext which describes the relaxation mechanism.
            It can have the initial population logic, transition between energy levels, dephasings, driven transition,
            out system transitions. For more complicated scenario the full relaxation superoperator can be used.
        """
        super().__init__(field, sample, spin_system_dim, batch_dims, mesh, intensity_calculator,
                         populator, spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters,
                         integration_chunk_size,
                         inference_mode, output_eigenvector, context,
                         device=device, dtype=dtype)

    def _init_res_algorithm(self, output_eigenvector: bool, secular: bool, device: torch.device, dtype: torch.dtype):
        return res_freq_algorithm.ResFreq(
            spin_system_dim=self.spin_system_dim,
            mesh_size=self.mesh_size,
            batch_dims=self.batch_dims,
            output_full_eigenvector=output_eigenvector,
            device=device,
            dtype=dtype
        )

    def __call__(self,
                sample: spin_system.MultiOrientedSample,
                freq: torch.Tensor, time: tp.Optional[torch.Tensor] = None, **kwargs):
        """
        :param sample: MultiOrientedSample object.

        :param freq: The frequency in Hz units
        :param time: It is used only for time resolved spectra
        :param kwargs:
        :return:
        """
        return super().__call__(sample, freq, time)

    def compute_parameters(self, sample: spin_system.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           vector_down: torch.Tensor, vector_up: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           res_freq: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor]) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, tp.Optional[torch.Tensor], tuple[tp.Any]]:
        """
        :param sample: The sample which transitions must be found.

        :param F: Magnetic free part of spin Hamiltonian H = F + B * G
        :param Gx: x-part of Hamiltonian Zeeman Term
        :param Gy: y-part of Hamiltonian Zeeman Term
        :param Gz: z-part of Hamiltonian Zeeman Term

        :param vector_down:
            Eigenvectors of the lower energy states. The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param vector_up:
            Eigenvectors of the upper energy states.The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param resonance_energies:
            Energies of spin states. The shape is [..., N]

        :param res_freq: Resonance frequencies. The shape os [..., N]

        :param full_system_vectors: Eigen vector of each level of a spin system. The shape os [..., N, N]

        :return: tuple of the next data
         - Resonance fields
         - Intensities of transitions
         - Width of transition lines
         - Eigen vectors of all system levels or None
         - extras parameters computed in _compute_additional
        """

        intensities = self.intensity_calculator.compute_intensity(
            Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies, res_freq, full_system_vectors
        )
        lines_dimension = tuple(range(intensities.ndim - 1))
        intensities_mask = (intensities / intensities.abs().max() > self.threshold).any(dim=lines_dimension)
        intensities = intensities[..., intensities_mask]

        extras = self._add_to_mask_additional(vector_down,
            vector_up, lvl_down, lvl_up, resonance_energies)

        extras = self._mask_components(intensities_mask, *extras)
        full_system_vectors = self._mask_full_system_eigenvectors(intensities_mask, full_system_vectors)

        res_fields = res_freq[..., intensities_mask]
        vector_u = vector_down[..., intensities_mask, :]
        vector_v = vector_up[..., intensities_mask, :]

        intensities = intensities / self.intensity_std
        width = self.broader(sample, vector_u, vector_v, res_fields)

        extras = self._compute_additional(
            sample, F, Gx, Gy, Gz, full_system_vectors, *extras
        )

        return res_fields, intensities, width, full_system_vectors, *extras
