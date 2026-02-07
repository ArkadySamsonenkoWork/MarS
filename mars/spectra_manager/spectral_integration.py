from abc import ABC, abstractmethod
import math
import typing as tp

import torch
import torch.nn as nn


def gaussian_approx(x: torch.Tensor) -> torch.Tensor:
    """
    Compute a fast rational approximation to the Gaussian function exp(-x²).

    This implements a (3,3) Padé-type rational approximation optimized for
    speed and accuracy over the interval |x| ≤ 2.5:

    .. math::

        \exp(-x^2) \approx
        \frac{0.5835}
             {1 + 1.3125\,x^2 + 0.625\,x^4 + 0.1041666667\,x^6}

    The coefficients correspond to a 6th‑order rational approximant derived
    from standard numerical libraries (e.g., Cephes/Boost) and are chosen to
    minimize relative error (~1e-4) while avoiding expensive transcendental
    operations.

    Operates in-place for memory efficiency and is particularly suited for
    GPU-accelerated spectral simulations with Gaussian cutoffs around 2.5.

    :param x: Input tensor of any shape containing real values.
    :type x: torch.Tensor
    :return: Tensor of same shape containing the approximated Gaussian values.
    """
    out = torch.square(x, out=torch.empty_like(x))
    out.mul_(0.1041666667).add_(0.625)
    out.mul_(x).mul_(x).add_(1.3125)
    out.mul_(x).mul_(x).add_(1.0)
    out.reciprocal_().mul_(0.5835)
    return out


def gaussian_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Compute exact Gaussian function exp(-x²) using PyTorch operations.


    :param x: Input tensor of any shape containing values to evaluate Gaussian at
    :type x: torch.Tensor
    :return: Tensor of same shape containing exact Gaussian values exp(-x²)
    """
    x = x.clone()
    return torch.exp(x.square_().neg_())


class BaseIntegrand(nn.Module, ABC):
    """
    Abstract base class for spectral line shape integrands.

    Defines the interface for computing absorption and derivative line shapes
    with configurable Gaussian evaluation methods. Subclasses implement specific
    physical models for spectral contributions from individual transitions.

    Provides factory methods to select appropriate summation and Gaussian
    evaluation strategies based on harmonic order and performance requirements.
    """
    def _sum_method_fabric(self, harmonic: int = 0) -> tp.Callable[[tp.Any, tp.Any], torch.Tensor]:
        """
        Factory method returning appropriate summation function based on harmonic order.

        :param harmonic: Harmonic order (0 for absorption, 1 for first derivative)
        :type harmonic: int
        :return: Callable method for computing either absorption or derivative term
        :rtype: Callable[[Any, Any], torch.Tensor]
        :raises ValueError: If harmonic is not 0 or 1
        """
        if harmonic == 0:
            return self._absorption
        elif harmonic == 1:
            return self._derivative
        else:
            raise ValueError("Harmonic must be 0 or 1")

    def _gaussian_method_fabric(self, gaussian_method: str = "exp") -> tp.Callable[[torch.Tensor], torch.Tensor]:
        """
        Factory method returning Gaussian evaluation function based on approximation choice.

        :param gaussian_method: Method selection ('exp' for exact, 'approx' for rational approximation)
        :type gaussian_method: str
        :return: Callable function for Gaussian evaluation
        :rtype: Callable[[torch.Tensor], torch.Tensor]
        :raises ValueError: If gaussian_method is not 'exp' or 'approx'
        """
        if gaussian_method == "exp":
            return gaussian_torch
        elif gaussian_approx == "approx":
            return gaussian_approx
        else:
            raise ValueError("gaussian_method can be 'exp' or 'approximate'")

    @abstractmethod
    def _absorption(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def _derivative(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass


class ZeroOrderIntegrand(BaseIntegrand):
    """
    Zero-order integrand implementing Gaussian-broadened line shapes for powder spectra.

    Computes spectral contributions using the analytical form derived
    for powder-averaged spectra. Supports both absorption (0th harmonic) and first
    derivative (1st harmonic) detection modes with configurable Gaussian evaluation.

    The integrand evaluates terms of the form:

    - Absorption:   (1/√π) · c · exp[-(c·ΔB)²]
    - Derivative:   (2/√π) · c² · ΔB · exp[-(c·ΔB)²]

    where c = √2 / width is the inverse width scaling factor and ΔB is the field
    offset from resonance.
    """

    def __init__(self,
                 harmonic: int, gaussian_method: str, gaussian_cutoff: float,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize zero-order integrand with harmonic selection and Gaussian parameters.

        :param harmonic: Harmonic order (0 for absorption, 1 for derivative)
        :type harmonic: int
        :param gaussian_method: Gaussian evaluation method ('exp' or 'approx')
        :type gaussian_method: str
        :param gaussian_cutoff: Absolute value cutoff for Gaussian evaluation (values beyond cutoff yield zero)
        :type gaussian_cutoff: float
        :param device: Computation device for internal buffers
        :type device: torch.device
        """
        super().__init__()
        self.sum_method = self._sum_method_fabric(harmonic)
        self.gaussian_method = self._gaussian_method_fabric(gaussian_method)
        self.register_buffer("pi_sqrt", torch.tensor(math.sqrt(math.pi), device=device))
        self.register_buffer("two", torch.tensor(2.0, device=device))
        self.register_buffer("cutoff", torch.tensor(gaussian_cutoff, device=device))
        self.register_buffer("inv_pi_sqrt", torch.tensor(1.0 / math.sqrt(math.pi), device=device))

    def _absorption(self, arg: torch.Tensor, c_val: torch.Tensor) -> torch.Tensor:
        """
        Compute absorption term (zeroth harmonic) with Gaussian broadening.

        Implements: (1/√π) * c * exp(-(c·arg)²)

        :param arg: Argument tensor after field subtraction and scaling
        :type arg: torch.Tensor
        :param c_val: Inverse width scaling factor (sqrt(2)/width)
        :type c_val: torch.Tensor
        :return: Absorption contribution tensor with same shape as arg
        """
        arg_sq = self.gaussian_method(arg)

        arg_sq.mul_(c_val)
        arg_sq.mul_(self.inv_pi_sqrt)
        return arg_sq

    def _derivative(self, arg: torch.Tensor, c_val: torch.Tensor) -> torch.Tensor:
        """
        Compute derivative term (first harmonic) with Gaussian broadening.

        Implements: (2/√π) * c² * arg * exp(-(c·arg)²)

        :param arg: Argument tensor after field subtraction and scaling
        :type arg: torch.Tensor
        :param c_val: Inverse width scaling factor (1/width)
        :type c_val: torch.Tensor
        :return: Derivative contribution tensor with same shape as arg

        """
        exp_val = self.gaussian_method(arg)
        c_sq = c_val.square()
        result = arg.mul(self.two)
        result.mul_(exp_val)
        result.mul_(c_sq)
        result.mul_(self.inv_pi_sqrt)
        return result

    def forward(self, B_mean: torch.Tensor, c_extended: torch.Tensor, B_val: torch.Tensor) -> torch.Tensor:
        """
        Compute integrand value with masking for performance optimization.

        Applies Gaussian cutoff mask to skip computation where |arg| > cutoff.

        :param B_mean: Mean resonance field values [..., M]
        :type B_mean: torch.Tensor
        :param c_extended: Extended inverse width values [..., 1, M] or broadcastable
        :type c_extended: torch.Tensor
        :param B_val: Spectral field values [..., chunk, 1]
        :type B_val: torch.Tensor
        :return: Computed integrand values with shape matching broadcasted inputs
        :rtype: torch.Tensor
        """
        arg = B_mean.sub(B_val)
        arg.mul_(c_extended)
        mask = arg.abs() <= self.cutoff
        if not mask.any():
            return torch.zeros_like(arg)
        if mask.all():
            return self.sum_method(arg, c_extended.expand_as(arg))
        else:
            out = torch.zeros_like(arg)
            arg_masked = arg[mask]
            c_masked = c_extended.expand_as(arg)[mask]
            out[mask] = self.sum_method(arg_masked, c_masked)
            return out


class BaseSpectraIntegrator(nn.Module):
    """
    Abstract base class for spectrum integrators.

    Provides common infrastructure for computing  magnetic resonance
    spectra by integrating transition contributions over orientation space.

    Subclasses implement specific integration schemes (spherical, axial symmetry,
    mean-field) by overriding the forward method and providing appropriate geometry
    handling.
    """
    def __init__(self,
                 harmonic: int = 1,
                 gaussian_method: str = "exp",
                 gaussian_cutoff: float = 2.5,
                 natural_width: float = 1e-6, chunk_size=128, integration_level: int = 0,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
                 ):
        """
        :param harmonic: Harmonic order (0 for absorption, 1 for derivative)
        :type harmonic: int

        :param gaussian_method: Gaussian evaluation method ('exp' or 'approx')
        :type gaussian_method: str
        :param gaussian_cutoff: Absolute cutoff value for Gaussian evaluation optimization
        :type gaussian_cutoff: float
        :param natural_width: Minimum inherent linewidth added to all transitions
        :type natural_width: float
        :param chunk_size: Chunk size for processing spectral fields in the array of resonance lines in the spectra.
                           The total number of resonance lines depend on the number of orientations and
                           resonance transitions in the sample
        :type chunk_size: int
        :param integration_level: Integration refinement level (0 for basic, >0 for barycentric)
        :type integration_level: int
        :param device: Computation device
        :type device: torch.device
        :param dtype: Floating point precision for computations
        :type dtype: torch.dtype
        """
        super().__init__()
        self.harmonic = harmonic

        self.register_buffer("natural_width", torch.tensor(natural_width, device=device, dtype=dtype))
        self.chunk_size = chunk_size
        self.integrand = ZeroOrderIntegrand(harmonic, gaussian_method, gaussian_cutoff, device=device)

        self.register_buffer("pi_sqrt", torch.tensor(math.sqrt(math.pi), device=device, dtype=dtype))
        self.register_buffer("two_sqrt", torch.tensor(math.sqrt(2.0), device=device, dtype=dtype))
        self.register_buffer("three", torch.tensor(3.0, device=device, dtype=dtype))
        self.register_buffer("field_to_width", torch.tensor(1/9, device=device, dtype=dtype))
        self.register_buffer("additional_factor", torch.tensor(1.0, device=device, dtype=dtype))
        self.register_buffer("_width_conversion", torch.tensor(1 / math.sqrt(2 * math.log(2)), device=device))
        self.natural_width = self.natural_width * self._width_conversion

    @abstractmethod
    def forward(self, res_fields: torch.Tensor,
                  width: torch.Tensor, A_mean: torch.Tensor,
                  area: torch.Tensor, spectral_field: torch.Tensor):
        """
        :param res_fields: The resonance fields with the shape [..., M, 3].

        :param width: The width of transitions. The shape is [..., M]. This value is given as FWHM
        :param A_mean: The intensities of transitions. The shape is [..., M]
        :param area: The area of transitions. The shape is [M]. It is the same for all batch dimensions
        :param spectral_field: The magnetic fields where spectra should be created. The shape is [...., N]
        :return: result: Tensor of shape (..., N) with the value of the integral for each B
        """
        pass

    def _width_to_gaussian_scale(self, width: torch.Tensor) -> torch.Tensor:
        """
        Convert width to Gaussian scale parameter for integrand evaluation.

        Transforms width to the scale parameter used
        in Gaussian integrand calculations: c = √2 / width.

        :param width: torch.Tensor
            Spectral width valuest. Modified in-place.
        :return: Gaussian scale parameter (c_extended) for use in integrand functions.
            Same tensor as input (modified in-place via reciprocal and scaling).

        """
        c_extended = width.reciprocal_()
        c_extended.mul_(self.two_sqrt)
        return c_extended


class SphereSpectraIntegrator(BaseSpectraIntegrator):
    """
    Spectral integrator for general powder patterns using spherical triangle integration.

    Computes powder-averaged spectra by integrating over the full sphere using
    triangular surface elements. Supports barycentric subdivision (levels 1-3) for
    improved angular sampling accuracy. Each triangle is defined by three resonance
    field vertices [B₁, B₂, B₃] per transition.

    Implements geometric broadening correction where effective width incorporates
    field variation across triangle vertices:

        w_eff = √[w₀² + ((B₁−B₂)² + (B₂−B₃)² + (B₁−B₃)²)/9]
    """
    def __init__(self,
                 harmonic: int = 1,
                 gaussian_method: str = "exp",
                 gaussian_cutoff: float = 2.5,
                 natural_width: float = 1e-5, chunk_size=128,
                 integration_level: int = 0,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """
        Spectral integrator for general powder using triangle-based integration.

        :param harmonic: Harmonic order (0 for absorption, 1 for derivative)
        :type harmonic: int
        :param gaussian_method: Gaussian evaluation method
        :type gaussian_method: str
        :param gaussian_cutoff: Absolute cutoff value for Gaussian evaluation optimization
        :type gaussian_cutoff: float
        :param natural_width: Minimum inherent linewidth added to all transitions
        :type natural_width: float
        :param chunk_size: Chunk size for processing spectral fields in the array of resonance lines in the spectra.
                           The total number of resonance lines depend on the number of orientations and
                           resonance transitions in the sample
        :type chunk_size: int
        :param integration_level: Integration refinement level (0=basic centroid, 1-3=barycentric subdivision)
        :type integration_level: int
        :param device: Computation device
        :type device: torch.device
        :param dtype: Floating point precision for computations
        :type dtype: torch.dtype
        """
        super().__init__(
            harmonic, gaussian_method, gaussian_cutoff, natural_width, chunk_size,
            integration_level, device=device, dtype=dtype
        )
        if integration_level == 0:
            self._infty_ratio = self._infty_ratio_base
        else:
            self._infty_ratio = self._infty_ratio_barycentric

        _broad_level_factor = math.pow(5 / 9, integration_level)
        self.field_to_width = self.field_to_width.mul_(_broad_level_factor)

        self.register_buffer(
            "subtriangle_area_scale",
            torch.tensor(math.pow(1 / 3, integration_level), device=device, dtype=dtype)
        )

        self.multipliers, self.denominator = self._build_barycentric_numerators(integration_level)

    def _build_barycentric_numerators(self, level: int) -> tp.Tuple[torch.Tensor, int]:
        """
        Generate unnormalized barycentric weights for recursive triangle subdivision.

        Computes integer weight numerators and a common denominator for barycentric coordinates
        of sub-triangle vertices created by recursively splitting a parent triangle. Each split
        divides every existing triangle into 3 smaller sub-triangles by connecting its centroid
        to its vertices.

        Barycentric coordinates for any sub-triangle vertex are obtained by:
        coordinates = numerators / denominator

        :param level: int
        Subdivision depth (number of recursive splits). Must be 1, 2, or 3.
        - Level 1: 3 sub-triangles (first split)
        - Level 2: 9 sub-triangles (3²)
        - Level 3: 27 sub-triangles (3³)

        :return:
        numerators : torch.Tensor
            Shape (K, 3) tensor of integer weights for the three parent vertices,
            where K = 3^level is the number of sub-triangles at this level.
            Each row [W0, W1, W2] represents unnormalized barycentric weights that sum to `denominator`.

        denominator : torch.Tensor
            Scalar tensor equal to 9^level, used to normalize numerators into proper
            barycentric coordinates (values in [0, 1] that sum to 1).
        """
        if level not in (0, 1, 2, 3):
            raise ValueError("level must be 1, 2 or 3")

        device = self.natural_width.device

        if level == 0:
            numerators = torch.tensor([[0, 0, 0]], dtype=torch.int64, device=device)
            denominator = 1
            return numerators, denominator

        M = torch.tensor(
            [[4, 4, 1],
             [1, 4, 4],
             [4, 1, 4]],
            dtype=torch.int64,
            device=device
        )
        if level == 1:
            return M, 9

        M2 = torch.einsum("ij,kj->ikj", M, M).reshape(-1, 3)
        if level == 2:
            return M2, 9 * 9

        if level == 3:
            M3 = torch.einsum("ij,kj->ikj", M2, M).reshape(-1, 3)
            return M3, 9 * 9 * 9
        else:
            raise NotImplementedError("level must be 1,2,3")

    def _compute_effective_width(
            self, width: torch.Tensor,
            B1: torch.Tensor, B2: torch.Tensor, B3: torch.Tensor,
            spectral_width: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute extended  width incorporating geometric broadening from triangle vertices.

        Combines natural width, spectral resolution, and field-dependent broadening from
        triangle geometry into effective width, then returns it.
        Modifies the input tensor in-place

        :param width: Original transition widths (FWHM) with shape [..., M]
        :type width: torch.Tensor
        :param B1: First vertex resonance fields with shape [..., M]
        :type B1: torch.Tensor
        :param B2: Second vertex resonance fields with shape [..., M]
        :type B2: torch.Tensor
        :param B3: Third vertex resonance fields with shape [..., M]
        :type B3: torch.Tensor
        :param spectral_width: Part of spectral resolution (ΔB / alpha)
        :type spectral_width: torch.Tensor
        :return: Extended  width w_effective = w_effective with shape [..., M]
        :rtype: torch.Tensor
        """
        width.mul_(self._width_conversion)
        threshold = self.natural_width
        torch.where(width > threshold, width, width + threshold, out=width)

        d13 = B1.sub(B3)
        d13.div_(width)

        d23 = B2.sub(B3)
        d23.div_(width)

        d12 = B1.sub(B2)
        d12.div_(width)

        d13.square_()
        d23.square_()
        d12.square_()

        additional_width_square = d13
        additional_width_square.add_(d23)
        additional_width_square.add_(d12)
        additional_width_square.mul_(self.field_to_width)

        clamp_param = additional_width_square.clone()

        additional_width_square.add_(1.0)

        # I do no why, but it really works good. I set this coefficient as 2.0. It can be increased to 3.0
        additional_width_square.clamp_(min=3.0*clamp_param)

        width.square_()
        width.mul_(additional_width_square)

        threshold = spectral_width.unsqueeze_(-1).square_()
        torch.where(width > threshold, width, width + threshold, out=width)
        width.sqrt_()

        return width

    def _infty_ratio_base(self,
        B1: torch.Tensor,
        B2: torch.Tensor,
        B3: torch.Tensor,
        c_extended: torch.Tensor,
        B_val: torch.Tensor,) -> torch.Tensor:
        """
        Compute integrand at triangle centroid for basic integration level.

        :param B1: First vertex resonance fields [..., 1, M]
        :type B1: torch.Tensor
        :param B2: Second vertex resonance fields [..., 1, M]
        :type B2: torch.Tensor
        :param B3: Third vertex resonance fields [..., 1, M]
        :type B3: torch.Tensor
        :param c_extended: reciprocal line width [..., 1, M]
        :type c_extended: torch.Tensor
        :param B_val: Spectral field value [..., chunk, 1]
        :type B_val: torch.Tensor
        :return: Integrand values at centroid with shape [..., chunk, M]
        :rtype: torch.Tensor
        """
        B_cent_buf = (B1 + B2 + B3) / self.three

        return self.integrand(B_cent_buf, c_extended, B_val)

    def _infty_ratio_barycentric(
        self,
        B1: torch.Tensor,
        B2: torch.Tensor,
        B3: torch.Tensor,
        c_extended: torch.Tensor,
        B_val: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the 'infty ratio' integrated over all sub-triangle centroids defined
        by self.multipliers / self.denominator.

        Returns a tensor with shape [..., chunk, M] (same as ZeroOrderIntegrand.forward would give
        for a single centroid).

        Inputs:
         - B1,B2,B3: [..., 1, M]  (resonance vertices per transition)
         - width: [..., 1, M]  spectral width
         - B_val: [..., chunk, 1]
        """

        K = self.multipliers.shape[0]
        B_cent_buf = (B1 + B2 + B3) / self.three

        acc = None

        for i in range(K):
            n1 = float(self.multipliers[i, 0].item())
            n2 = float(self.multipliers[i, 1].item())
            n3 = float(self.multipliers[i, 2].item())

            B_cent_buf.mul_(0.0)
            B_cent_buf.add_(B1, alpha=n1)
            B_cent_buf.add_(B2, alpha=n2)
            B_cent_buf.add_(B3, alpha=n3)
            B_cent_buf.div_(self.denominator)
            ratio_i = self.integrand(B_cent_buf, c_extended, B_val)

            if acc is None:
                acc = ratio_i
            else:
                acc.add_(ratio_i)
        acc.mul_(1.0 / float(K))

        return acc

    def forward(self, res_fields: torch.Tensor,
                  width: torch.Tensor, A_mean: torch.Tensor,
                  area: torch.Tensor, spectral_field: torch.Tensor):
        r"""Convert width from FWHM to distance between infection points
        Computes the integral I(B) = 1/2 sqrt(2/pi) * (1/width) * A_mean *
        I_triangle(B) * area,

            at large B because of the instability of analytical solution we use easyspin-like solution with
            effective width
            w_additional = (((B1 - B2)**2 + (B2 - B3)**2 + (B1 - B3)**2) / 9).sqrt()
            w_effective = (w**2 + w_additional**2).sqrt()
        where
        :param res_fields: The resonance fields with the shape [..., M, 3]
        :param width: The width of transitions. The shape is [..., M]. This value is given as FWHM
        :param A_mean: The intensities of transitions. The shape is [..., M]

        if A is time dependant then res_fields, width, A_mean have dimensions:
        [..., 1, M, 3], [..., 1, M], [..., t, M] respectively

        :param area: The area of transitions. The shape is [M]. It is the same for all batch dimensions
        :param spectral_field: The magnetic fields where spectra should be created. The shape is [...., N]
        :return: result: Tensor of shape (..., N) with the value of the integral for each B
        """
        spectral_width = (spectral_field[..., 1] - spectral_field[..., 0])
        A_mean = A_mean * area
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)

        extended_width = self._compute_effective_width(width, B1, B2, B3, spectral_width).unsqueeze_(-2)
        c_extended = self._width_to_gaussian_scale(extended_width)

        A_mean = A_mean.unsqueeze_(-2)
        B1.unsqueeze_(-2)
        B2.unsqueeze_(-2)
        B3.unsqueeze_(-2)

        def lines_projection(B_val: torch.Tensor):
            """
            Compute total intensity contribution at specific spectral field value.

            :param B_val: Single spectral field value with shape [..., chunk, 1]
            :type B_val: torch.Tensor
            :return: Total intensity at B_val with shape [..., chunk]
            :rtype: torch.Tensor
            """
            ratio = self._infty_ratio(B1, B2, B3, c_extended, B_val)
            return (ratio * A_mean).sum(dim=-1)

        chunks = spectral_field.split(self.chunk_size, dim=-1)
        result = torch.cat([lines_projection(ch.unsqueeze(-1)) for ch in chunks], dim=-1)
        return result


class AxialSpectraIntegrator(BaseSpectraIntegrator):
    """
    Spectral integrator for axial symmetry powder patterns using bi-centric integration.

    Optimized for systems with axial symmetry (e.g., S=1 zero-field splitting with
    D ≠ 0, E=0) where powder integration reduces to a one-dimensional integral over
    the polar angle θ. Each transition is represented by two resonance field vertices
    corresponding to the extremal orientations (typically θ=0° and θ=90°).

    Implements geometric broadening correction specific to axial symmetry:

        w_eff = √[w₀² + 3·(B₁−B₂)²]

    Only basic centroid integration (level=0) is supported; barycentric subdivision
    is not applicable to the bi-centric geometry.
    """
    def __init__(self,
                 harmonic: int = 1,
                 gaussian_method: str = "exp",
                 gaussian_cutoff: float = 2.5,
                 natural_width: float = 1e-5, chunk_size=128,
                 integration_level: int = 0,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """
        Spectral integrator for axial symmetry powder patterns using bi-centric integration.

        :param harmonic: Harmonic order (0 for absorption, 1 for derivative)
        :type harmonic: int
        :param gaussian_method: Gaussian evaluation method ('exp' or 'approx')
        :type gaussian_method: str
        :param gaussian_cutoff: Absolute cutoff value for Gaussian evaluation optimization
        :type gaussian_cutoff: float
        :param natural_width: Minimum inherent linewidth added to all transitions
        :type natural_width: float
        :param chunk_size: Chunk size for processing spectral fields in the array of resonance lines in the spectra.
                           The total number of resonance lines depend on the number of orientations and
                           resonance transitions in the sample
        :type chunk_size: int
        :param integration_level: Integration refinement level (only 0 supported)
        :type integration_level: int
        :param device: Computation device
        :type device: torch.device
        :param dtype: Floating point precision for computations
        :type dtype: torch.dtype
        :raises NotImplementedError: If integration_level != 0
        """
        super().__init__(
            harmonic, gaussian_method,
            gaussian_cutoff, natural_width,
            chunk_size, integration_level, device=device, dtype=dtype
        )
        self.register_buffer("two", torch.tensor(2.0, device=device, dtype=dtype))
        self.register_buffer("field_to_width", torch.tensor(3.0, device=device, dtype=dtype))
        if integration_level != 0:
            raise NotImplementedError("Please set integration_level equal to zero for Axial integrator")

    def _compute_effective_width(
            self, width: torch.Tensor,
            B1: torch.Tensor, B2: torch.Tensor,
            spectral_width: torch.Tensor
    ):
        """
        Compute extended  width incorporating geometric broadening from triangle vertices.

        Combines natural width, spectral resolution, and field-dependent broadening from
        triangle geometry into effective width, then returns it.
        Modifies the input tensor in-place

        :param width: Original transition widths (FWHM) with shape [..., M]
        :type width: torch.Tensor
        :param B1: First vertex resonance fields with shape [..., M]
        :type B1: torch.Tensor
        :param B2: Second vertex resonance fields with shape [..., M]
        :type B2: torch.Tensor
        :param spectral_width: Part of spectral resolution (ΔB / alpha)
        :type spectral_width: torch.Tensor
        :return: Extended inverse width c_extended = √2 / w_effective with shape [..., M]
        :rtype: torch.Tensor
        """
        width.mul_(self._width_conversion)
        threshold = self.natural_width
        torch.where(width > threshold, width, width + threshold, out=width)

        d12 = B1.sub(B2)
        d12.div_(width)
        d12.square_()
        additional_width_square = d12
        additional_width_square.mul_(self.field_to_width)

        #torch.where(width > 2 * threshold, width, width + 2 * threshold, out=width)

        width.square_()
        clamp_param = additional_width_square.clone()
        additional_width_square.add_(1.0)

        # I do not why, but it really works
        additional_width_square.clamp_(min=2.0 * clamp_param)
        width.mul_(additional_width_square)

        threshold = spectral_width.unsqueeze_(-1).square_()
        torch.where(width > threshold, width, width + threshold, out=width)

        width.sqrt_()
        return width

    def _infty_ratio_base(self,
        B1: torch.Tensor,
        B2: torch.Tensor,
        c_extended: torch.Tensor,
        B_val: torch.Tensor) -> torch.Tensor:

        B_cent_buf = (B1 + B2) / self.two

        return self.integrand(B_cent_buf, c_extended, B_val)

    def _infty_ratio_biacentric(self,
        B1: torch.Tensor,
        B2: torch.Tensor,
        c_extended: torch.Tensor,
        B_val: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Please set integration_level equal to zero for Axial integrator")

    def forward(self, res_fields: torch.Tensor,
                  width: torch.Tensor, A_mean: torch.Tensor,
                  area: torch.Tensor, spectral_field: torch.Tensor):
        """
        Compute axial symmetry powder spectrum by integrating over bi-centric orientations.

        Converts FWHM to effective width incorporating geometric broadening from two vertices.

        :param res_fields: Resonance fields for two vertices with shape [..., M, 2]
        :type res_fields: torch.Tensor
        :param width: Transition widths given as FWHM with shape [..., M]
        :type width: torch.Tensor
        :param A_mean: Transition intensities with shape [..., M]
        :type A_mean: torch.Tensor
        :param area: Transition areas with shape [M] (same for all batch dimensions)
        :type area: torch.Tensor
        :param spectral_field: Magnetic field values where spectrum is evaluated with shape [..., N]
        :type spectral_field: torch.Tensor
        :return: Computed spectrum with shape [..., N]
        :rtype: torch.Tensor
        """
        A_mean = A_mean * area
        B1, B2 = torch.unbind(res_fields, dim=-1)
        spectral_width = (spectral_field[..., 1] - spectral_field[..., 0])

        width = self._compute_effective_width(width, B1, B2, spectral_width)
        c_extended = self._width_to_gaussian_scale(width)

        A_mean.unsqueeze_(-2)
        B1.unsqueeze_(-2)
        B2.unsqueeze(-2)
        c_extended.unsqueeze_(-2)

        def lines_projection(B_val: torch.Tensor):
            """
            :param B_val: the value of  spectral magnetic field.

            :return: The total intensity at this magnetic field
            """
            ratio = self._infty_ratio_base(B1, B2, c_extended, B_val)
            return (ratio * A_mean).sum(dim=1)

        chunks = spectral_field.split(self.chunk_size, dim=-1)
        result = torch.cat([lines_projection(ch.unsqueeze(-1)) for ch in chunks], dim=-1)
        return result


class MeanIntegrator(BaseSpectraIntegrator):
    """
   Mean-field spectral integrator for single-orientation or solution spectra.

   Computes spectra without angular dependence—suitable for:
     - Isotropic solution samples
     - Single-crystal simulations at fixed orientation
     - Mean-field approximations where powder averaging is not required

   Each transition contributes a single Gaussian line centered at its resonance
   field. No geometric broadening correction is applied since orientation spread
   is absent.
    """
    def forward(self, res_fields: torch.Tensor,
                  width: torch.Tensor, A_mean: torch.Tensor,
                  area: torch.Tensor, spectral_field: torch.Tensor):
        """
        Compute mean-field spectrum without angular dependence (single orientation).

        Simplest integrator that treats each transition as having a single resonance field.

        :param res_fields: Resonance fields with shape [..., M]
        :type res_fields: torch.Tensor
        :param width: Transition widths given as FWHM with shape [..., M]
        :type width: torch.Tensor
        :param A_mean: Transition intensities with shape [..., M]
        :type A_mean: torch.Tensor
        :param area: Transition areas with shape [M] (same for all batch dimensions)
        :type area: torch.Tensor
        :param spectral_field: Magnetic field values where spectrum is evaluated with shape [..., N]
        :type spectral_field: torch.Tensor
        :return: Computed spectrum with shape [..., N]
        :rtype: torch.Tensor
        """
        res_fields = res_fields.squeeze(-1)
        width.mul_(self._width_conversion)
        A_mean = A_mean * area

        width = self.natural_width + width
        c_extended = self._compute_effective_width(width)

        c_extended.unsqueeze_(-2)
        A_mean.unsqueeze_(-2)
        res_fields.unsqueeze_(-2)

        def lines_projection(B_val: torch.Tensor):
            """
            Compute total intensity contribution at specific spectral field value.

            :param B_val: Single spectral field value with shape [..., chunk, 1]
            :type B_val: torch.Tensor
            :return: Total intensity at B_val with shape [..., chunk]
            :rtype: torch.Tensor
            """
            ratio = self.integrand(res_fields, c_extended, B_val)
            return (ratio * A_mean).sum(dim=-1)

        chunks = spectral_field.split(self.chunk_size, dim=-1)
        result = torch.cat([lines_projection(ch.unsqueeze(-1)) for ch in chunks], dim=-1)
        return result
