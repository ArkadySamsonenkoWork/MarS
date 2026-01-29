import typing as tp

import torch
import torch.nn as nn

from . import spin_model

### The energy computation is not effective due to the usage of common interface for population computation
### It should be rebuild without expand operation
class Locator(nn.Module):
    """Initialize a resonance locator that identifies EPR-allowed transitions
    within a frequency window.

    at a fixed magnetic field.
    This class evaluates the Hamiltonian H = F + B * Gz at a given field `B`,
    computes its eigenvalues and eigenvectors, and then determines which pairwise energy differences
    (i.e., transition frequencies) fall within the interval [freq_low, freq_high].
    Only upper-triangular level pairs (i < j) are considered to avoid double-counting degenerate transitions.
    """
    def __init__(self, output_full_eigenvector: bool, spin_dim: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.output_full_eigenvector = output_full_eigenvector
        self._triu_indices = torch.triu_indices(spin_dim, spin_dim, offset=1, device=device)
        self.device = device

    def _get_resonance_indexes(self, eigen_values: torch.Tensor,
                               freq_low: torch.Tensor, freq_high: torch.Tensor):
        transitions = eigen_values[..., None, :] - eigen_values[..., :, None]

        i_indices, j_indices = self._triu_indices
        transition_freq = transitions[..., i_indices, j_indices]

        valid_mask = (transition_freq >= freq_low[..., None]) & (transition_freq <= freq_high[..., None])
        return valid_mask, i_indices, j_indices, transition_freq

    def forward(self,
                F: torch.Tensor, Gz: torch.Tensor, freq_low: torch.Tensor,
                freq_high: torch.Tensor, resonance_field: torch.Tensor
                ):
        """Compute all spin transitions whose frequencies lie within [freq_low,
        freq_high] at a fixed magnetic field `resonance_field`.

        The Hamiltonian is constructed as H = F + resonance_field * Gz,
        diagonalized to obtain eigenvalues {E_k} and eigenvectors {|ψ_k⟩}.
        Transition frequencies ν_{ij} = E_j - E_i (with i < j) are compared against the frequency window.
        Valid transitions are collected, and corresponding eigenvectors are extracted.
        Output tensors are masked to include only transitions satisfying the frequency condition.

        :param F:
            'torch.Tensor'
                Field-independent part of the Hamiltonian. Shape: [..., N, N].
        :param Gz:
            'torch.Tensor'
                Zeeman interaction operator (z-component). Shape: [..., N, N].
        :param freq_low:
            'torch.Tensor'
                Lower bound of the target frequency interval. Shape: [...].
        :param freq_high:
            'torch.Tensor'
                Upper bound of the target frequency interval. Shape: [...].
        :param resonance_field:
            'torch.Tensor'
                Fixed magnetic field value at which to evaluate the spectrum.

        :return:
            'tuple[torch.Tensor, torch.Tensor]'
                Eigenvectors of lower (`vectors_u`) and upper (`vectors_v`) states for each valid transition.
                Shape: [..., N_trans, K], where N_trans is the number of transitions in the frequency window.
            'tuple[torch.Tensor, torch.Tensor]'
                Level indices (in ascending energy order) involved in each transition: (lvl_down, lvl_up).
                Shape: [N_trans,] — global across batch due to masking logic.
            'torch.Tensor'
                Transition frequencies ν_{ij} = E_j - E_i that satisfy freq_low ≤ ν ≤ freq_high.
                Shape: [..., N_trans].
            'torch.Tensor'
                Full energy spectra (all K levels) repeated per valid transition.
                Shape: [..., N_trans, K].
            'torch.Tensor | None'
                Full eigenvector matrices (all K states) for each valid transition, if `output_full_eigenvector=True`.
                Shape: [..., N_trans, K, K]. Otherwise, None.
        """
        H = F + Gz * resonance_field
        eigen_values, eigen_vectors = torch.linalg.eigh(H)
        valid_mask, i_indices, j_indices, transition_freq = self._get_resonance_indexes(
            eigen_values, freq_low, freq_high
        )
        mask_trans = valid_mask.any(dim=-2)

        lvl_down = i_indices[mask_trans]
        lvl_up = j_indices[mask_trans]
        transition_freq = transition_freq[..., mask_trans]


        vectors_u = eigen_vectors[..., lvl_down].transpose(-2, -1) * valid_mask[..., mask_trans].unsqueeze(-1)
        vectors_v = eigen_vectors[..., lvl_up].transpose(-2, -1) * valid_mask[..., mask_trans].unsqueeze(-1)

        if self.output_full_eigenvector:
            full_eigen_vectors =\
                eigen_vectors.unsqueeze(-3).expand(-1, lvl_down.shape[-1], -1, -1) *\
                valid_mask[..., mask_trans].unsqueeze(-1).unsqueeze(-1)
        else:
            full_eigen_vectors = None

        eigen_values = eigen_values.unsqueeze(-2).expand(-1, lvl_down.shape[-1], -1)
        return (vectors_u, vectors_v), (lvl_down, lvl_up),\
            transition_freq, eigen_values, full_eigen_vectors


class ResFreq(nn.Module):
    """Clss to compute EPR transitions within a user-defined frequency
    interval.

    at a fixed magnetic field, over a structured batch/mesh of spin
    systems.

    Designed for use in frequency-swept simulations (e.g., fixed-field
    EPR or THz spectroscopy), where one scans over frequency rather than
    magnetic field.
    """
    def __init__(self,
                 spin_system_dim: int,
                 mesh_size: torch.Size,
                 batch_dims: tp.Union[torch.Size, tuple],
                 output_full_eigenvector: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """Initialize the ResFreq resonance solver for spin systems.

        :param spin_system_dim: Dimension of the Hilbert space of the spin system
        (i.e., size N of the Hamiltonian matrices).
        :type spin_system_dim: int
        :param mesh_size: Shape of the spatial or parameter mesh over which resonance calculations are performed.
                          Used during output aggregation to restore tensor layout.
        :type mesh_size: torch.Size
        :param batch_dims: Shape of the batch dimensions for vectorized computation
        across multiple samples or configurations.
        :type batch_dims: torch.Size or tuple of int
        :param eigen_finder: Eigenvalue/eigenvector solver used internally for Hamiltonian diagonalization.
                             Defaults to :class:`EighEigenSolver`, which uses :func:`torch.linalg.eigh`.
        :type eigen_finder: BaseEigenSolver, optional
        :param output_full_eigenvector: If True, the forward pass returns full eigenvector matrices
        (for all energy levels)
                                        at each resonance field. If False (default), only eigenvectors of the two states
                                        involved in each transition are returned.
        :type output_full_eigenvector: bool, optional
        :param device: Device on which all internal tensors and computations are allocated (e.g., CPU or CUDA).
                       Default is CPU.
        :type device: torch.device, optional
        :param dtype: Floating-point data type for internal computations (e.g., ``torch.float32`` or ``torch.float64``).
                      Complex counterparts (``torch.complex64``/``torch.complex128``)
                      are derived automatically where needed.
        :type dtype: torch.dtype, optional
        """
        super().__init__()
        self.register_buffer("spin_system_dim", torch.tensor(spin_system_dim))
        self.output_full_eigenvector = output_full_eigenvector
        self.mesh_size = mesh_size
        self.batch_dims = batch_dims
        self.locator = Locator(output_full_eigenvector, spin_system_dim, device, dtype=dtype)
        self.device = device

    def forward(self, sample: spin_model.BaseSample,
                 resonance_field: torch.Tensor,
                 freq_low: torch.Tensor, freq_high: torch.Tensor, F: torch.Tensor, Gz: torch.Tensor) ->\
            tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor, torch.Tensor, tp.Union[torch.Tensor, None]]:
        """
        :param sample: The sample for which the resonance parameters need to be found.

        :param resonance_field: the resonance field. The shape is []
        :param freq_low: low limit of frequency intervals. The shape is [batch_dim]
        :param freq_high: high limit of frequency intervals. The shape is [batch_dim]
        :param F: The magnetic free part of Hamiltonian
        :param Gz: z-part of Zeeman magnetic field term B * Gz
        :return: list of next data:
        - tuple of the eigen vectors of high transition states and of low transition states and Vi and Vj where i>j
        is EPR transition
        - tuple of valid indexes of levels between which transition occurs
        - magnetic frequency of transitions
        - resonance energies
        - vector_full_system | None. The eigen vectors for all energy levels
        """
        config_dims = (*self.batch_dims, *self.mesh_size)
        (vectors_u, vectors_v), (lvl_down, lvl_up),\
            transition_freq, eigen_values, full_eigen_vectors = self.locator(
            F.flatten(0, -3), Gz.flatten(0, -3), freq_low.flatten(0, -1), freq_high.flatten(0, -1), resonance_field
        )

        max_columns = lvl_down.shape[-1]

        vectors_u = vectors_u.view(*config_dims, max_columns, self.spin_system_dim)
        vectors_v = vectors_v.view(*config_dims, max_columns, self.spin_system_dim)

        eigen_values = eigen_values.view(*config_dims, max_columns, self.spin_system_dim)
        transition_freq = transition_freq.view(*config_dims, max_columns)

        if full_eigen_vectors is not None:
            full_eigen_vectors = full_eigen_vectors.view(
                *config_dims, max_columns, self.spin_system_dim, self.spin_system_dim
            )
        return (vectors_u, vectors_v), (lvl_down, lvl_up),\
            transition_freq, eigen_values, full_eigen_vectors