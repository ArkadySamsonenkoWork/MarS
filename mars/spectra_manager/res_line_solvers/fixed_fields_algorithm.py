import torch
import torch.nn as nn

import typing as tp


class FixedField(nn.Module):
    """Class to compute spin Hamiltonian eigenvalues and eigenvectors at specified fixed magnetic fields.

    Evaluates the Hamiltonian H = F + B * Gz and diagonalizes it at a set of user-defined
    magnetic field points, over a structured batch/mesh of spin systems.

    Designed for field-swept simulations or direct spectral evaluation at discrete field values,
    where energy levels and eigenstates are required pointwise rather than solving for resonance conditions.
    """
    def __init__(self,
                 spin_system_dim: int,
                 mesh_size: torch.Size,
                 batch_dims: tp.Union[torch.Size, tuple],
                 output_full_eigenvector: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """Initialize the fixed-field spectral solver for spin systems.

        :param spin_system_dim: Dimension of the Hilbert space of the spin system
        (i.e., size N of the Hamiltonian matrices).
        :type spin_system_dim: int
        :param mesh_size: Shape of the spatial or parameter mesh over which resonance calculations are performed.
                          Used during output aggregation to restore tensor layout.
        :type mesh_size: torch.Size
        :param batch_dims: Shape of the batch dimensions for vectorized computation
        across multiple samples or configurations.
        :type batch_dims: torch.Size or tuple of int
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
        self.device = device

    def forward(self, resonance_fields: torch.Tensor, F: torch.Tensor, Gz: torch.Tensor) ->\
            tuple[torch.Tensor, tp.Union[torch.Tensor, None]]:
        """Compute eigenvalues and optionally eigenvectors at specified magnetic fields

        :param resonance_fields: the resonance field. The shape is [K], where K is a number of the fields
        :param F: The magnetic free part of Hamiltonian
        :param Gz: z-part of Zeeman magnetic field term B * Gz
        :return: Tuple containing:
                 - Eigenvalues of the Hamiltonian at each field point. Shape: ``[..., K, N]``
                 - Eigenvectors if ``output_full_eigenvector=True``, else ``None``. Shape: ``[..., K, N, N]``
        :rtype: tuple[torch.Tensor, torch.Tensor | None]
        """
        H = F.unsqueeze(-3) + resonance_fields.unsqueeze(-1).unsqueeze(-2) * Gz.unsqueeze(-3)
        if self.output_full_eigenvector:
            return torch.linalg.eigh(H)
        else:
            torch.linalg.eigvalsh(H), None
