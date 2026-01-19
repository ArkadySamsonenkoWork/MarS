import typing as tp

import torch
import torch.nn as nn

from .res_field_algorithm import BaseEigenSolver, EighEigenSolver
from . import spin_system


class SecSolver(nn.Module):
    """
    Computes resonance fields and associated transition data under the secular approximation,.

    where only diagonal elements of the Zeeman interaction in the eigenbasis of the field-free Hamiltonian F
    contribute to the resonance condition.

    This class assumes that the full Hamiltonian can be written as H(B) = F + B * Gz,
    and [F, Gz] = 0
    Under this approximation in their general basis, each pair of energy levels evolves linearly with magnetic field:
        E_i(B) = E_i^F + B * g_i,
    where E_i^F are eigenvalues of F and g_i are diagonal elements of Gz in the same basis.

    The resonance condition for a transition (i ← j) at frequency ν is:
        ν = E_i(B) - E_j(B) = (E_i^F - E_j^F) + B * (g_i - g_j),
    which yields an explicit solution for the resonance field:
        B_res = (ν - ΔE_F) / Δg,
    provided |Δg| exceeds a small threshold (`mz_threshold`) to avoid division by near-zero values.

    The solver:
      - Identifies all level pairs satisfying the resonance condition within [B_low, B_high],
      - Filters out transitions with negligible Zeeman splitting (|Δg| < mz_threshold),
      - Reorders eigenstates according to actual energy ordering at each B_res (since level crossing may occur),
      - Optionally returns full eigenvector sets for all states at resonance fields.

    Output includes:
      - Transition eigenvectors (lower and upper states),
      - Level indices involved in transitions (in energy-ordered basis at resonance),
      - Resonance magnetic fields,
      - Corresponding resonance energies (≈ ν, up to numerical precision),
      - Full system eigenvectors if `output_full_eigenvector=True`.
    """
    def __init__(self,
                 spin_dim: int, output_full_eigenvector: bool = False,
                 mz_threshold: float = 1e-8,
                 tolerance: float = 1e-8,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer("mz_threshold", torch.tensor(mz_threshold, device=device, dtype=dtype))
        self.register_buffer("tolerance", torch.tensor(tolerance, device=device, dtype=dtype))
        self._triu_indices = torch.triu_indices(spin_dim, spin_dim, offset=1, device=device)
        self.output_full_eigenvector = output_full_eigenvector

    def _merge_edges(self, lvl_down: torch.Tensor, lvl_up: torch.Tensor, mask_out: torch.Tensor):
        """
        Aggregates identical (lower, upper) level-pair transitions across batch elements into.

        a unified set of unique transitions.

        Returns:
          - Unique lower/upper level indices (in energy-sorted basis),
          - Column mapping from batched transitions to unified transition list,
          - Updated validity mask aligned with merged structure.
        """
        device = lvl_down.device
        dtype = lvl_down.dtype
        batch_shape = lvl_down.shape[:-1]
        n = lvl_down.shape[-1]
        total_batch_elements = batch_shape.numel()

        d_flat = lvl_down.reshape(-1)
        u_flat = lvl_up.reshape(-1)

        if d_flat.numel() == 0:
            lvl_down_out = torch.tensor([], device=device, dtype=dtype)
            lvl_up_out = torch.tensor([], device=device, dtype=dtype)
            column_indices = torch.empty(batch_shape + (0,), device=device, dtype=torch.long)
            mask = torch.empty(batch_shape + (0,), device=device, dtype=torch.bool)
            return lvl_down_out, lvl_up_out, column_indices, mask

        pairs = torch.stack([d_flat, u_flat], dim=1)
        unique_pairs = torch.unique(pairs, dim=0)
        k = unique_pairs.shape[0]

        all_u = torch.cat([u_flat, unique_pairs[:, 1]])
        max_val = all_u.max() + 1
        if max_val <= 0:
            max_val = 1

        unique_keys = unique_pairs[:, 0] * max_val + unique_pairs[:, 1]

        lvl_down_reshaped = lvl_down.reshape(total_batch_elements, n)
        lvl_up_reshaped = lvl_up.reshape(total_batch_elements, n)
        batch_keys = lvl_down_reshaped * max_val + lvl_up_reshaped

        if k > 0 and n > 0:
            unique_keys_expanded = unique_keys.view(1, k, 1)
            batch_keys_expanded = batch_keys.view(total_batch_elements, 1, n)
            matches = (unique_keys_expanded == batch_keys_expanded)
            mask_flat = matches.any(dim=2)
            matches_long = matches.long()
            column_indices_flat = matches_long.argmax(dim=2)
        else:
            mask_flat = torch.zeros((total_batch_elements, k), dtype=torch.bool, device=device)
            column_indices_flat = torch.zeros((total_batch_elements, k), dtype=torch.long, device=device)

        sorted_indices = torch.argsort(unique_keys)
        unique_pairs_sorted = unique_pairs[sorted_indices]
        lvl_down_out = unique_pairs_sorted[:, 0]
        lvl_up_out = unique_pairs_sorted[:, 1]

        if k > 0:
            indices_for_gather = sorted_indices.view(1, k).expand(total_batch_elements, k)
            column_indices_sorted = torch.gather(column_indices_flat, 1, indices_for_gather)
            mask_sorted = torch.gather(mask_flat, 1, indices_for_gather)
        else:
            column_indices_sorted = column_indices_flat
            mask_sorted = mask_flat

        new_shape = batch_shape + (k,)
        column_indices_out = column_indices_sorted.reshape(new_shape)
        mask_sorted = mask_sorted.reshape(new_shape)

        mask = torch.zeros_like(mask_sorted, dtype=mask_out.dtype)
        mask.masked_scatter_(mask_sorted, mask_out)
        return lvl_down_out, lvl_up_out, column_indices_out, mask

    def _find_resonance_fields(
            self, eigenvals_F: torch.Tensor,
            gz_eigen: torch.Tensor, B_low: torch.Tensor,
            B_high: torch.Tensor, resonance_frequency: torch.Tensor
    ):
        """
        Analytically computes candidate resonance fields for all pairs of spin levels under the secular approximation.

        For each pair (i, j), solves B_res = (ν - (E_i^F - E_j^F)) / (g_i - g_j),
        then filters results to retain only those where:
          - |g_i - g_j| > mz_threshold (to avoid singular or unphysical transitions),
          - B_res ∈ [B_low, B_high].
        Returns resonance fields and indexing information needed to reconstruct transition data per batch element.
        """
        delta_E_F = eigenvals_F[..., :, None] - eigenvals_F[..., None, :]
        delta_gz = gz_eigen[..., :, None] - gz_eigen[..., None, :]

        valid_transition_mask = torch.abs(delta_gz) > self.mz_threshold
        B_resonance = (resonance_frequency - delta_E_F) / (delta_gz + self.tolerance)

        in_range_mask = (B_resonance >= B_low[..., None, None]) &\
                        (B_resonance <= B_high[..., None, None]) & valid_transition_mask

        union_mask = in_range_mask.any(dim=0)
        low_idx, high_idx = torch.where(union_mask)

        B_resonance_union = B_resonance[..., low_idx, high_idx]
        mask_valid = (B_resonance_union >= B_low.unsqueeze(-1)) &\
                     (B_resonance_union <= B_high.unsqueeze(-1))

        indices_safe, out_mask = self._select_indices_with_padding(mask_valid)
        B_resonance_out = torch.gather(B_resonance_union, dim=-1, index=indices_safe)
        B_resonance_out = torch.where(out_mask, B_resonance_out, torch.zeros_like(B_resonance_out))

        return B_resonance_out, indices_safe, out_mask, high_idx[indices_safe], low_idx[indices_safe]

    def _select_indices_with_padding(self, mask: torch.Tensor):
        k = mask.sum(dim=-1).max().item()
        n = mask.shape[-1]
        indices_range = torch.arange(n, device=mask.device, dtype=torch.long).expand_as(mask)
        indices_with_sentinel = torch.where(mask, indices_range,
                                           torch.tensor(n, device=mask.device, dtype=torch.long))

        sorted_indices, _ = torch.sort(indices_with_sentinel, dim=-1)
        indices = sorted_indices[..., :k]

        out_mask = indices < n
        indices_safe = torch.where(out_mask, indices, torch.tensor(0, device=mask.device, dtype=torch.long))
        return indices_safe, out_mask


    def _extract_transition_levels(self,
                                   energy_to_eigenbasis: torch.Tensor,
                                   upper_level_in_eigenbasis: torch.Tensor,
                                   lower_level_in_eigenbasis: torch.Tensor):
        """
        :param energy_to_eigenbasis: energy_to_eigenbasis: Mapping from energy levels to eigenbasis, shape [..., k, n].

        :param upper_level_in_eigenbasis:  upper_level_in_eigenbasis: Upper level indices in F eigenbasis, shape [m].
        :param lower_level_in_eigenbasis: lower_level_in_eigenbasis: Lower level indices in F eigenbasis, shape [m].
        :return:
        upper_energy_level: Upper level indices in energy ordering, shape [..., m].
        lower_energy_level: Lower level indices in energy ordering, shape [..., m].
        """

        upper_energy_level = torch.gather(
            energy_to_eigenbasis,
            dim=-1,
            index=upper_level_in_eigenbasis.unsqueeze(-1)
        ).squeeze(-1)

        lower_energy_level = torch.gather(
            energy_to_eigenbasis,
            dim=-1,
            index=lower_level_in_eigenbasis.unsqueeze(-1)
        ).squeeze(-1)

        return upper_energy_level, lower_energy_level


    def _get_eigen_basis_data(self,
                              eigenvals_F: torch.Tensor,
                              gz_eigen: torch.Tensor,
                              resonance_fields: torch.Tensor,
                              i_idx: torch.Tensor, j_idx: torch.Tensor, mask: torch.Tensor):
        energies = eigenvals_F.unsqueeze(-2) + resonance_fields.unsqueeze(-1) * gz_eigen.unsqueeze(-2)

        eigenbasis_to_energy = torch.argsort(energies, dim=-1)  # from F_eigenvectors to energies
        energy_to_eigenbasis = torch.argsort(eigenbasis_to_energy, dim=-1)  # from F_eigenvectors to energies

        lvl_down, lvl_up = self._extract_transition_levels(energy_to_eigenbasis, i_idx, j_idx)
        lvl_down, lvl_up, column_indices, mask_out = self._merge_edges(lvl_down, lvl_up, mask)

        eigenbasis_to_energy = torch.gather(
            eigenbasis_to_energy, dim=-2,
            index=column_indices.unsqueeze(-1).expand(*column_indices.shape, eigenbasis_to_energy.shape[-1])
        )
        energy_to_eigenbasis = torch.gather(
            energy_to_eigenbasis, dim=-2,
            index=column_indices.unsqueeze(-1).expand(*column_indices.shape, energy_to_eigenbasis.shape[-1])
        )

        energies = torch.gather(
            energies, dim=-2,
            index=column_indices.unsqueeze(-1).expand(*column_indices.shape, energy_to_eigenbasis.shape[-1])
        )

        resonance_fields = torch.gather(
            resonance_fields, dim=-1,
            index=column_indices
        )
        out_energies = torch.gather(energies, dim=-1, index=eigenbasis_to_energy)

        return resonance_fields, out_energies, lvl_down, lvl_up, eigenbasis_to_energy, mask_out

    def forward(self, F: torch.Tensor, Gz: torch.Tensor,
                B_low: torch.Tensor, B_high: torch.Tensor,
                resonance_frequency: torch.Tensor, *args):
        """
        B = (ℏω - ΔE_F) / Δgz.

        Calculate the resonance fields, where the resonance field is possible.
        :param F: Magnetic filed free stationary Hamiltonian matrix. The shape is [..., K, K],
        where K is spin system dimension
        :param Gz: Magnetic field dependant part of stationary Hamiltonian with the shape [..., K, K].
        :param B_low: The start of the interval to find roots. The shape is [...]
        :param B_high: The end of the interval to find roots. The shape is [...]
        :param resonance_frequency: The resonance frequency. The shape is []

        :return: list of next data:
        - tuple of the eigen vectors of high transition states and of low transition states and Vi and Vj where i>j
        is EPR transition
        - tuple of valid indexes of levels between which transition occurs
        - magnetic field of transitions
        - resonance energies
        - vector_full_system | None. The eigen vectors for all energy levels
        """

        gz_diag = torch.diagonal(Gz, dim1=-2, dim2=-1).real
        eigenvals_F, eigenvecs_F = torch.linalg.eigh(F)
        gz_eigen = torch.einsum('...ij,...j,...ij->...i',
                                eigenvecs_F.conj(), gz_diag, eigenvecs_F).real

        resonance_fields, indices_safe, mask_out, i_idx, j_idx =\
            self._find_resonance_fields(eigenvals_F, gz_eigen, B_low, B_high, resonance_frequency)

        resonance_fields, out_energies, lvl_down, lvl_up, eigenbasis_to_energy, mask_out =\
            self._get_eigen_basis_data(eigenvals_F, gz_eigen, resonance_fields, i_idx, j_idx, mask_out)

        n = eigenbasis_to_energy.shape[-1]
        k = eigenbasis_to_energy.shape[-2]
        eigenvecs_F = eigenvecs_F.unsqueeze(-3).expand(*eigenvecs_F.shape[:-2], k, -1, -1)

        full_eigen_vectors = torch.gather(
            eigenvecs_F,
            dim=-1,
            index=eigenbasis_to_energy.unsqueeze(-2).expand(-1, -1, n, -1)
        ) * mask_out.unsqueeze(-1).unsqueeze(-1)
        col_idx_down = lvl_down.unsqueeze(-1).unsqueeze(-1)
        col_idx_up = lvl_up.unsqueeze(-1).unsqueeze(-1)

        vectors_down = torch.gather(
            full_eigen_vectors, dim=-1, index=col_idx_down.expand(*full_eigen_vectors.shape[:-1], 1)).squeeze(-1)
        vectors_up = torch.gather(
            full_eigen_vectors, dim=-1, index=col_idx_up.expand(*full_eigen_vectors.shape[:-1], 1)).squeeze(-1)

        #print((vectors_down, vectors_up))
        #print((lvl_down, lvl_up))
        #print(full_eigen_vectors)
        return (vectors_down, vectors_up), (lvl_down, lvl_up),\
            resonance_fields, out_energies * resonance_frequency, full_eigen_vectors


class ResSecular(nn.Module):
    """
    Class that enforces the secular approximation to compute EPR resonance parameters.

    over a batched grid of spin systems.

    This approach is valid when [F, Gz] = 0 or close to 0

    Outputs match those of `ResField` for compatibility in hybrid workflows:
      - Transition eigenvectors,
      - Energy-level indices,
      - Resonance fields,
      - Resonance energies,
      - Optional full eigenvector sets.
    """
    def __init__(self, spin_system_dim: int,
                 mesh_size: torch.Size,
                 batch_dims: tp.Union[torch.Size, tuple],
                 eigen_finder: BaseEigenSolver = EighEigenSolver(), output_full_eigenvector: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """:param eigen_finder: The eigen solver that should find eigen values and eigen vectors."""
        super().__init__()
        self.register_buffer('spin_system_dim', torch.tensor(spin_system_dim))
        self.output_full_eigenvector = output_full_eigenvector
        self.solver = SecSolver(spin_system_dim, device=device, dtype=dtype)
        self.mesh_size = mesh_size
        self.batch_dims = batch_dims
        self.device = device
        self.eigen_finder = eigen_finder

    def forward(self, sample: spin_system.BaseSample,
                 resonance_frequency: torch.Tensor,
                 B_low: torch.Tensor, B_high: torch.Tensor, F: torch.Tensor, Gz: torch.Tensor) ->\
            tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor, torch.Tensor, tp.Union[torch.Tensor, None]]:
        """
        :param sample: The sample for which the resonance parameters need to be found.

        :param resonance_frequency: the resonance frequency. The shape is []
        :param B_low: low limit of magnetic field intervals. The shape is [batch_dim]
        :param B_high: high limit of magnetic field intervals. The shape is [batch_dim]
        :param F: The magnetic free part of Hamiltonian
        :param Gz: z-part of Zeeman magnetic field term B * Gz
        :return: list of next data:
        - tuple of the eigen vectors of high transition states and of low transition states and Vi and Vj where i>j
        is EPR transition
        - tuple of valid indexes of levels between which transition occurs
        - magnetic field of transitions
        - resonance energies
        - vector_full_system | None. The eigen vectors for all energy levels
        """
        return self.solver(
            F.flatten(0, -3) / resonance_frequency,
            Gz.flatten(0, -3) / resonance_frequency, B_low.flatten(0, -1),
            B_high.flatten(0, -1), resonance_frequency / resonance_frequency)