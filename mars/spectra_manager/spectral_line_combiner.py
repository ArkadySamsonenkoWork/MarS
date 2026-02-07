import torch
import typing as tp


def _compute_merge_decisions(Bres_sorted: torch.Tensor, width_sorted: torch.Tensor,
                             merge_threshold: float) -> torch.Tensor:
    """
    Compute which adjacent lines should be merged.

    :param Bres_sorted: Sorted spectral positions
    :param width_sorted: Sorted line widths
    :param merge_threshold: Threshold factor for merging
    :return: Boolean tensor indicating merges
    """
    diff = Bres_sorted[:, 1:] - Bres_sorted[:, :-1]
    avg_width = (width_sorted[:, 1:] + width_sorted[:, :-1]) / 2
    should_merge = diff < merge_threshold * avg_width

    return should_merge


def _compute_group_assignments(should_merge: torch.Tensor) -> torch.Tensor:
    """
    Assign group IDs to lines based on merge decisions.

    :param should_merge: Boolean tensor indicating merges
    :return: Tensor of group assignments
    """
    batch_size, M_minus_1 = should_merge.shape
    M = M_minus_1 + 1

    group_ids = torch.zeros((batch_size, M), dtype=torch.long, device=should_merge.device)

    for b in range(batch_size):
        current_group = 0
        group_ids[b, 0] = current_group

        for i in range(1, M):
            if should_merge[b, i - 1]:
                group_ids[b, i] = group_ids[b, i - 1]
            else:
                current_group += 1
                group_ids[b, i] = current_group

    return group_ids


def _combine_groups(Bres_sorted: torch.Tensor, A_sorted: torch.Tensor, width_sorted: torch.Tensor,
                    group_ids: torch.Tensor) -> tp.Tuple[list, list, list]:
    """
    Combine lines within each group.

    :param Bres_sorted: Sorted spectral positions
    :param A_sorted: Sorted amplitudes
    :param width_sorted: Sorted widths
    :param group_ids: Group assignments
    :return: Lists of combined Bres, A, and width tensors
    """
    batch_size = Bres_sorted.shape[0]
    combined_Bres_list = []
    combined_A_list = []
    combined_width_list = []

    for b in range(batch_size):
        unique_groups = torch.unique(group_ids[b])

        n_groups = len(unique_groups)
        combined_Bres = torch.zeros(n_groups, device=Bres_sorted.device, dtype=Bres_sorted.dtype)
        combined_A = torch.zeros(n_groups, device=A_sorted.device, dtype=A_sorted.dtype)
        combined_width = torch.zeros(n_groups, device=width_sorted.device, dtype=width_sorted.dtype)

        for i, group_id in enumerate(unique_groups):
            mask = group_ids[b] == group_id

            if mask.any():
                group_A = A_sorted[b, mask]
                group_Bres = Bres_sorted[b, mask]
                group_width = width_sorted[b, mask]

                total_A = group_A.sum()

                if total_A > 0:
                    combined_Bres[i] = (group_Bres * group_A).sum() / total_A
                    combined_A[i] = total_A
                    combined_width[i] = torch.sqrt((group_width ** 2 * group_A).sum() / total_A)

        combined_Bres_list.append(combined_Bres)
        combined_A_list.append(combined_A)
        combined_width_list.append(combined_width)

    return combined_Bres_list, combined_A_list, combined_width_list


def _pad_and_reshape(combined_Bres_list: list[torch.Tensor],
                     combined_A_list: list[torch.Tensor],
                     combined_width_list: list[torch.Tensor],
                     original_shape: torch.Size) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad results to uniform shape and reshape.

    :param combined_Bres_list: List of combined Bres tensors
    :param combined_A_list: List of combined A tensors
    :param combined_width_list: List of combined width tensors
    :param original_shape: Original shape of input
    :return: Padded and reshaped tensors with mask
    """
    batch_size = len(combined_Bres_list)
    max_len = max(tensor.shape[0] for tensor in combined_Bres_list)

    padded_Bres = torch.zeros(batch_size, max_len, device=combined_Bres_list[0].device,
                              dtype=combined_Bres_list[0].dtype)
    padded_A = torch.zeros(batch_size, max_len, device=combined_A_list[0].device, dtype=combined_A_list[0].dtype)
    padded_width = torch.zeros(batch_size, max_len, device=combined_width_list[0].device,
                               dtype=combined_width_list[0].dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=combined_Bres_list[0].device)

    for i in range(batch_size):
        length = combined_Bres_list[i].shape[0]
        padded_Bres[i, :length] = combined_Bres_list[i]
        padded_A[i, :length] = combined_A_list[i]
        padded_width[i, :length] = combined_width_list[i]
        mask[i, :length] = True

    new_shape = list(original_shape[:-1]) + [max_len]
    padded_Bres = padded_Bres.reshape(new_shape)
    padded_A = padded_A.reshape(new_shape)
    padded_width = padded_width.reshape(new_shape)
    mask = mask.reshape(new_shape)

    return padded_Bres, padded_A, padded_width, mask


def combine_spectral_lines(Bres: torch.Tensor, A: torch.Tensor, width: torch.Tensor,
                           merge_threshold: float = 1.0) -> tp.Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine spectral lines that are close relative to their widths.

    :param Bres: Spectral positions tensor of shape [..., M] or [..., 1, M]
    :param A: Amplitude tensor of shape [..., M] or [..., t, M]
    :param width: Line width tensor of shape [..., M] or [..., 1, M]
    :param merge_threshold: Threshold factor for merging (distance < threshold * avg_width)
    :return: Tuple of (combined_Bres, combined_A, combined_width, mask) where mask indicates valid entries
    """

    if Bres.dim() > 1 and Bres.shape[-2] == 1:
        Bres = Bres.squeeze(-2)
    if width.dim() > 1 and width.shape[-2] == 1:
        width = width.squeeze(-2)

    if A.dim() == Bres.dim() + 1:
        return _combine_with_t_dimension(Bres, A, width, merge_threshold)
    else:
        return _combine_standard_case(Bres, A, width, merge_threshold)


def _combine_standard_case(Bres: torch.Tensor, A: torch.Tensor, width: torch.Tensor,
                           merge_threshold: float) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine spectral lines for the standard case where all tensors have same dimensions.

    :param Bres: Spectral positions tensor of shape [..., M]
    :param A: Amplitude tensor of shape [..., M]
    :param width: Line width tensor of shape [..., M]
    :param merge_threshold: Threshold factor for merging
    :return: Tuple of combined tensors and mask
    """
    original_shape = Bres.shape
    M = original_shape[-1]

    Bres_flat = Bres.reshape(-1, M)
    A_flat = A.reshape(-1, M)
    width_flat = width.reshape(-1, M)
    batch_size = Bres_flat.shape[0]

    sort_idx = torch.argsort(Bres_flat, dim=-1)
    Bres_sorted = torch.gather(Bres_flat, 1, sort_idx)
    A_sorted = torch.gather(A_flat, 1, sort_idx)
    width_sorted = torch.gather(width_flat, 1, sort_idx)

    should_merge = _compute_merge_decisions(Bres_sorted, width_sorted, merge_threshold)
    group_assignments = _compute_group_assignments(should_merge)
    combined_Bres, combined_A, combined_width = _combine_groups(
        Bres_sorted, A_sorted, width_sorted, group_assignments
    )
    return _pad_and_reshape(combined_Bres, combined_A, combined_width, original_shape)


def _combine_with_t_dimension(Bres: torch.Tensor, A: torch.Tensor, width: torch.Tensor,
                              merge_threshold: float) ->\
        tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine spectral lines when A has an extra t dimension.

    :param Bres: Spectral positions tensor of shape [..., M]
    :param A: Amplitude tensor of shape [..., t, M]
    :param width: Line width tensor of shape [..., M]
    :param merge_threshold: Threshold factor for merging
    :return: Tuple of combined tensors and mask
    """
    *batch_dims, t, M = A.shape
    batch_dims = tuple(batch_dims)
    batch_size = int(torch.prod(torch.tensor(batch_dims))) if batch_dims else 1

    if batch_dims:
        Bres_expanded = Bres.reshape(*batch_dims, 1, M)
        width_expanded = width.reshape(*batch_dims, 1, M)
        Bres_expanded = Bres_expanded.expand(*batch_dims, t, M)
        width_expanded = width_expanded.expand(*batch_dims, t, M)
    else:
        Bres_expanded = Bres.unsqueeze(0).expand(t, M)
        width_expanded = width.unsqueeze(0).expand(t, M)

    flat_shape = (batch_size * t, M) if batch_dims else (t, M)

    Bres_flat = Bres_expanded.reshape(flat_shape)
    A_flat = A.reshape(flat_shape)
    width_flat = width_expanded.reshape(flat_shape)

    sort_idx = torch.argsort(Bres_flat, dim=-1)
    Bres_sorted = torch.gather(Bres_flat, 1, sort_idx)
    A_sorted = torch.gather(A_flat, 1, sort_idx)
    width_sorted = torch.gather(width_flat, 1, sort_idx)

    if batch_dims:
        batch_first_slice = Bres_sorted[:batch_size]
        width_first_slice = width_sorted[:batch_size]

        should_merge = _compute_merge_decisions(batch_first_slice, width_first_slice, merge_threshold)

        should_merge = should_merge.repeat_interleave(t, dim=0)
    else:
        should_merge = _compute_merge_decisions(Bres_sorted, width_sorted, merge_threshold)

    group_assignments = _compute_group_assignments(should_merge)

    combined_Bres, combined_A, combined_width = _combine_groups(
        Bres_sorted, A_sorted, width_sorted, group_assignments
    )

    if batch_dims:
        combined_Bres_reshaped = []
        combined_A_reshaped = []
        combined_width_reshaped = []

        for i in range(batch_size):
            start_idx = i * t
            end_idx = (i + 1) * t

            batch_Bres = torch.stack(combined_Bres[start_idx:end_idx])
            batch_A = torch.stack(combined_A[start_idx:end_idx])
            batch_width = torch.stack(combined_width[start_idx:end_idx])

            combined_Bres_reshaped.append(batch_Bres)
            combined_A_reshaped.append(batch_A)
            combined_width_reshaped.append(batch_width)

        max_len = max(tensor.shape[1] for batch in combined_A_reshaped for tensor in [batch])

        result_shape = (*batch_dims, t, max_len)
        result_Bres = torch.zeros(result_shape, device=Bres.device, dtype=Bres.dtype)
        result_A = torch.zeros(result_shape, device=A.device, dtype=A.dtype)
        result_width = torch.zeros(result_shape, device=width.device, dtype=width.dtype)
        mask = torch.zeros(result_shape, dtype=torch.bool, device=Bres.device)

        for i in range(batch_size):
            if batch_dims:
                indices = []
                temp = i
                for dim in reversed(batch_dims):
                    indices.insert(0, temp % dim)
                    temp //= dim
                indices = tuple(indices)

                length = combined_A_reshaped[i].shape[1]
                result_Bres[indices] = combined_A_reshaped[i][:, :length]
                result_A[indices] = combined_A_reshaped[i][:, :length]
                result_width[indices] = combined_width_reshaped[i][:, :length]
                mask[indices] = True
    else:
        max_len = max(tensor.shape[0] for tensor in combined_Bres)
        result_Bres = torch.stack([t for t in combined_Bres])
        result_A = torch.stack([t for t in combined_A])
        result_width = torch.stack([t for t in combined_width])

        mask = torch.ones((t, max_len), dtype=torch.bool, device=Bres.device)

    return result_Bres, result_A, result_width, mask
