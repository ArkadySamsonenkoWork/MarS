import torch
import torch.nn as nn


class L2PairwiceObjectiveFunction(nn.Module):
    def __init__(self, n_common_points: int = 3000):
        super().__init__()
        self.n_common_points = n_common_points
        self.eps = 1e-8

    def forward(
            self,
            x: torch.Tensor,
            y1: torch.Tensor,
            y2: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        x_min = torch.min(x[:, 0], dim=-1)[0]
        x_max = torch.max(x[:, -1], dim=-1)[0]
        x_common = torch.linspace(0, 1, self.n_common_points, device=x.device, dtype=x.dtype)
        x_common = x_min + x_common * (x_max - x_min)
        x_common = x_common.expand(len(x), -1)

        y1_common = self._interpolate_batch(x, y1, x_common)
        y2_common = self._interpolate_batch(x, y2, x_common)

        diff = torch.mean(
            (y1_common.unsqueeze(-2) - y2_common.unsqueeze(-3)) ** 2, dim=-1
        )

        diff_baseline_y1 = torch.mean(
            y1_common.unsqueeze(-2) ** 2, dim=-1
        )
        diff_baseline_y2 = torch.mean(
            y2_common.unsqueeze(-2) ** 2, dim=-1
        )
        loss = 2 * diff / (diff_baseline_y1 + diff_baseline_y2 + self.eps)
        return torch.sqrt(loss)

    def _interpolate_batch(
            self,
            x_original: torch.Tensor,
            y_original: torch.Tensor,
            x_common: torch.Tensor
    ) -> torch.Tensor:
        batch_size, n_original = x_original.shape
        m_new = x_common.shape[1]

        indices = torch.searchsorted(x_original, x_common.expand(len(x_original), -1))

        indices_lower = torch.clamp(indices - 1, min=0, max=n_original - 2)
        indices_upper = torch.clamp(indices, min=0, max=n_original - 1)

        batch_indices = torch.arange(batch_size, device=x_original.device).view(batch_size, 1)
        batch_indices = batch_indices.expand(batch_size, m_new)

        x_lower = x_original[batch_indices, indices_lower]
        x_upper = x_original[batch_indices, indices_upper]
        y_lower = y_original[batch_indices, indices_lower]
        y_upper = y_original[batch_indices, indices_upper]

        denom = x_upper - x_lower
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        weights = (x_common - x_lower) / (denom + 1e-9)
        weights = torch.clamp(weights, min=0.0, max=1.0)

        y_interp = y_lower + weights * (y_upper - y_lower)

        x_min = x_original[:, 0].unsqueeze(1)
        x_max = x_original[:, -1].unsqueeze(1)
        mask = (x_common >= x_min) & (x_common <= x_max)
        y_interp = torch.where(mask, y_interp, torch.zeros_like(y_interp))

        return y_interp


class SpectraMatchingObjective(nn.CosineSimilarity):
    def forward(self, y1_feature, y2_feature):
        cos_objective = super().forward(y1_feature.unsqueeze(-2), y2_feature.unsqueeze(-3))
        return (1 - cos_objective) / 2


class CosSpectraLoss(nn.Module):
    def __init__(self, n_common_points: int = 3000, eps: float = 1e-3):
        super().__init__()
        self.l2_objective = L2PairwiceObjectiveFunction(n_common_points=n_common_points)
        self.cos_objective = SpectraMatchingObjective(dim=-1)

        self.cos_objective_test = nn.CosineSimilarity(dim=-1)
        self.l1_loss = nn.L1Loss(reduction="none")

        self.eps = eps

    def forward(self, fields, spec, spec_distorted, out_feature, mask, batch_size_per_graph):
        start_idx = 0
        total_sum = 0
        total_count = 0
        l2_pair_objective_sum = 0
        for idx, batch_size in enumerate(batch_size_per_graph):
            batch_mask = mask[idx]
            """
            l1_objective =\
                self.l1_loss(spec[start_idx: start_idx+batch_size], spec_distorted[start_idx: start_idx+batch_size])
            l1_objective = l1_objective.mean(dim=-1)
            l1_objective = torch.clip(l1_objective, min=-1, max=1) * 4


            loss = (out_feature[idx] - l1_objective)
            total_sum += (loss**2).sum()
            """

            l2_pair_objective = torch.clip(self.l2_objective(
                fields[start_idx: start_idx+batch_size],
                spec[start_idx: start_idx+batch_size],
                spec_distorted[start_idx: start_idx+batch_size]), min=0, max=1
            ) / 2

            #denom = (l2_pair_objective + self.eps)
            loss = (out_feature[idx] - l2_pair_objective)
            total_sum += (loss**2).sum()
            total_count += loss.numel()

            start_idx += batch_size
        return total_sum / total_count