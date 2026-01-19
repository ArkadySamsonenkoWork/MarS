import torch


class BaseObjectiveFunction:
    """Base class for objective functions used in fitting simulated EPR spectra to experimental data.

    Subclasses should implement the ``__call__`` method to compute a scalar loss that quantifies
    the mismatch between predicted and target spectra. Lower values indicate better agreement.
    """

    def __init__(self):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        pass


class MSEObjective(BaseObjectiveFunction):
    """Mean Squared Error (MSE) objective for EPR spectral fitting.

    Computes the average squared difference between simulated and experimental spectra.
    """

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.mse_loss(pred, target)


class MAEObjective(BaseObjectiveFunction):
    """Mean Absolute Error (MAE) objective for EPR spectral fitting.

    Computes the average absolute difference between simulated and experimental spectra.
    Less sensitive to outliers than MSE, making it suitable when minor spectral distortions
    or noise should not dominate the optimization.
    """

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.l1_loss(pred, target)


class CrossCorrelation(BaseObjectiveFunction):
    """Normalized cross-correlation–based objective for EPR spectral alignment.

    Measures the linear correlation between simulated and experimental spectra after
    subtracting their means. Returns ``1 - correlation``, so the loss is minimized when
    the two spectra are maximally correlated. It can be used when relative shape matters more
    than absolute intensity scaling.
    """

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        vx = pred - pred.mean(dim=-1, keepdim=True)
        vy = target - target.mean(dim=-1, keepdim=True)
        corr =\
            torch.sum(vx * vy, dim=-1) /\
            (torch.sqrt(torch.sum(vx ** 2, dim=-1)) * torch.sqrt(torch.sum(vy ** 2, dim=-1)))
        return 1 - corr


class CosineSimilarity(BaseObjectiveFunction):
    """Cosine similarity–based objective for EPR spectral comparison.

    Computes the cosine similarity between simulated and experimental spectra treated as vectors,
    then returns ``1 - mean(cosine_similarity)`` across batch elements. Focuses on angular
    alignment in signal space, making it insensitive to overall amplitude differences.
    Particularly helpful when only the spectral profile - not its magnitude - should guide fitting.
    """

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1)
        loss = 1 - cos_sim.mean()
        return loss
