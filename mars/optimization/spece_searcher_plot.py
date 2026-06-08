import matplotlib.pyplot as plt
import numpy as np
import typing as tp
from sklearn.decomposition import PCA

from pandas.plotting import parallel_coordinates
import pandas as pd


class SpaceSearcherPlots:
    """Visualization utilities for SpaceSearcher results.

    Transforms loss to weights for visualization:
        weight = exp( -(L - L_min) / scale )

    :param scale_method: How to compute scale for loss transformation.
        Options: ``"mad"`` (median absolute deviation), ``"sigma2_hat"``,
        ``"fixed"``, or a float value.
    :param fixed_scale: Value to use if ``scale_method="fixed"``.
    """

    def __init__(
            self,
            scale_method: str = "mad",
            fixed_scale: tp.Optional[float] = None,
    ):
        self.scale_method = scale_method
        self.fixed_scale = fixed_scale

    def _compute_weights(self, losses: np.ndarray, best_loss: float) -> np.ndarray:
        """Transform losses to weights for visualization."""
        delta = losses - best_loss
        if self.scale_method == "fixed" and self.fixed_scale is not None:
            scale = self.fixed_scale
        elif self.scale_method == "mad":
            scale = np.median(np.abs(delta - np.median(delta))) * 1.4826
            scale = max(scale, 1e-8)
        elif self.scale_method == "sigma2_hat":
            scale = np.var(delta) if np.var(delta) > 0 else 1.0
        else:
            scale = np.percentile(delta[delta > 0], 75) if np.any(delta > 0) else 1.0
        return np.exp(-delta / scale)

    def plot_1d_marginals(
            self,
            P: np.ndarray,
            losses: np.ndarray,
            param_names: tp.List[str],
            best_params: tp.Optional[tp.Dict[str, float]] = None,
            bins: int = 30,
            figsize: tp.Tuple[int, int] = (12, 8),
    ) -> tp.List[plt.Figure]:
        """Weighted 1D histograms for each parameter."""
        best_loss = float(losses.min())
        weights = self._compute_weights(losses, best_loss)
        weights = weights / weights.sum()

        figs = []
        for i, name in enumerate(param_names):
            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(P[:, i], bins=bins, weights=weights, edgecolor="black", alpha=0.7)
            ax.set_xlabel(name)
            ax.set_ylabel("Weighted frequency")
            ax.set_title(f"Marginal distribution: {name}")
            if best_params and name in best_params:
                ax.axvline(best_params[name], color="red", linestyle="--", label="Best fit")
                ax.legend()
            ax.grid(alpha=0.3)
            figs.append(fig)
        return figs

    def plot_2d_pairs(
            self,
            P: np.ndarray,
            losses: np.ndarray,
            param_names: tp.List[str],
            pair_indices: tp.Optional[tp.List[tp.Tuple[int, int]]] = None,
            cmap: str = "rainbow",
            figsize: tp.Tuple[int, int] = (6, 5),
    ) -> tp.List[plt.Figure]:
        """Weighted 2D scatter plots for parameter pairs."""
        best_loss = float(losses.min())
        weights = self._compute_weights(losses, best_loss)

        if pair_indices is None:
            n = P.shape[1]
            pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

        figs = []
        for i, j in pair_indices:
            fig, ax = plt.subplots(figsize=figsize)
            sc = ax.scatter(
                P[:, i], P[:, j],
                c=losses,
                s=20 * weights / weights.max(),
                cmap=cmap,
                alpha=0.7,
                linewidth=0.3,
            )
            ax.set_xlabel(param_names[i])
            ax.set_ylabel(param_names[j])
            ax.set_title(f"{param_names[i]} vs {param_names[j]}")
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Loss")
            figs.append(fig)
        return figs

    def plot_clusters(
            self,
            cluster_result: tp.Dict[str, tp.Any],
            highlight_best: bool = True,
            figsize: tp.Tuple[int, int] = (8, 6),
    ) -> tp.List[plt.Figure]:
        """2D PCA projection of clustered trials."""
        P_good = cluster_result.get("_P_good")
        L_good = cluster_result.get("_L_good")
        labels = cluster_result.get("labels")
        param_names = cluster_result.get("_param_names", [])

        if P_good is None or labels is None or len(param_names) < 2:
            return []

        pca = PCA(n_components=2)
        P_pca = pca.fit_transform(P_good)

        fig, ax = plt.subplots(figsize=figsize)
        best_loss = float(L_good.min())
        sizes = 50 * np.exp(-(L_good - best_loss) / np.std(L_good))
        scatter = ax.scatter(
            P_pca[:, 0], P_pca[:, 1],
            c=labels,
            s=sizes,
            cmap="tab10",
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title("Clustered parameter space (PCA projection)")
        plt.colorbar(scatter, ax=ax, label="Cluster ID (-1 = noise)")

        if highlight_best:
            best_idx = int(np.argmin(L_good))
            ax.plot(P_pca[best_idx, 0], P_pca[best_idx, 1], "gold", marker="*", markersize=15, label="Best")
            ax.legend()

        ax.grid(alpha=0.3)
        return [fig]

    def plot_loss_distribution(
            self,
            losses: np.ndarray,
            best_loss: float,
            bins: int = 50,
            figsize: tp.Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Histogram of losses with best-loss marker."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(losses, bins=bins, edgecolor="black", alpha=0.7)
        ax.axvline(best_loss, color="red", linestyle="--", linewidth=2, label=f"Best: {best_loss:.4g}")
        ax.set_xlabel("Loss")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of losses")
        ax.legend()
        ax.grid(alpha=0.3)
        return fig

    def plot_correlation_matrix(
            self,
            P: np.ndarray,
            param_names: tp.List[str],
            cmap: str = "rainbow",
            figsize: tp.Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """Correlation heatmap of parameters."""
        corr = np.corrcoef(P.T)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(param_names)))
        ax.set_yticks(range(len(param_names)))
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_yticklabels(param_names)

        for i in range(len(param_names)):
            for j in range(len(param_names)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)

        ax.set_title("Parameter correlation matrix")
        plt.colorbar(im, ax=ax, label="Correlation")
        fig.tight_layout()
        return fig

    def plot_parallel_coordinates(
            self,
            P: np.ndarray,
            losses: np.ndarray,
            param_names: tp.List[str],
            cmap: str = "rainbow",
            figsize: tp.Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """Parallel coordinates plot for top-k trials."""
        mins = P.min(axis=0)
        maxs = P.max(axis=0)
        range_ = maxs - mins
        range_[range_ == 0] = 1.0
        P_norm = (P - mins) / range_

        df = pd.DataFrame(P_norm, columns=param_names)
        df["loss"] = losses

        fig, ax = plt.subplots(figsize=figsize)
        parallel_coordinates(df, "loss", colormap=cmap, ax=ax)
        ax.set_title("Parallel coordinates (normalized parameters)")
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Normalized value")
        ax.grid(alpha=0.3)
        return fig

    def plot_rescaled_loss_scatter(
            self,
            P: np.ndarray,
            losses: np.ndarray,
            param_names: tp.List[str],
            min_loss: float,
            avg_loss: float,
            max_loss: float,
            n_cols: int = 2,
            figsize: tp.Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """Single figure with scatter subplots: parameters vs rescaled loss [0,1]."""
        if np.isclose(min_loss, max_loss):
            rescaled = np.full_like(losses, 0.5)
        elif np.isclose(avg_loss, min_loss):
            rescaled = (losses - min_loss) / (max_loss - min_loss)
        elif np.isclose(avg_loss, max_loss):
            rescaled = (losses - min_loss) / (max_loss - min_loss)
        else:
            rescaled = np.zeros_like(losses)
            mask_low = losses <= avg_loss
            mask_high = ~mask_low
            if np.any(mask_low):
                rescaled[mask_low] = 0.5 * (losses[mask_low] - min_loss) / (avg_loss - min_loss)
            if np.any(mask_high):
                rescaled[mask_high] = 0.5 + 0.5 * (losses[mask_high] - avg_loss) / (max_loss - avg_loss)
        rescaled = np.clip(rescaled, 0.0, 1.0)

        n_params = len(param_names)
        n_rows = int(np.ceil(n_params / n_cols))

        if n_rows == 1 and n_cols == 1:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = np.array([[axes]])
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        fig.suptitle("Parameter sensitivity: rescaled loss (0=best, 0.5=avg, 1=worst)",
                     fontsize=14, fontweight='bold', y=1.02)

        for idx, name in enumerate(param_names):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            param_data = P[:, idx]

            ax.scatter(param_data, rescaled, s=15, alpha=0.6, edgecolors='white', linewidth=0.3)
            ax.set_xlabel(name)
            ax.set_ylabel("Rescaled loss")
            ax.set_title(name, fontsize=10)
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.grid(alpha=0.3, axis='y')
            ax.set_ylim(-0.05, 1.05)

        for idx in range(n_params, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

        fig.tight_layout()
        return fig

    def plot_loss_boxplots_by_bin(
            self,
            P: np.ndarray,
            losses: np.ndarray,
            param_names: tp.List[str],
            n_bins: int = 5,
            min_points_per_bin: int = 3,
            n_cols: int = 2,
            figsize: tp.Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """Single figure with box-plot subplots: loss distribution across adaptive bins."""
        n_params = len(param_names)
        n_rows = int(np.ceil(n_params / n_cols))

        if n_rows == 1 and n_cols == 1:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = np.array([[axes]])
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        fig.suptitle("Loss distribution across parameter bins",
                     fontsize=14, fontweight='bold', y=1.02)

        for idx, name in enumerate(param_names):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            param_vals = P[:, idx]

            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = np.unique(np.quantile(param_vals, quantiles))

            if len(bin_edges) < 2:
                ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel(name)
                ax.set_ylabel("Loss")
                ax.set_title(name, fontsize=10)
                continue

            bin_losses = []
            bin_labels = []

            for j in range(len(bin_edges) - 1):
                left, right = bin_edges[j], bin_edges[j + 1]
                mask = (param_vals >= left) & (param_vals <= right if j == len(bin_edges) - 2 else param_vals < right)
                bin_data = losses[mask]
                if len(bin_data) >= min_points_per_bin:
                    bin_losses.append(bin_data)
                    center = (left + right) / 2
                    bin_labels.append(f"{center:.3g}\n(n={len(bin_data)})")

            if not bin_losses:
                ax.text(0.5, 0.5, f"No bins with ≥{min_points_per_bin} points",
                        ha='center', va='center', transform=ax.transAxes, fontsize=9)
                ax.set_xlabel(name)
                ax.set_ylabel("Loss")
                ax.set_title(name, fontsize=10)
                continue

            bp = ax.boxplot(bin_losses, labels=bin_labels, patch_artist=True,
                            widths=0.7, showfliers=False)

            medians = [np.median(bl) for bl in bin_losses]
            norm = plt.Normalize(vmin=min(medians), vmax=max(medians))
            cmap = plt.get_cmap('rainbow')

            for patch, median in zip(bp['boxes'], medians):
                patch.set_facecolor(cmap(norm(median)))
                patch.set_alpha(0.8)

            ax.set_xlabel(f"{name} (bin center)")
            ax.set_ylabel("Loss")
            ax.set_title(name, fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(axis='y', alpha=0.3)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
            cbar.set_label('Median', rotation=270, labelpad=12, fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        for idx in range(n_params, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

        fig.tight_layout()
        return fig