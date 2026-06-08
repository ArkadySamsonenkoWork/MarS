import typing as tp

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import optuna
from scipy.spatial import KDTree
import hdbscan

from .spece_searcher_plot import SpaceSearcherPlots
from .fitter import FitResult, BaseSpectrumFitter, NevergradTrial


class SpaceSearcher:
    """
    For some cases not only the best fitting parameters are useful but all
    'good' parameters.

    This searcher first identifies local minima among all trials (points whose loss
    is lower than the loss of their k nearest neighbours in the scaled parameter space).
    From those local minima, it selects those with loss below a cutoff relative to the
    global best loss, and then chooses the ones that are farthest from the global best
    parameter vector (scaled Euclidean distance), respecting a distance fraction.
    """

    def __init__(
        self,
        loss_rel_tol: tp.Optional[float] = 1.0,
        k_neighbors: int = 3,
        distance_fraction: float = 0.2,
    ):
        """
        :param loss_rel_tol: Multiplicative tolerance relative to the best loss.
            A trial is considered "good" if its loss ≤ best_loss * (1 + loss_rel_tol).
            Default is 1.0. If it is None, then all trials are computed
        :param k_neighbors: Number of nearest neighbours (in scaled parameter space)
            to consider when determining whether a point is a local minimum.
            A point is a local minimum if its loss is strictly less than the loss
            of all its k nearest neighbours. Default is 3.
        :param distance_fraction: Minimum distance fraction from the global best.
            Among the local minima that also satisfy the loss cutoff, only those
            with scaled Euclidean distance > distance_fraction * max_distance
            (the maximum distance among all considered minima) are kept.
            The actual number returned may be less than the number of such minima.
            Default is 0.2.
        """
        if loss_rel_tol is None:
            self.loss_rel_tol = float("inf")
        else:
            self.loss_rel_tol = loss_rel_tol
        self.k_neighbors = k_neighbors
        self.distance_fraction = float(distance_fraction)

    def _parse_trials(self, trials: list[tp.Union[NevergradTrial, optuna.Trial]], param_names: list[str]):
        param_rows = []
        losses = []
        trial_ids = []
        for t in trials:
            if t.value is None:
                continue
            vals = []
            for name in param_names:
                if name not in t.params:
                    vals = None
                    break
                vals.append(float(t.params[name]))
            if vals is None:
                continue
            param_rows.append(vals)
            losses.append(float(t.value))
            trial_ids.append(t._trial_id)
        if len(param_rows) == 0:
            return np.zeros((0, 0)), np.array([]), []
        P = np.asarray(param_rows, dtype=float)
        L = np.asarray(losses, dtype=float)
        return P, L, np.asarray(trial_ids, dtype=np.int32)

    def _extract_trials_from_fit(self, fit_result: FitResult,
                                   param_names: tp.Optional[list[str]] = None):
        """
        Return arrays: (param_matrix, losses, trial_indices).

        param_matrix shape: (n_trials, n_varying_params)
        losses: array of length n_trials (float)
        trial_indices: list of optuna trial numbers corresponding to rows
        """
        backend = fit_result.optimizer_info["backend"]
        opt_info = fit_result.optimizer_info

        if backend == "nevergrad":
            trials = opt_info.get("trials", [])
        elif backend == "optuna":
            if "study" in opt_info:
                trials = [t for t in opt_info["study"].trials if t.state.is_finished()]
            elif "trials" in opt_info:
                trials = [t for t in opt_info["trials"] if t.get("state") == "COMPLETE"]
            else:
                trials = []
        else:
            raise KeyError(f"Unknown backend: {backend}")

        if len(trials) == 0:
            return np.zeros((0, 0)), np.array([]), []

        if param_names is None:
            first = trials[0]
            p_dict = first.params if hasattr(first, "params") else first.get("params", {})
            param_names = list(p_dict.keys())
        return trials, param_names

    def _detect_local_minima(self, P_scaled: np.ndarray, losses: np.ndarray) -> np.ndarray:
        """
        Identify indices of points that are local minima in the scaled parameter space.

        A point is a local minimum if its loss is strictly lower than the loss
        of all its k nearest neighbours (Euclidean distance in scaled space).

        :param P_scaled: (n_points, n_dims) scaled parameter matrix.
        :param losses: (n_points,) array of loss values.
        :return: boolean mask of shape (n_points,) where True indicates a local minimum.
        """
        n = len(P_scaled)
        if n == 0:
            return np.zeros(0, dtype=bool)
        if n <= self.k_neighbors:
            return np.ones(n, dtype=bool)

        tree = KDTree(P_scaled)
        dist, idx = tree.query(P_scaled, k=self.k_neighbors + 1)
        neighbor_losses = losses[idx[:, 1:]]

        is_min = np.all(losses[:, np.newaxis] < neighbor_losses, axis=1)
        return is_min

    def __call__(self, fit_result: FitResult, param_names: tp.Optional[tp.List[str]] = None) ->\
            tp.List[tp.Dict[str, tp.Any]]:
        """
       Find diverse local minima among the optimisation trials.

       Steps:
       1. Extract all completed trials and their parameters/losses.
       2. Scale parameters to zero mean and unit variance (StandardScaler).
       3. Identify local minima: points whose loss is lower than the loss of their
          `k_neighbors` nearest neighbours (default k=3) in the scaled space.
       4. From these minima, keep those with loss ≤ best_loss * (1 + loss_rel_tol).
       5. From the kept minima, select the ones farthest from the global best
          (in scaled Euclidean distance).

       :param fit_result: The output of fitter (contains optimisation history).
       :param param_names: The names of parameters that should be included in search.
           Default None means all varying parameters from the spec are used.
       :return: List of dictionaries, each containing:
           - "trial_number": int
           - "params": dict of parameter values
           - "delta": dict of parameter differences from best_params
           - "loss": float
           - "distance": float (scaled Euclidean distance from global best)
       """
        trials, param_names = self._extract_trials_from_fit(fit_result, param_names)
        P, L, trial_numbers = self._parse_trials(trials, param_names)
        best_params = fit_result.best_params

        if P.size == 0 or L.size == 0:
            return []

        scaler = StandardScaler()
        P_scaled = scaler.fit_transform(P)

        best_loss = float(L.min())
        loss_cutoff = best_loss * (1.0 + self.loss_rel_tol)
        if self.k_neighbors != 0:
            is_local_min = self._detect_local_minima(P_scaled, L)
            if not np.any(is_local_min):
                return []
            good_mask = is_local_min & (L <= loss_cutoff)
        else:
            good_mask = L <= loss_cutoff

        if not np.any(good_mask):
            return []

        P_good = P_scaled[good_mask]
        L_good = L[good_mask]
        trials_good = trial_numbers[good_mask]

        best_idx_in_good = int(np.argmin(L_good))
        best_vector = P_good[best_idx_in_good].reshape(1, -1)

        distances = cdist(best_vector, P_good, metric="euclidean").flatten()
        sorted_idx = np.argsort(distances)
        sorted_idx = sorted_idx[sorted_idx != best_idx_in_good][::-1]

        max_dist = max(distances)
        if self.distance_fraction > 0:
            thresh = self.distance_fraction * max_dist
            within_thresh = [i for i in sorted_idx if distances[i] >= thresh]
            if within_thresh:
                chosen_idx = within_thresh
            else:
                chosen_idx = sorted_idx
        else:
            chosen_idx = sorted_idx

        results: tp.List[tp.Dict[str, tp.Any]] = []

        trial_map = {getattr(t, "number", getattr(t, "_trial_id", None)): t for t in trials}

        for idx in chosen_idx:
            tn = int(trials_good[idx])
            t_obj = trial_map.get(tn)
            params = getattr(t_obj, "params", {}) if t_obj is not None else {}

            delta = {}
            for key, value in params.items():
                value_best = best_params.get(key, None)
                delta_value = value - value_best
                delta[key] = delta_value

            results.append(
                {
                    "trial_number": tn,
                    "params": params,
                    "delta": delta,
                    "loss": float(L_good[idx]),
                    "distance": float(distances[idx]),
                }
            )
        return results

    def cluster_trials(
        self,
        fit_result: FitResult,
        param_names: tp.Optional[tp.List[str]] = None,
        min_cluster_size: int = 5,
        min_samples: tp.Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
    ) -> tp.Dict[str, tp.Any]:
        """Cluster 'good' trials using HDBSCAN.

        Filters trials by loss threshold, scales parameters, and runs HDBSCAN
        to identify distinct regions of parameter space with comparable loss.

        :param fit_result: Output of ``fitter.fit(...)``.
        :param param_names: Parameters to include in clustering. ``None`` uses all varying.
        :param min_cluster_size: Minimum number of points to form a cluster (HDBSCAN param).
        :param min_samples: Minimum samples for core point. ``None`` defaults to ``min_cluster_size``.
        :param cluster_selection_epsilon: Allow clusters to merge if within this distance.
        :return: Dict with keys:
            - ``"clusters"``: dict mapping cluster_id (int) to list of trial dicts
            - ``"labels"``: array of cluster labels for each good trial (-1 = noise)
            - ``"probabilities"``: cluster membership probabilities (if available)
            - ``"centroids"``: mean parameter vector per cluster
            - ``"n_good"``: number of trials passing loss cutoff
        :note: Requires ``hdbscan`` package. Noise points (label=-1) are returned
            separately under cluster_id ``-1``.
        """
        trials, param_names = self._extract_trials_from_fit(fit_result, param_names)
        P, L, trial_numbers = self._parse_trials(trials, param_names)

        if P.size == 0 or L.size == 0:
            return {"clusters": {}, "labels": np.array([]), "probabilities": None, "centroids": {}, "n_good": 0}

        best_loss = float(L.min())
        loss_cutoff = best_loss * (1.0 + self.loss_rel_tol)
        good_mask = L <= loss_cutoff

        if not np.any(good_mask):
            return {"clusters": {}, "labels": np.array([]), "probabilities": None, "centroids": {}, "n_good": 0}

        P_good = P[good_mask]
        L_good = L[good_mask]
        trials_good = trial_numbers[good_mask]

        scaler = StandardScaler()
        P_all_scaled = scaler.fit_transform(P)
        P_good_scaled = P_all_scaled[good_mask]

        min_s = min_samples if min_samples is not None else min_cluster_size
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_s,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(P_good_scaled)

        clusters: tp.Dict[int, tp.List[tp.Dict[str, tp.Any]]] = {}
        centroids: tp.Dict[int, np.ndarray] = {}

        for cid in np.unique(labels):
            mask = labels == cid
            cluster_trials = []
            for idx in np.where(mask)[0]:
                tn = int(trials_good[idx])
                t_obj = next((t for t in trials if getattr(t, "number", getattr(t, "_trial_id", None)) == tn), None)
                params = getattr(t_obj, "params", {}) if t_obj is not None else {}
                cluster_trials.append({
                    "trial_number": tn,
                    "params": params,
                    "loss": float(L_good[idx]),
                })
            clusters[int(cid)] = cluster_trials
            centroids[int(cid)] = P_good[mask].mean(axis=0)

        return {
            "clusters": clusters,
            "labels": labels,
            "centroids": centroids,
            "n_good": int(np.sum(good_mask)),
            "_P_good": P_good,
            "_L_good": L_good,
            "_param_names": param_names,
        }

    def plot_1d_marginals(
            self,
            fit_result: FitResult,
            param_names: tp.Optional[tp.List[str]] = None,
            best_params: tp.Optional[tp.Dict[str, float]] = None,
            scale_method: str = "mad",
            bins: int = 30,
            figsize: tp.Tuple[int, int] = (12, 8),
            show: bool = True,
    ) -> tp.List["plt.Figure"]:
        """Plot weighted 1D marginal distributions for each parameter.

        :param fit_result: Output of ``fitter.fit(...)``.
        :param param_names: Parameters to plot. ``None`` uses all varying.
        :param best_params: Best-fit values to mark with vertical lines.
        :param scale_method: Loss-to-weight transformation (``"mad"``, ``"sigma2_hat"``, ``"fixed"``).
        :param bins: Number of histogram bins.
        :param figsize: Figure size per subplot.
        :param show: If ``True``, display plots immediately.
        :return: List of matplotlib Figure objects.
        """
        trials, param_names = self._extract_trials_from_fit(fit_result, param_names)
        P, L, _ = self._parse_trials(trials, param_names)

        if P.size == 0:
            return []

        best_loss = float(L.min())
        loss_cutoff = best_loss * (1.0 + self.loss_rel_tol)
        good_mask = L <= loss_cutoff
        if not np.any(good_mask):
            return []

        P_good = P[good_mask]
        L_good = L[good_mask]

        if best_params is None:
            best_params = fit_result.best_params

        plotter = SpaceSearcherPlots(scale_method=scale_method)
        figs = plotter.plot_1d_marginals(
            P_good, L_good, param_names, best_params, bins, figsize
        )
        if show:
            for fig in figs:
                fig.show()
        return figs

    def plot_2d_pairs(
            self,
            fit_result: FitResult,
            param_names: tp.Optional[tp.List[str]] = None,
            pair_indices: tp.Optional[tp.List[tp.Tuple[int, int]]] = None,
            scale_method: str = "mad",
            cmap: str = "rainbow",
            figsize: tp.Tuple[int, int] = (6, 5),
            show: bool = True,
    ) -> tp.List["plt.Figure"]:
        """Plot weighted 2D scatter plots for parameter pairs.

        :param fit_result: Output of ``fitter.fit(...)``.
        :param param_names: Parameters to include. ``None`` uses all varying.
        :param pair_indices: List of ``(i, j)`` index pairs. ``None`` plots all upper-triangle.
        :param scale_method: Loss-to-weight transformation.
        :param cmap: Matplotlib colormap.
        :param figsize: Size per subplot.
        :param show: If ``True``, display plots immediately.
        :return: List of matplotlib Figure objects.
        """
        trials, param_names = self._extract_trials_from_fit(fit_result, param_names)
        P, L, _ = self._parse_trials(trials, param_names)

        if P.size == 0:
            return []
        best_loss = float(L.min())
        loss_cutoff = best_loss * (1.0 + self.loss_rel_tol)
        good_mask = L <= loss_cutoff
        if not np.any(good_mask):
            return []

        P_good = P[good_mask]
        L_good = L[good_mask]

        plotter = SpaceSearcherPlots(scale_method=scale_method)
        figs = plotter.plot_2d_pairs(
            P_good, L_good, param_names, pair_indices, cmap, figsize
        )
        if show:
            for fig in figs:
                fig.show()
        return figs

    def plot_clusters(
            self,
            cluster_result: tp.Dict[str, tp.Any],
            highlight_best: bool = True,
            figsize: tp.Tuple[int, int] = (8, 6),
            show: bool = True,
    ) -> tp.List["plt.Figure"]:
        """Plot clustered parameter space (PCA projection).

        :param cluster_result: Output of ``self.cluster_trials(...)``.
        :param highlight_best: Mark best-fit point with a star.
        :param figsize: Figure size.
        :param show: If ``True``, display plots immediately.
        :return: List of matplotlib Figure objects.
        """
        plotter = SpaceSearcherPlots()
        figs = plotter.plot_clusters(cluster_result, highlight_best, figsize)
        if show:
            for fig in figs:
                fig.show()
        return figs

    def plot_loss_distribution(
            self,
            fit_result: FitResult,
            param_names: tp.Optional[tp.List[str]] = None,
            bins: int = 50,
            figsize: tp.Tuple[int, int] = (10, 6),
            show: bool = True,
    ) -> "plt.Figure":
        """Plot distribution of losses for 'good' trials.

        :param fit_result: Output of ``fitter.fit(...)``.
        :param param_names: Parameters to filter. ``None`` uses all varying.
        :param bins: Number of histogram bins.
        :param figsize: Figure size.
        :param show: If ``True``, display plot immediately.
        :return: matplotlib Figure object.
        """
        trials, param_names = self._extract_trials_from_fit(fit_result, param_names)
        _, L, _ = self._parse_trials(trials, param_names)

        if L.size == 0:
            return None

        best_loss = float(L.min())
        loss_cutoff = best_loss * (1.0 + self.loss_rel_tol)
        good_mask = L <= loss_cutoff
        if not np.any(good_mask):
            return None

        L_good = L[good_mask]
        plotter = SpaceSearcherPlots()
        fig = plotter.plot_loss_distribution(L_good, best_loss, bins, figsize)
        if show:
            fig.show()
        return fig

    def plot_correlation_matrix(
            self,
            fit_result: FitResult,
            param_names: tp.Optional[tp.List[str]] = None,
            cmap: str = "rainbow",
            figsize: tp.Tuple[int, int] = (10, 8),
            show: bool = True,
    ) -> "plt.Figure":
        """Plot correlation matrix of parameters for 'good' trials.

        :param fit_result: Output of ``fitter.fit(...)``.
        :param param_names: Parameters to include. ``None`` uses all varying.
        :param cmap: Matplotlib colormap.
        :param figsize: Figure size.
        :param show: If ``True``, display plot immediately.
        :return: matplotlib Figure object.
        """
        trials, param_names = self._extract_trials_from_fit(fit_result, param_names)
        P, L, _ = self._parse_trials(trials, param_names)

        if P.size == 0:
            return None
        best_loss = float(L.min())
        loss_cutoff = best_loss * (1.0 + self.loss_rel_tol)
        good_mask = L <= loss_cutoff
        if not np.any(good_mask):
            return None

        P_good = P[good_mask]

        plotter = SpaceSearcherPlots()
        fig = plotter.plot_correlation_matrix(P_good, param_names, cmap, figsize)
        if show:
            fig.show()
        return fig

    def plot_parallel_coordinates(
            self,
            fit_result: FitResult,
            param_names: tp.Optional[tp.List[str]] = None,
            top_k: tp.Optional[int] = 20,
            cmap: str = "rainbow",
            figsize: tp.Tuple[int, int] = (12, 8),
            show: bool = True,
    ) -> "plt.Figure":
        """Plot parallel coordinates of top-k trials by loss.

        :param fit_result: Output of ``fitter.fit(...)``.
        :param param_names: Parameters to include. ``None`` uses all varying.
        :param top_k: Number of best trials to plot. ``None`` plots all good trials.
        :param cmap: Matplotlib colormap for loss encoding.
        :param figsize: Figure size.
        :param show: If ``True``, display plot immediately.
        :return: matplotlib Figure object.
        """
        trials, param_names = self._extract_trials_from_fit(fit_result, param_names)
        P, L, _ = self._parse_trials(trials, param_names)

        if P.size == 0:
            return None

        best_loss = float(L.min())
        loss_cutoff = best_loss * (1.0 + self.loss_rel_tol)
        good_mask = L <= loss_cutoff
        if not np.any(good_mask):
            return None

        P_good = P[good_mask]
        L_good = L[good_mask]

        if top_k is not None:
            top_idx = np.argsort(L_good)[:top_k]
            P_good = P_good[top_idx]
            L_good = L_good[top_idx]

        plotter = SpaceSearcherPlots()
        fig = plotter.plot_parallel_coordinates(P_good, L_good, param_names, cmap, figsize)
        if show:
            fig.show()
        return fig

    def plot_rescaled_loss_scatter(
            self,
            fit_result: FitResult,
            param_names: tp.Optional[tp.List[str]] = None,
            n_cols: int = 2,
            figsize: tp.Tuple[int, int] = (14, 10),
            show: bool = True,
    ) -> "plt.Figure":
        """Single figure with scatter subplots: parameters vs rescaled loss.

        Conceptually: we rescale losses so the full optimisation history
        spans a consistent [0, 1] range with the average at 0.5. This makes
        it easier to compare parameter sensitivity across different runs.

        Rescaling formula (piecewise linear):
            if loss <= avg:  scaled = 0.5 * (loss - min) / (avg - min)
            if loss > avg:   scaled = 0.5 + 0.5 * (loss - avg) / (max - avg)

        This ensures: min_loss → 0.0, avg_loss → 0.5, max_loss → 1.0

        :param fit_result: Output of ``fitter.fit(...)``.
        :param param_names: Parameters to plot. ``None`` uses all varying.
        :param n_cols: Number of columns in the subplot grid.
        :param figsize: Overall figure size (width, height) in inches.
        :param show: If ``True``, display the plot immediately.
        :return: Single matplotlib Figure with subplots.
        """
        trials, param_names = self._extract_trials_from_fit(fit_result, param_names)
        P, L, _ = self._parse_trials(trials, param_names)

        if P.size == 0 or L.size == 0 or not param_names:
            return None

        min_loss, avg_loss, max_loss = float(L.min()), float(L.mean()), float(L.max())

        plotter = SpaceSearcherPlots()
        fig = plotter.plot_rescaled_loss_scatter(
            P, L, param_names, min_loss, avg_loss, max_loss, n_cols, figsize
        )
        if show and fig is not None:
            fig.show()
        return fig

    def plot_loss_boxplots_by_bin(
            self,
            fit_result: FitResult,
            param_names: tp.Optional[tp.List[str]] = None,
            n_bins: int = 5,
            min_points_per_bin: int = 3,
            n_cols: int = 2,
            figsize: tp.Tuple[int, int] = (14, 10),
            show: bool = True,
    ) -> "plt.Figure":
        """Single figure with box-plot subplots: loss distribution across parameter bins.

        Conceptually: we split each parameter's range into bins (adaptive width),
        then show how the loss distribution changes across those bins. This helps
        identify parameter regions that consistently yield better fits.

        Binning strategy:
            - Uses quantile-based bins to ensure roughly equal population per bin
            - Bins with fewer than ``min_points_per_bin`` points are skipped
            - Bin widths can vary (narrower where data is dense)

        :param fit_result: Output of ``fitter.fit(...)``.
        :param param_names: Parameters to plot. ``None`` uses all varying.
        :param n_bins: Target number of bins (actual may be fewer due to filtering).
        :param min_points_per_bin: Skip bins with fewer points than this.
        :param n_cols: Number of columns in the subplot grid.
        :param figsize: Overall figure size (width, height) in inches.
        :param show: If ``True``, display the plot immediately.
        :return: Single matplotlib Figure with subplots.
        """
        trials, param_names = self._extract_trials_from_fit(fit_result, param_names)
        P, L, _ = self._parse_trials(trials, param_names)

        if P.size == 0 or L.size == 0 or not param_names:
            return None

        plotter = SpaceSearcherPlots()
        fig = plotter.plot_loss_boxplots_by_bin(
            P, L, param_names, n_bins, min_points_per_bin, n_cols, figsize
        )
        if show and fig is not None:
            fig.show()
        return fig
