import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import torch
import typing as tp


def to_numpy(arr: tp.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert torch tensor to numpy array or ensure numpy array."""
    if torch.is_tensor(arr):
        return arr.cpu().numpy()
    return np.asarray(arr)


def plot_2d_timeresolved(
        fields: tp.Union[np.ndarray, torch.Tensor],
        time: tp.Union[np.ndarray, torch.Tensor],
        result: tp.Union[np.ndarray, torch.Tensor],
        field_unit: tp.Literal["T", "mT", "G"] = "T",
        time_unit: tp.Literal["s", "ms", "us", "ns"] = "s",
        cmap: str = "seismic",
        aspect: tp.Union[float, str] = "auto",
        interpolation: str = "bicubic",
        colorbar: bool = True,
        colorbar_label: str = "Intensity (arb. units)",
        **kwargs
) -> None:
    """
    Plot a 2D heatmap of time-resolved spectral data on current axes.

    :param fields: Array of magnetic field values (in Tesla)
    :param time: Array of time values (in seconds)
    :param result: 2D array of spectral intensities with shape (n_fields, n_time)
    :param field_unit: Unit for magnetic field display ("T", "mT", or "G")
    :param time_unit: Unit for time display ("s", "ms", "us", "ns")
    :param cmap: Colormap name (default: "coolwarm")
    :param aspect: Aspect ratio control ("auto" or numeric value)
    :param interpolation: Interpolation method for imshow
    :param colorbar: Whether to show colorbar
    :param colorbar_label: Label for colorbar
    :param kwargs: Additional arguments passed to imshow
    """
    ax = plt.gca()

    fields = to_numpy(fields)
    time = to_numpy(time)
    result = to_numpy(result).T

    field_scale = {"T": 1, "mT": 1e3, "G": 1e4}[field_unit]
    time_scale = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}[time_unit]

    fields_conv = fields * field_scale
    time_conv = time * time_scale

    abs_max = np.max(np.abs(result))
    if abs_max == 0:
        abs_max = 1e-10

    norm = TwoSlopeNorm(
        vmin=-abs_max * 1.05,
        vcenter=0,
        vmax=abs_max * 1.05
    )
    extent = [time_conv[0], time_conv[-1], fields_conv[0], fields_conv[-1]]
    img = ax.imshow(
        result,
        norm=norm,
        cmap=cmap,
        interpolation=interpolation,
        extent=extent,
        origin="lower",
        aspect=aspect,
        **kwargs
    )
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel(f"Magnetic Field ({field_unit})")
    if colorbar:
        cbar = plt.colorbar(img, ax=ax, pad=0.02)
        cbar.set_label(colorbar_label)


def plot_kinetic(
        field_value: float,
        fields: tp.Union[np.ndarray, torch.Tensor],
        time: tp.Union[np.ndarray, torch.Tensor],
        result: tp.Union[np.ndarray, torch.Tensor],
        time_unit: tp.Literal["s", "ms", "us", "ns"] = "s",
        label: tp.Optional[str] = None,
        **kwargs
) -> None:
    """
    Plot kinetic trace at a specific magnetic field value on current axes.

    :param field_value: Magnetic field value (in Tesla) to extract trace
    :param fields: Array of magnetic field values (in Tesla)
    :param time: Array of time values (in seconds)
    :param result: 2D array of spectral intensities with shape (n_fields, n_time)
    :param time_unit: Unit for time display ("s", 'ms', 'us', 'ns')
    :param label: Label for the plot line
    :param kwargs: Additional arguments passed to plot
    """
    ax = plt.gca()

    fields = to_numpy(fields)
    time = to_numpy(time)
    result = to_numpy(result)

    time_scale = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}[time_unit]
    time_conv = time * time_scale

    field_idx = np.argmin(np.abs(fields - field_value))

    ax.plot(time_conv, result[:, field_idx], label=label, **kwargs)

    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel("Intensity (arb. units)")

    if label:
        ax.legend()


def plot_field_dependence(
        time_value: float,
        fields: tp.Union[np.ndarray, torch.Tensor],
        time: tp.Union[np.ndarray, torch.Tensor],
        result: tp.Union[np.ndarray, torch.Tensor],
        field_unit: tp.Literal["T", "mT", "G"] = "T",
        label: tp.Optional[str] = None,
        **kwargs
) -> None:
    """
    Plot field dependence at a specific time point on current axes.

    :param time_value: Time value (in seconds) to extract spectrum
    :param fields: Array of magnetic field values (in Tesla)
    :param time: Array of time values (in seconds)
    :param result: 2D array of spectral intensities with shape (n_fields, n_time)
    :param field_unit: Unit for magnetic field display ('T', 'mT', or 'G')
    :param label: Label for the plot line
    :param kwargs: Additional arguments passed to plot
    """
    ax = plt.gca()

    fields = to_numpy(fields)
    time = to_numpy(time)
    result = to_numpy(result)

    field_scale = {"T": 1, "mT": 1e3, "G": 1e4}[field_unit]
    fields_conv = fields * field_scale

    time_idx = np.argmin(np.abs(time - time_value))

    ax.plot(fields_conv, result[time_idx, :], label=label, **kwargs)

    ax.set_xlabel(f"Magnetic Field ({field_unit})")
    ax.set_ylabel("Intensity (arb. units)")

    if label:
        ax.legend()
