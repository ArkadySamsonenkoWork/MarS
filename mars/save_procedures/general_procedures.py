import pathlib
import typing as tp
import os

import torch

from ..serialization import serialization
from ..spin_model import BaseSample
from ..spectra_manager import BaseResSpectra


from . import easyspin_procedures
from . import mars_procedures


def save(
        filepath: str,
        sample: tp.Optional[BaseSample] = None,
        spectra_creator: tp.Optional[BaseResSpectra] = None,
        field: tp.Optional[torch.Tensor] = None,
        format_type: str = "mars"
):
    """Save experimental and sample parameters to a file.

    :param filepath: The file path where data should be saved.
    :param sample: BaseSample instance. Default is None.
    :param spectra_creator: SpectraCreator instance. Default is None.
    :param field: The magnetic field tensor in Tesla units.
    :param format_type: {'mars', 'easyspin'}, optional.
        The output format for saved data:
        - 'mars': Saves data as a native PyTorch-based format (.json and .safetensors).
        - 'easyspin': Creates EasySpin-compatible Sys and Exp parameter files (.mat).
        Default is 'mars'.
    """
    if format_type is None:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in [".pt", ".pth", ".mars", ".json", ".safetensors"]:
            format_type = "mars"
        elif ext in [".mat", ".npz"]:
            format_type = "easyspin"
        else:
            raise ValueError(f"Cannot infer format from extension {ext}. Please specify format_type.")

    format_handlers = {
        "mars": mars_procedures.save_mars,
        "easyspin": easyspin_procedures.save_easyspin
    }

    if format_type not in format_handlers:
        raise ValueError(f"Unsupported format: {format_type}")

    return format_handlers[format_type](filepath, sample, spectra_creator, field)


def load(filepath: tp.Union[str, pathlib.Path],
         format_type: str = None, serialized: bool = False,
         device: torch.device = torch.device("cpu"),
         dtype: torch.dtype = torch.float32
         ) -> tp.Union[serialization.SerializedMarsSession, serialization.MarsSession]:
    """Load experimental and sample parameters from a file.

    :param filepath: The file path to load data from.
    :param format_type: {'mars', 'easyspin'}, optional.
        The format of the saved data:
        - 'mars': Load data from native Mars format (.json and .safetensors).
        - 'easyspin': Load EasySpin data (.mat files).
        Default is 'mars'.
    :param serialized: If True, returns the raw Serialized objects instead of reconstructed instances.
    :return: Dictionary of loaded data containing 'sample', 'creator', and 'field'.
    """
    if format_type is None:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in [".pt", ".pth", ".json", ".safetensors", ""]:
            format_type = "mars"
        elif ext in [".mat", ".npz"]:
            format_type = "easyspin"
        else:
            raise ValueError(f"Cannot infer format from extension {ext}. Please specify format_type.")
    if serialized:
        if format_type == "mars":
            return mars_procedures.load_mars_serialized(filepath, device=device, dtype=dtype)
        elif format_type == "easyspin":
            return easyspin_procedures.load_easyspin_serialized(filepath, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    else:
        if format_type == "mars":
            return mars_procedures.load_mars(filepath, device=device, dtype=dtype)
        elif format_type == "easyspin":
            return easyspin_procedures.load_easyspin(filepath, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported format: {format_type}")