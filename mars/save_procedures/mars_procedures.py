import pathlib
import typing as tp
import torch

from ..spin_model import BaseSample
from ..spectra_manager import BaseResSpectra

from ..serialization import serialization


def save_mars(filepath: tp.Union[str, pathlib.Path],
              sample: tp.Optional[BaseSample],
              spectra_creator: tp.Optional[BaseResSpectra],
              field: tp.Optional[torch.Tensor]):
    """Save data in the native Mars format.

    :param filepath: The file path or folder to save the data.
    :param sample: BaseSample instance.
    :param spectra_creator: SpectraCreator instance.
    :param field: The magnetic field tensor in Tesla units.
    """
    temperature = None
    if spectra_creator is not None and hasattr(spectra_creator, "intensity_calculator"):
        temperature = spectra_creator.intensity_calculator.temperature

    session = serialization.MarsSession(
        sample=sample,
        creator=spectra_creator,
        field=field,
        temperature=temperature
    )
    session.to_file(filepath)


def load_mars(filepath: tp.Union[str, pathlib.Path],
              device: torch.device = torch.device("cpu"),
              dtype: torch.dtype = torch.float32) -> serialization.MarsSession:
    """Load data from the native Mars format.

    :param filepath: The file path or folder to load data from.
    :param device: Torch device to place the tensors on.
    :param dtype: Torch data type for the tensors.
    :return: A fully instantiated MarsSession.
    """
    return serialization.MarsSession.from_file(filepath, device=device, dtype=dtype)


def load_mars_serialized(filepath: tp.Union[str, pathlib.Path],
                         device: torch.device = torch.device("cpu"),
                         dtype: torch.dtype = torch.float32
                         ) -> serialization.SerializedMarsSession:
    """Load serialized data from the native Mars format.

    :param filepath: The file path or folder to load data from.
    :return: A SerializedMarsSession dataclass.
    """
    return serialization.SerializedMarsSession.from_file(filepath, device=device, dtype=dtype)
