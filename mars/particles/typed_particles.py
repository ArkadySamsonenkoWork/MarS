import typing as tp
from dataclasses import dataclass, field, InitVar
import os
import pickle
import torch
import math
import threading


@dataclass(frozen=True)
class SpinMatricesHalf:
    """Spin matrices for spin-1/2 particles."""
    x: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 0.5], [0.5, 0]], dtype=torch.complex64))
    y: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, -0.5j], [0.5j, 0]], dtype=torch.complex64))
    z: torch.Tensor = field(default_factory=lambda: torch.tensor([[0.5, 0], [0, -0.5]], dtype=torch.complex64))
    plus: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64))
    minus: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 0], [1, 0]], dtype=torch.complex64))

    @property
    def matrices(self):
        return [self.x, self.y, self.z]


@dataclass(frozen=True)
class SpinMatricesOne:
    """Spin matrices for spin-1/2 particles."""
    x: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 0.5], [0.5, 0]], dtype=torch.complex64))
    y: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, -0.5j], [0.5j, 0]], dtype=torch.complex64))
    z: torch.Tensor = field(default_factory=lambda: torch.tensor([[0.5, 0], [0, -0.5]], dtype=torch.complex64))
    plus: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64))
    minus: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 0], [1, 0]], dtype=torch.complex64))

    @property
    def matrices(self):
        return [self.x, self.y, self.z]


# Лучше этим пользоваться
def get_spin_operators(spin: tp.Union[float, int],
                       device: torch.device = torch.device("cpu"),
                       complex_dtype: torch.dtype = torch.complex64):

    """Generate spin matrices for a given spin s.
    :param: spin: the value of spin
    :param device: device to compute (cpu / gpu)
    :param complex_dtype: complex64/complex128
    """
    spin = float(spin)
    dim = int(2 * spin + 1)
    if not (2 * spin).is_integer():
        raise ValueError("Spin must be an integer or half-integer.")

    sz = torch.diag(torch.tensor([spin - i for i in range(dim)], dtype=complex_dtype, device=device))
    splus = torch.zeros((dim, dim), dtype=complex_dtype, device=device)
    sminus = torch.zeros((dim, dim), dtype=complex_dtype, device=device)

    for i in range(dim):
        m_i = spin - i
        if m_i + 1 <= spin:
            j = i - 1
            value = math.sqrt((spin - m_i) * (spin + m_i + 1))
            splus[j, i] = value
        if m_i - 1 >= -spin:
            j = i + 1
            value = math.sqrt((spin + m_i) * (spin - m_i + 1))
            sminus[j, i] = value

    sx = (splus + sminus) / 2
    sy = (splus - sminus) / (2j)
    return {
        "x": sx,
        "y": sy,
        "z": sz,
        "plus": splus,
        "minus": sminus,
        "matrices": (sx, sy, sz)
    }


@dataclass
class Particle:
    """Represents a particle with spin and associated matrices.

    Spin must be an integer or half-integer.
    """
    spin: float

    device: InitVar[torch.device] = torch.device("cpu")
    complex_dtype: InitVar[torch.dtype] = torch.complex64

    spin_matrices: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = field(init=False)
    identity: torch.Tensor = field(init=False)

    def __post_init__(self, device: torch.device, complex_dtype: torch.dtype):
        dim = int(2 * self.spin + 1)
        self.identity = torch.eye(dim, dtype=complex_dtype, device=device)
        self.spin_matrices = get_spin_operators(self.spin, device, complex_dtype)["matrices"]


@dataclass
class Electron(Particle):
    """Represents the electron Particle.
    Spin must be an integer or half-integer.
    """


class Nucleus(Particle):
    """Represents a nucleus with spin and g-factor loaded from a pre-parsed
    database."""
    _isotope_data = None
    _data_loaded = False  # To load data only one time
    _load_lock = threading.Lock()   # Ensures thread-safe lazy loading

    def __init__(self, nucleus_str: str, device: torch.device, complex_dtype: torch.dtype):
        self.nucleus_str = nucleus_str
        self._ensure_data_loaded()
        spin, g_factor = self._parse_nucleus_str(nucleus_str)
        super().__init__(spin, device, complex_dtype)
        self.g_factor = torch.tensor(
            g_factor, device=device,
            dtype=torch.float64 if complex_dtype == torch.complex128 else torch.float32)

    @classmethod
    def _ensure_data_loaded(cls) -> None:
        """Thread-safe lazy loading of the isotope database (Double-Checked Locking)."""
        if not cls._data_loaded:
            with cls._load_lock:
                if not cls._data_loaded:
                    data_path = cls._get_data_path("nuclei_db", "nuclear_data.pkl")
                    cls._load_isotope_data(data_path)

    @classmethod
    def _load_isotope_data(cls, data_path: str):
        """Load isotope data from a pickle file."""
        try:
            with open(data_path, "rb") as f:
                cls._isotope_data = pickle.load(f)
            cls._data_loaded = True
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Isotope data file '{data_path}' not found.")

    @staticmethod
    def _get_data_path(*parts: str) -> str:
        """Get the absolute path to the data file, relative to the location of
        this class."""
        class_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(class_dir, *parts)

    @classmethod
    def _get_nucleus_data(cls, nucleus_str: str) -> tp.Dict[str, tp.Any]:
        """Internal helper to fetch nucleus data, ensuring the database is loaded."""
        cls._ensure_data_loaded()
        if cls._isotope_data is None:
            raise RuntimeError("Isotope data failed to load.")
        data = cls._isotope_data.get(nucleus_str)
        if data is None:
            raise KeyError(f"No data found for nucleus: '{nucleus_str}'")
        return data

    def _parse_nucleus_str(self, nucleus_str: str) -> tuple[float, float]:
        """Extract spin and g-factor of a given nucleus."""
        data = self._get_nucleus_data(nucleus_str)
        return float(data["spin"]), float(data["gn"])

    @staticmethod
    def get_spin(nucleus_str: str) -> float:
        """Fast lookup of nuclear spin from the cached isotope database."""
        return float(Nucleus._get_nucleus_data(nucleus_str)["spin"])

    @staticmethod
    def get_g_factor(nucleus_str: str) -> float:
        """Fast lookup of nuclear g-factor from the cached isotope database."""
        return float(Nucleus._get_nucleus_data(nucleus_str)["gn"])
