import copy
import os
import json

import pathlib
from dataclasses import dataclass
import typing as tp
import math
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import torch

import nevergrad as ng
import optuna

from ..spectra_processing import normalize_spectrum, normalize_spectrum2d
from . import objectives
from optuna_dashboard import run_server


class TrialResult(tp.TypedDict):
    trial_number: int
    params: dict[str, float]
    delta: dict[str, float]
    loss: float
    distance: float


def print_trial_results(results: tp.Union[TrialResult, list[TrialResult]], max_params=None, precision=6) -> None:
    """Print trial results.

    :param results: Single trial dict or list of trial dicts
    :param max_params: Maximum number of parameters to display (None for
        all)
    :param precision: Number of decimal places for numeric values
    :return: None
    """
    if isinstance(results, dict):
        results = [results]

    for i, trial in enumerate(results):
        if i > 0:
            print("\n" + "=" * 80 + "\n")

        # Print header
        print(f"TRIAL #{trial['trial_number']}")
        print(f"index: {i}")
        print("-" * 40)
        print(f"Loss:     {trial['loss']:.{precision}f}")
        print(f"Distance: {trial['distance']:.{precision}f}")
        print()

        params = trial["params"]
        deltas = trial["delta"]
        param_names = list(params.keys())

        if max_params and len(param_names) > max_params:
            param_names = param_names[:max_params]
            truncated = True
        else:
            truncated = False

        print("PARAMETERS:")
        print("-" * 40)
        max_name_len = max(len(name) for name in param_names) if param_names else 0
        for param_name in param_names:
            value = params[param_name]
            delta = deltas.get(param_name, None)

            if isinstance(value, float):
                if abs(value) > 1000000:
                    value_str = f"{value:.{precision - 2}e}"
                else:
                    value_str = f"{value:.{precision}f}"
            else:
                value_str = str(value)

            if isinstance(delta, float):
                if abs(delta) > 1000000:
                    delta_str = f"{delta:+.{precision - 2}e}"
                else:
                    delta_str = f"{delta:+.{precision}f}"
            else:
                delta_str = str(delta)

            print(f"  {param_name:<{max_name_len}} = {value_str:>15} (Δ {delta_str})")
        if truncated:
            remaining = len(params) - max_params
            print(f"  ... and {remaining} more parameters")


def print_params(params: dict[str, float], max_params=None, precision=6) -> None:
    """
    :param params: the dict of parameter names and their values.

    :param max_params: maximum number of parameters
    :param precision: Number of decimal places for numeric values
    :return: None
    """

    param_names = list(params.keys())
    if max_params and len(param_names) > max_params:
        param_names = param_names[:max_params]
        truncated = True
    else:
        truncated = False

    print("PARAMETERS:")
    print("-" * 40)
    max_name_len = max(len(name) for name in param_names) if param_names else 0

    for param_name, in param_names:
        value = params[param_name]

        if isinstance(value, float):
            if abs(value) > 1000000:
                value_str = f"{value:.{precision - 2}e}"
            else:
                value_str = f"{value:.{precision}f}"
        else:
            value_str = str(value)

        print(f"  {param_name:<{max_name_len}} = {value_str:>15} ")

    if truncated:
        remaining = len(params) - max_params
        print(f"  ... and {remaining} more parameters")


@dataclass
class FitResult:
    best_params: tp.Dict[str, float]
    best_loss: float
    best_spectrum: tp.Optional[torch.Tensor]
    optimizer_info: tp.Dict

    def to_json_dict(self) -> dict:
        """Return a JSON‑compatible dict (spectrum is omitted, optimizer_info is sanitized)."""
        info = self.optimizer_info
        backend = info.get("backend")
        serialized_info = {"backend": backend}

        if backend == "optuna":
            study = info.get("study")
            if study is not None:
                serialized_info["trials"] = [
                    {
                        "trial_id": t.number,
                        "params": t.params,
                        "value": t.value,
                        "state": t.state.name if hasattr(t.state, "name") else str(t.state),
                    }
                    for t in study.trials
                ]
                serialized_info["best_trial_id"] = study.best_trial.number

        elif backend == "nevergrad":
            trials = info.get("trials", [])
            serialized_info["optimizer"] = info.get("optimizer")
            serialized_info["trials"] = [
                {
                    "trial_id": t._trial_id,
                    "params": t.params,
                    "value": t.value,
                }
                for t in trials
            ]

        else:
            for k, v in info.items():
                if k != "backend":
                    try:
                        json.dumps(v)
                        serialized_info[k] = v
                    except TypeError:
                        serialized_info[k] = repr(v)

        return {
            "best_params": self.best_params,
            "best_loss": self.best_loss,
            "optimizer_info": serialized_info,
        }

    def save(self, path: tp.Union[str, pathlib.Path], save_spectrum: bool = True) -> None:
        """Save FitResult to disk.

        :param path: Base filename (without extension). Creates ``path.json``
                     and optionally ``path_spectrum.pt``.
        :param save_spectrum: If True, save the best_spectrum tensor(s) as well.
        """
        json_path = path + ".json"
        with open(json_path, "w") as f:
            json.dump(self.to_json_dict(), f, indent=2)
        if save_spectrum and self.best_spectrum is not None:
            torch.save(self.best_spectrum, path + "_spectrum.pt")

    @classmethod
    def load(cls, path: str, load_spectrum: bool = True) -> "FitResult":
        """Load FitResult from disk."""
        json_path = path + ".json"
        with open(json_path, "r") as f:
            data = json.load(f)

        spectrum = None
        if load_spectrum:
            spec_path = path + "_spectrum.pt"
            try:
                spectrum = torch.load(spec_path)
            except FileNotFoundError:
                pass

        return cls(
            best_params=data["best_params"],
            best_loss=data["best_loss"],
            best_spectrum=spectrum,
            optimizer_info=data["optimizer_info"],
        )


@dataclass
class ExperementalParameters:
    best_params: tp.Dict[str, float]
    best_loss: float
    best_spectrum: tp.Optional[torch.Tensor]
    optimizer_info: tp.Dict


@dataclass
class NevergradTrial:
    params: tp.Dict[str, float]
    value: float
    _trial_id: int

    def __repr__(self):
        return f"_trial_id: {self._trial_id}, loss: {self.value}"

    def __str__(self):
        return f"_trial_id: {self._trial_id}, loss: {self.value}"


class TrialsTracker:
    def __init__(self):
        self.trials = []
        self.losses = []
        self.step = 0

    def __call__(self, optimizer: ng.optimization.Optimizer,
                 candidate: ng.p.Instrumentation, loss: float):
        """Callback function called after each evaluation."""
        self.trials.append(candidate.value[0])
        self.losses.append(loss)
        self.step += 1

        # Optional: print progress
        if self.step % 10 == 0:
            print(f"Step {self.step}: Loss = {loss:.6f}")

    def get_best_trial(self):
        """Get the trial with the lowest loss."""
        best_idx = np.argmin(self.losses)
        return {
            '_trial_id': best_idx + 1,
            'params': self.trials[best_idx],
            'value': self.losses[best_idx]
        }

    def get_all_trials(self):
        """Get all trials as a list of dictionaries."""
        return [
            {
                '_trial_id': i + 1,
                'params': trial,
                'value': loss
            }
            for i, (trial, loss) in enumerate(zip(self.trials, self.losses))
        ]


class LogTransform:
    def __call__(self, x: float) -> float:
        return math.pow(10, x)

    def inverse(self, y: float) -> float:
        return math.log(y)


@dataclass
class ParamSpec:
    """Specification for a single scalar parameter.

    Attributes:
        name: parameter name

        bounds: (low, high) bounds for optimizer search (floats)

        default: optional default value to use for initialization

        transform: optional callable applied to a raw optimizer value to map
                   it to the physical parameter (for example, log-scales)

        vary: bool: Whether the parameter should vary or not.
        In the latter case, this is equivalent to specifying the parameter in fixed_parameters.
        If you don't plan to vary the parameter, then the more correct way is to specify it in fixed_parameters.
    """
    name: str
    bounds: tp.Tuple[float, float]
    default: tp.Optional[float] = None
    transform: tp.Optional[tp.Callable[[float], float]] = None
    vary: bool = True

    def clip(self, x: float) -> float:
        """
        :param x: the value that should be clipped with respect to bounds
        :return: clipped values
        """
        lo, hi = self.bounds
        return float(min(max(x, lo), hi))

    def apply(self, x: float) -> float:
        """
        Gives the real value if transformed is skipped or return trasformed value
        :param x:
        :return:
        """
        x = self.clip(x)
        return self.transform(x) if self.transform is not None else x

    def set_bounds(self, bounds: tp.Tuple[float, float]):
        """Update the bounds for this parameter spec."""
        self.bounds = bounds


class ParameterSpace:
    print_precision: int = 4

    def __init__(self, specs: tp.Sequence[ParamSpec],
                 fixed_params: tp.Optional[tp.Dict[str, float]] = None):
        """
        :param specs: The sequence of ParamSpec instances.

        The list include parameters that should be varied (if spec.vary = True. For more details
        see ParamSpec documentation)
        :param fixed_params: The parameters that are fixed during fit.
        """
        self.specs = list(specs)

        self.fixed_params: tp.Dict[str, float] = {} if fixed_params is None else dict(fixed_params)
        self.fixed_params.update({s.name: s.default for s in self.specs if not getattr(s, "vary")})

        self._varying_specs = [s for s in self.specs if getattr(s, "vary", True)]
        self.varying_names = [s.name for s in self._varying_specs]
        self.varying_params = {s.name: s.default for s in self._varying_specs}

        for name in list(self.fixed_params.keys()):
            if name in self.varying_names:
                idx = next(i for i, s in enumerate(self._varying_specs) if s.name == name)
                del self._varying_specs[idx]
                self.varying_names.remove(name)

    def __deepcopy__(self, memo):
        new_obj = type(self).__new__(type(self))

        new_obj.specs = copy.deepcopy(self.specs, memo)
        new_obj.fixed_params = copy.deepcopy(self.fixed_params, memo)
        new_obj._varying_specs = copy.deepcopy(self._varying_specs, memo)
        new_obj.varying_names = copy.deepcopy(self.varying_names, memo)
        new_obj.varying_params = copy.deepcopy(self.varying_params, memo)
        new_obj.print_precision = self.print_precision
        return new_obj

    def __getitem__(self, key: str):
        try:
            return self.fixed_params[key]
        except KeyError:
            try:
                return self.varying_params[key]
            except KeyError:
                raise KeyError(f"Key '{key}' not found in fixed_params or _varying_specs")

    def __setitem__(self, key: str, value: float):
        if key in self.fixed_params:
            self.fixed_params[key] = value
        elif key in self.varying_names:
            for spec in self._varying_specs:
                if spec.name == key:
                    spec.default = value
                    self.varying_params[key] = value
        else:
            raise KeyError(f"Key '{key}' not found in fixed_params or varying_params")

    def __dict__(self) -> dict[str, float]:
        return {**self.varying_params, **self.fixed_params}

    @classmethod
    def from_json_dict(cls, data: dict) -> "ParameterSpace":
        """Reconstruct a ParameterSpace from a dictionary."""
        specs = []
        transform_registry = {"LogTransform": LogTransform()}
        for s_dict in data["specs"]:
            transform = None
            if "transform" in s_dict:
                transform = transform_registry.get(s_dict["transform"])
            spec = ParamSpec(
                name=s_dict["name"],
                bounds=tuple(s_dict["bounds"]),
                default=s_dict.get("default"),
                transform=transform,
                vary=s_dict.get("vary", True),
            )
            specs.append(spec)
        return cls(specs, fixed_params=data.get("fixed_params"))

    def __iter__(self):
        return iter(self.__dict__().items())

    def __repr__(self) -> str:
        """Print parameters space."""
        text = ""
        text += f"____Fixed parameters_____ \n"
        text += "-" * 40 + "\n"
        param_names = list(self.fixed_params.keys())
        for key, value in self.fixed_params.items():
            max_name_len = max(len(name) for name in param_names) if param_names else 0
            if isinstance(value, float):
                if abs(value) > 1000:
                    value_str = f"{value:.{self.print_precision - 2}e}"
                else:
                    value_str = f"{value:.{self.print_precision}f}"
            else:
                value_str = str(value)
            text += f"  {key:<{max_name_len}} = {value_str:>15}\n"

        text += "\n\n"
        text += f"______Varying parameters_____ \n"
        text += "-" * 40 + "\n"

        param_names = list(self.varying_names)
        for spec in self._varying_specs:
            max_name_len = max(len(name) for name in param_names) if param_names else 0

            value = spec.default
            name = spec.name
            (low, up) = spec.bounds

            if isinstance(value, float):
                if abs(value) > 1000:
                    value_str = f"{value:.{self.print_precision - 2}e}"
                else:
                    value_str = f"{value:.{self.print_precision}f}"
            else:
                value_str = str(value)

            if isinstance(low, float):
                if abs(low) > 1000000:
                    low = f"{low:+.{self.print_precision - 2}e}"
                else:
                    low = f"{low:+.{self.print_precision}f}"
            else:
                low = str(low)

            if isinstance(up, float):
                if abs(up) > 1000:
                    up = f"{up:+.{self.print_precision - 2}e}"
                else:
                    up = f"{up:+.{self.print_precision}f}"
            else:
                up = str(up)
            text += f"  {name:<{max_name_len}} = {value_str:>15}   (low:   {low}  up:   {up})\n"

        return text

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self, name: str, value: tp.Optional[float] = None):
        """Freeze a parameter by name.

        If value is provided, use it; otherwise use its default (or
        current) value.
        """
        if name not in self.varying_names:
            raise KeyError(name)
        spec = next(s for s in self.specs if s.name == name)

        if value is None:
            if spec.default is not None:
                value = float(spec.default)
            else:
                lo, hi = spec.bounds
                value = 0.5 * (lo + hi)
        self.fixed_params[name] = float(value)

        self._varying_specs = [s for s in self._varying_specs if s.name != name]
        self.varying_names = [s.name for s in self._varying_specs]
        self.varying_params = {s.name: s.default for s in self._varying_specs}

    def unfreeze(self, name: str):
        """Unfreeze a parameter previously frozen with `freeze` or
        fixed_params."""
        if name in self.fixed_params:
            del self.fixed_params[name]

        for s in self.specs:
            if s.name == name and s not in self._varying_specs and getattr(s, 'vary', True):
                self._varying_specs.append(s)
                self.varying_names.append(s.name)
        self.varying_params = {s.name: s.default for s in self._varying_specs}

    def vector_to_dict(self, vec: tp.Sequence[float]) -> tp.Dict[str, float]:
        """Convert an optimizer vector (ordered only over *varying* params).

        into a full parameter dict that includes fixed parameters.
        """
        if len(vec) != len(self._varying_specs):
            raise ValueError(f"Expected vector of length {len(self._varying_specs)}, got {len(vec)}")
        out = dict(self.fixed_params)  # start with fixed
        for s, v in zip(self._varying_specs, vec):
            out[s.name] = s.apply(float(v))
        return out

    def varying_vector_to_dict(self, vec: tp.Sequence[float]) -> tp.Dict[str, float]:
        """Convert an optimizer vector (ordered only over *varying* params).

        into a full parameter dict that includes fixed parameters.
        """
        if len(vec) != len(self._varying_specs):
            raise ValueError(f"Expected vector of length {len(self._varying_specs)}, got {len(vec)}")
        out = {}
        for s, v in zip(self._varying_specs, vec):
            out[s.name] = s.apply(float(v))
        return out

    def dict_to_vector(self, params: tp.Dict[str, float]) -> np.ndarray:
        return np.array([params[n] for n in self.varying_names], dtype=float)

    def defaults_vector(self) -> np.ndarray:
        vals = []
        for s in self._varying_specs:
            if s.default is not None:
                vals.append(float(s.default))
            else:
                lo, hi = s.bounds
                vals.append(0.5 * (lo + hi))
        return np.array(vals, dtype=float)

    def _set_single_bounds(self, param_name: str, bounds: tp.Tuple[float, float]):
        """
        :param param_name: name of the varying parameter.

        :param bounds: new bounds of the parameter
        :return: None
        """
        if param_name not in self.varying_names:
            raise KeyError(f"Parameter {param_name} not found in varying parameter space")

        low, high = bounds
        if low >= high:
            raise ValueError(f"Invalid bounds: low ({low}) must be less than high ({high})")

        for spec in self.specs:
            if spec.name == param_name:
                spec.set_bounds(bounds)
                break

    def set_default(self, params: dict[str, float]):
        """
        :param params: the dict of parameters.

        Set default value for parameters given in params
        :return:
        """
        for key, value in params.items():
            if key in self.fixed_params:
                self.fixed_params[key] = value
            elif key in self.varying_params:
                for spec in self._varying_specs:
                    if spec.name == key:
                        spec.default = value
                        self.varying_params[spec.name] = value
                        break
            else:
                raise KeyError(f"Key '{key}' not found in fixed_params or _varying_specs")

    def reduce_bounds(self, names: tp.Optional[str] = None, alpha: float = 0.2):
        """Reduces bounds of varying parameters.

        If bounds was (a, b) and default value c than the new bounds are:
        delta = (b-a)*alpha
        new_bounds = (c - delta, c+delta)
        :param names: names of parameters to reduce
        :param alpha: reducing coefficient
        :return: None
        """
        if names is None:
            names = self.varying_names
        for name in names:
            for spec in self._varying_specs:
                if spec.name == name:
                    default = spec.default
                    low, up = spec.bounds
                    delta = (up - low) * alpha

                    new_low = max(low, default-delta)
                    new_up = min(up, default+delta)
                    spec.bounds = (new_low, new_up)
                    break

    def set_bounds(self, bounds_dict: tp.Dict[str, tp.Tuple[float, float]]):
        """
        :param bounds_dict: the dict with names of parameters and their new bounds.

        :return: None
        """
        for param_name, bounds in bounds_dict.items():
            self._set_single_bounds(param_name, bounds)

    def suggest_optuna(self, trial) -> tp.Dict[str, float]:
        out = dict(self.fixed_params)  # start with fixed
        for s in self._varying_specs:
            lo, hi = s.bounds
            val = trial.suggest_float(s.name, lo, hi)
            out[s.name] = s.apply(val)
        return out

    def instrument_nevergrad(self) -> ng.p.Instrumentation:
        params = []
        for s in self._varying_specs:
            lo, hi = s.bounds
            params.append(ng.p.Scalar(lower=lo, upper=hi))
        return ng.p.Instrumentation(*params)

    def to_json_dict(self) -> dict:
        """Convert ParameterSpace to a JSON-serializable dictionary."""
        specs_data = []
        for s in self.specs:
            spec_dict = {
                "name": s.name,
                "bounds": list(s.bounds),
                "default": s.default,
                "vary": getattr(s, "vary", True),
            }
            if s.transform is not None:
                spec_dict["transform"] = s.transform.__class__.__name__
            specs_data.append(spec_dict)

        return {
            "specs": specs_data,
            "fixed_params": self.fixed_params,
        }

    def save(self, path: tp.Union[str, pathlib.Path]) -> None:
        """Save the parameter space to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_json_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ParameterSpace":
        """Load a parameter space from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_json_dict(data)


class CWSpectraSimulator:
    """Example of CW spectra simulator."""
    def __init__(self,
                 sample_updator: tp.Callable[[dict[str, float], tp.Any], tp.Any],
                 spectra_creator: tp.Callable[[tp.Any, torch.Tensor], torch.Tensor], *args):
        """
        :param sample_updator: Callable object that updates sample.

        :param spectra_creator: Callable object that creates spectra
        :param args:
        """
        self.sample_updator = sample_updator
        self.spectra_creator = spectra_creator
        self.args = args

    def __call__(self, fields: torch.Tensor, params: dict[str, float]):
        """
        :param fields: magnetic fields in Tesla units.

        :param params: parameters of param space
        :return:
        """
        sample = self.sample_updator(params, *self.args)
        return self.spectra_creator(sample, fields)


class BaseSpectrumFitter(ABC):
    """Base class for spectrum fitting."""
    __available_optimizer__ = {
        "nevergrad": sorted(ng.optimizers.registry.keys()),
        "optuna": [
            optuna.integration.BoTorchSampler,
            optuna.samplers.RandomSampler,
            optuna.samplers.TPESampler,
            optuna.samplers.BruteForceSampler,
            optuna.samplers.GridSampler,
            optuna.samplers.CmaEsSampler,
            optuna.samplers.NSGAIISampler,
            optuna.samplers.NSGAIIISampler,
        ],
    }

    def __init__(
        self,
        param_space: ParameterSpace,
        spectra_simulator: tp.Callable,
        norm_mode: str = "integral",
        objective=objectives.MSEObjective(),
        weights: tp.Optional[list[float]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param param_space: The object of ParameterSpace class where all varying parameters are included.
        :param spectra_simulator: Callable that takes x-data and parameters and returns simulated spectra
        :param norm_mode: Norm mode to fit data. 'integral' / 'max'
        :param objective: Used objective function. It should be an inheritor of objectives.BaseObjective
        :param weights: The weights for multi-spectra fit. Default is None
        :param device: Device for computation
        """
        self.norm_mode = norm_mode
        self._simulate_callable = spectra_simulator
        self.param_space = param_space
        self._objective = objective

        self.x_exp = None
        self.y_exp = None
        self.multisample = False
        self.weights = None
        self._loss_normalization = None

    @abstractmethod
    def _set_experimental(self, *args, **kwargs):
        """Process experimental data and set  them self.x_exp, self.y_exp,
        self.multisample."""
        pass

    @abstractmethod
    def _simulate_single_spectrum(self, params: dict[str, float], **kwargs) -> torch.Tensor:
        """Simulate spectrum from set of parameters.

        :param params: Full parameter dictionary
        :return: Normalized single simulated spectrum
        """
        pass

    @abstractmethod
    def _simulate_spectral_set(self, params: dict[str, float], **kwargs) -> list[torch.Tensor]:
        """Simulate set of spectra from set of parameters.

        :param params: Full parameter dictionary
        :return: List of normalized simulated spectra
        """
        pass

    def _get_loss_norm(self):
        if self.multisample:
            return [self._objective(torch.zeros_like(y), y).reciprocal() for y in self.y_exp]
        else:
            return self._objective(torch.zeros_like(self.y_exp), self.y_exp).reciprocal()

    def simulate_spectroscopic_data(self, params: tp.Dict[str, float], **kwargs) ->\
            tp.Union[list[torch.Tensor], torch.Tensor]:
        """
        :param params: fict of parameter names: parameter values.

        The names of parameters are names from param_space.
        Example:
        fitter.simulate_spectroscopic_data(dict(param_space))
        :param kwargs:
        :return: Simulated spectra - list or single spectra
        """
        if self.multisample:
            model = self._simulate_spectral_set(params, **kwargs)
        else:
            model = self._simulate_single_spectrum(params, **kwargs)
        return model

    def simulate_spectra_from_trial_params(self, trial_params: tp.Dict[str, float], **kwargs) ->\
            tp.Union[list[torch.Tensor], torch.Tensor]:
        """
        :param trial_params: Simulate spectra from parameters given as trial_params (only varied parameters).

        As fixed_parameters the parameters from self.param_space are used
        :param kwargs:
        :return: Simulated spectra - list or single spectra
        """
        return self.simulate_spectroscopic_data({**self.param_space.fixed_params, **trial_params}, **kwargs)

    def _loss_from_params(self, params: tp.Dict[str, tp.Union[float, torch.Tensor]], **kwargs) -> torch.Tensor:
        """Compute model - experiment residuals as a torch.Tensor."""
        with torch.no_grad():
            if self.multisample:
                models = self._simulate_spectral_set(params, **kwargs)
                loss = sum(self.weights[idx] * self._loss_normalization[idx] * self._objective(
                    models[idx], self.y_exp[idx]) for idx in range(len(models))) / len(models)
            else:
                model = self._simulate_single_spectrum(params, **kwargs)
                loss = self._loss_normalization * self._objective(model, self.y_exp)
            return loss

    def _tracker_to_trials(self, trials_tracker: TrialsTracker) -> list[NevergradTrial]:
        trials_all_results = trials_tracker.get_all_trials()
        ng_trials = [
            NevergradTrial(params=self.param_space.varying_vector_to_dict(trial["params"]),
                           _trial_id=trial["_trial_id"],
                           value=trial["value"]
                           ) for trial in trials_all_results
        ]
        return ng_trials

    def fit_optuna(
            self,
            show_progress: bool,
            seed: tp.Optional[int],
            return_best_spectrum: bool,

            n_trials: int = 300,
            timeout: tp.Optional[float] = None,
            n_jobs: int = 1,
            sampler: tp.Optional[optuna.samplers.BaseSampler] = None,
            study_name: tp.Optional[str] = None,
            run_dashboard: bool = True,
            **kwargs,
    ) -> FitResult:
        """Fit spectra using Optuna.

        Requires optuna to be installed.
        """

        def loss_function(trial):
            p = self.param_space.suggest_optuna(trial)
            loss = self._loss_from_params(p, **kwargs)
            return loss

        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if run_dashboard:
            storage = optuna.storages.InMemoryStorage()
            study = optuna.create_study(direction="minimize", sampler=sampler,
                                        study_name=study_name, load_if_exists=True, storage=storage)
            study.optimize(
                loss_function, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=show_progress)
            run_server(storage)
        else:
            study = optuna.create_study(direction="minimize", sampler=sampler,
                                        study_name=study_name, load_if_exists=True)
            study.optimize(
                loss_function, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=show_progress)

        best_params = {k: float(v) for k, v in study.best_params.items()}
        best_spec = None
        if return_best_spectrum:
            best_spec = self.simulate_spectroscopic_data({**self.param_space.fixed_params, **best_params}, **kwargs)
        return FitResult(best_params, float(study.best_value), best_spec, {"backend": "optuna", "study": study})

    def fit_nevergrad(
            self,
            show_progress: bool,
            seed: tp.Optional[int],
            return_best_spectrum: bool,

            budget: int = 200,
            optimizer_name: str = "TwoPointsDE",
            track_trials: bool = True,

            **kwargs,
    ) -> FitResult:
        """Fit spectra using Nevergrad (if installed)."""
        if ng is None:
            raise RuntimeError("Nevergrad is required for fit_nevergrad but not installed")

        instr = self.param_space.instrument_nevergrad()
        if seed is not None:
            ng.optimizers.registry.seed(seed)
        opt = ng.optimizers.registry[optimizer_name](parametrization=instr, budget=budget)

        def _loss_from_tuple(*args):
            params = self.param_space.vector_to_dict(args)
            return self._loss_from_params(params).item()

        if show_progress:
            progress_bar = ng.callbacks.ProgressBar()
            progress_bar.update_frequency = 25
            opt.register_callback("tell", progress_bar)

        trials_tracker = None
        if track_trials:
            trials_tracker = TrialsTracker()
            opt.register_callback("tell", trials_tracker)

        recommendation = opt.minimize(_loss_from_tuple)
        x = recommendation.value
        best_params = self.param_space.varying_vector_to_dict(x[0])
        best_spec = None
        if return_best_spectrum:
            best_spec = self.simulate_spectroscopic_data({**self.param_space.fixed_params, **best_params})

        trials = None
        if track_trials:
            trials = self._tracker_to_trials(trials_tracker)

        return FitResult(
            best_params, self._loss_from_params({**self.param_space.fixed_params, **best_params}), best_spec,
            {"backend": "nevergrad", "optimizer": optimizer_name, "trials": trials}
        )

    def fit(
            self,
            backend: str = "optuna",
            seed: tp.Optional[int] = None,
            show_progress: bool = True,
            return_best_spectrum: bool = True,
            **backend_kwargs,
    ) -> FitResult:
        """All fitting methods can be viewed in
        ''SpectrumFitter.__available_optimizer__.''

        :param backend: optuna / nevergrad. Sets which library should be used to fit data.
            Optuna supports not as many methods as nevergrad but they are quite powerful. Default fitting method is TPE.
            TPE has quite high exploration abilities and not as dramatic speed of work as Bayesian models.
            After the initial fitting process it is recommended to reduce
            the bounds and continue fitting with any method of convex optimization from Nevergrad:
            For example, with COBYLA.

        :param backend_kwargs: The kwargs of fit settings described in optuna / nevergrad library
            NOTE! Optuna and Nevergrad have different backend parameters.
            We have saved the initial naming from these libraries

            Key differences:
                                        optuna                                   nevergrad
            ----------------------------------------------------------------------------------------------------
            method type          optuna.samplers.BaseSampler                    str object
            ----------------------------------------------------------------------------------------------------
            number of iterations      n_trials                                    budget
            ----------------------------------------------------------------------------------------------------

        :return: None
        """
        method = backend.lower()
        if method == "optuna":
            return self.fit_optuna(seed=seed, show_progress=show_progress,
                                   return_best_spectrum=return_best_spectrum, **backend_kwargs
                                   )
        if method in ("nevergrad", "ng"):
            return self.fit_nevergrad(seed=seed, show_progress=show_progress,
                                      return_best_spectrum=return_best_spectrum, **backend_kwargs
                                      )
        raise ValueError(f"Unknown fit method: {method}")

    def save_state(self, path: tp.Union[str, pathlib.Path]) -> None:
        """Save the parameter space and experimental data.

        Subclasses should call `super().save_state(path)` and then save their
        specific experimental tensors.
        """
        state = {
            "param_space": self.param_space.to_json_dict(),
            "norm_mode": self.norm_mode,
            "objective": self._objective.__class__.__name__,
            "multisample": self.multisample,
            "weights": self.weights.tolist() if self.weights is not None else None,
        }
        with open(path + "_state.json", "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def _load_state_common(cls, path: tp.Union[str, pathlib.Path], device: torch.device, dtype: torch.dtype):
        """Load common state from JSON file and return components."""
        with open(path + "_state.json", "r") as f:
            state = json.load(f)

        param_space = ParameterSpace.from_json_dict(state["param_space"])
        objective_name = state["objective"]
        objective_cls = objectives.OBJECTIVE_REGISTRY.get(objective_name)
        if objective_cls is None:
            raise ValueError(f"Unknown objective class: {objective_name}")
        objective = objective_cls()

        weights = state.get("weights")
        if weights is not None:
            weights = torch.tensor(weights, dtype=dtype, device=device)

        return param_space, objective, state["norm_mode"], state["multisample"], weights


class SpectrumFitter(BaseSpectrumFitter):
    """General fitter for spectra.

    The user must provide either a `simulate_spectrum_callable` that maps a
    parameter dict -> torch.Tensor (spectrum on the same B-grid), or override
    the `simulate_spectrum` method in a subclass.

    Typical usage:
      - construct with B grid, experimental spectrum (np or torch), device
      - provide parameter specs
      - call fit(method='optuna'|'nevergrad')
    """
    __available_optimizer__ = {"nevergrad": sorted(ng.optimizers.registry.keys()),
                               "optuna": [optuna.integration.BoTorchSampler,
                                          optuna.samplers.RandomSampler,
                                          optuna.samplers.TPESampler,
                                          optuna.samplers.BruteForceSampler,
                                          optuna.samplers.GridSampler,
                                          optuna.samplers.CmaEsSampler,
                                          optuna.samplers.NSGAIISampler,
                                          optuna.samplers.NSGAIIISampler,
                                          ]
                               }

    def __init__(
            self,
            x_exp: tp.Union[np.ndarray, torch.Tensor, tp.List[tp.Union[np.ndarray, torch.Tensor]]],
            y_exp: tp.Union[np.ndarray, torch.Tensor, tp.List[tp.Union[np.ndarray, torch.Tensor]]],
            param_space: ParameterSpace,
            spectra_simulator: tp.Callable[
                [tp.Union[tp.List[torch.Tensor], torch.Tensor], tp.Dict[str, float], tp.Dict],
                tp.Union[torch.Tensor, tp.List[torch.Tensor]]
            ],
            norm_mode: str = "integral",
            objective=objectives.MSEObjective(),
            weights: tp.Optional[tp.List[float]] = None,
            device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        :param x_exp: Experimental x-axis data.

        It can be magnetic field (T), time (s)
            It is possible to pass a list for multi-object fit
        :param y_exp: Experimental y-axis data.
        :param param_space: The object of ParameterSpace class where all varying parameters are included
        :param spectra_simulator: Any callable object that takes x_data and parameters and returns simulated
            spectra or list of simulated spectra.
            It is highly recommended for all new parameters to use update methods:
            sample.update(new_params) or spec_creator.update_config(config)

            Example:
            class CWSpectraSimulator:
                def __init__(self,
                             sample_updator: tp.Callable[[dict[str, float], tp.Any], tp.Any],
                             spectra_creator: tp.Callable[[tp.Any, torch.Tensor], torch.Tensor], *args):
                    self.sample_updator = sample_updator
                    self.spectra_creator = spectra_creator
                    self.args = args

                def __call__(self, fields: torch.Tensor, params: dict[str, float]):
                    sample = self.sample_updator(params, *self.args)
                    return self.spectra_creator(sample, fields)

        :param norm_mode: Norm mode to fit data. 'integral' / 'max'
        :param device: Device for computation
        :param objective: Used objective function. It should be an inheritor of objectives.BaseObjective
        :param weights: The weights for multi-data fit. Default is None
        """
        super().__init__(param_space, spectra_simulator, norm_mode, objective, weights, device, dtype)
        self.x_exp, self.y_exp, self.multisample = self._set_experimental(x_exp, y_exp, device, dtype)

        if self.multisample and (weights is None):
            self.weights = torch.ones(len(self.x_exp), dtype=dtype, device=device)
        elif weights is not None:
            self.weights = torch.tensor(weights, dtype=dtype, device=device)
        else:
            self.weights = None
        self._loss_normalization = self._get_loss_norm()

    def _set_experimental(
            self, x_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
                  y_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
                  device: torch.device, dtype: torch.dtype) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, bool]:
        """Set expereimental given parameter.

        :param x_exp: Experimental x-axis data. It can be magnetic field
            (T), time (s)
        :param y_exp: Experimental y-axis data.
        :return: x_exp, y_exp in appropriate format and flag that is
            multisample fitting data
        """
        if isinstance(x_exp, list):
            if len(x_exp) != len(y_exp):
                raise ValueError("The number of x array and experimental arrays must be the same")
            else:
                x_exp = [torch.tensor(b, dtype=dtype, device=device) for b in x_exp]
                y_exp = [torch.tensor(y, dtype=dtype, device=device) for y in y_exp]
                for idx, b in enumerate(x_exp):
                    y_exp[idx] = normalize_spectrum(b, y_exp[idx], mode=self.norm_mode)
                multisample = True
        else:
            x_exp = torch.tensor(x_exp, dtype=dtype, device=device)
            y_exp = torch.tensor(y_exp, dtype=dtype, device=device)
            y_exp = normalize_spectrum(x_exp, y_exp, mode=self.norm_mode)
            multisample = False

        return x_exp, y_exp, multisample

    def _simulate_single_spectrum(self, params: tp.Dict[str, float], **kwargs) -> torch.Tensor:
        return normalize_spectrum(self.x_exp, self._simulate_callable(self.x_exp, params, **kwargs), mode=self.norm_mode)

    def _simulate_spectral_set(self, params: tp.Dict[str, float], **kwargs) -> list[torch.Tensor]:
        models = self._simulate_callable(self.x_exp, params, **kwargs)
        for idx in range(len(models)):
            models[idx] = normalize_spectrum(self.x_exp[idx], models[idx], mode=self.norm_mode)
        return models

    def save_state(self, path: str) -> None:
        """Save the fitter state, including experimental 1D data."""
        super().save_state(path)
        torch.save(
            {
                "x_exp": self.x_exp,
                "y_exp": self.y_exp,
            },
            path + "_exp.pt",
        )

    @classmethod
    def load_state(
        cls,
        path: tp.Union[str, pathlib.Path],
        spectra_simulator: tp.Callable,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> "SpectrumFitter":
        """
        Reconstruct a SpectrumFitter from saved state.

        :param path: Base path (without extension) used in `save_state`.
        :param spectra_simulator: The same (or equivalent) simulator callable.
        :param device: torch device (if None, use CPU).
        :param dtype: torch dtype.
        :return: A new SpectrumFitter instance.
        """
        param_space, objective, norm_mode, multisample, weights = cls._load_state_common(
            path, device, dtype
        )

        exp_data = torch.load(path + "_exp.pt", map_location=device)
        x_exp = exp_data["x_exp"]
        y_exp = exp_data["y_exp"]

        if isinstance(x_exp, list):
            x_exp = [t.to(device=device, dtype=dtype) for t in x_exp]
            y_exp = [t.to(device=device, dtype=dtype) for t in y_exp]
        else:
            x_exp = x_exp.to(device=device, dtype=dtype)
            y_exp = y_exp.to(device=device, dtype=dtype)

        return cls(
            x_exp=x_exp,
            y_exp=y_exp,
            param_space=param_space,
            spectra_simulator=spectra_simulator,
            norm_mode=norm_mode,
            objective=objective,
            weights=weights.tolist() if weights is not None else None,
            device=device,
            dtype=dtype,
        )


class Spectrum2DFitter(BaseSpectrumFitter):
    """Spectrum Fitter for 2D data.

    y_exp should be 2d array, x1_exp and x2_exp are axis
    """
    def __init__(
            self,
            x1_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
            x2_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
            y_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
            param_space: ParameterSpace,
            spectra_simulator: tp.Callable[
                [tp.Union[list[torch.Tensor], torch.Tensor],
                 tp.Union[list[torch.Tensor], torch.Tensor],
                 tp.Dict[str, float], tp.Dict],
                tp.Union[torch.Tensor, list[torch.Tensor]]
            ],
            norm_mode: str = "integral",
            objective=objectives.MSEObjective(),
            weights: list[float] = None,
            device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float32,
    ):
        """
        :param x1_exp: Experimental x1-axis data.

        It can be magnetic field (T), time (s),
            It is possible to pass a list for multi-object fit
        :param x2_exp: Experimental x2-axis data. It can be magnetic field (T), time (s),
            It is possible to pass a list for multi-object fit

        :param y_exp: Experimental y-axis data.
        :param param_space: The object of ParameterSpace class where all varying parameters are included
        :param spectra_simulator: Any callable object that takes x_data and parameters and returns simulated
            spectra or list of simulated spectra.
            It is highly recommended for all new parameters to use update methods:
            sample.update(new_params) or spec_creator.update_config(config)

            Example:
            class CWSpectraSimulator:
                def __init__(self,
                             sample_updator: tp.Callable[[dict[str, float], tp.Any], tp.Any],
                             spectra_creator: tp.Callable[[tp.Any, torch.Tensor], torch.Tensor], *args):
                    self.sample_updator = sample_updator
                    self.spectra_creator = spectra_creator
                    self.args = args

                def __call__(self, fields: torch.Tensor, params: dict[str, float]):
                    sample = self.sample_updator(params, *self.args)
                    return self.spectra_creator(sample, fields)

        :param norm_mode: Norm mode to fit data. 'integral' / 'max'
        :param device: Device for computation
        :param objective: Used objective function. It should be an inheritor of objectives.BaseObjective
        :param weights: The weights for multi-data fit. Default is None
        """
        super().__init__(param_space, spectra_simulator, norm_mode, objective, weights, device)
        self.x1_exp, self.x2_exp, self.y_exp, self.multisample = self._set_experimental(
            x1_exp, x2_exp, y_exp, device, dtype)

        if self.multisample and (weights is None):
            self.weights = torch.ones(len(self.x1_exp), dtype=dtype, device=device)
        elif weights is not None:
            self.weights = torch.tensor(weights, dtype=dtype, device=device)
        else:
            self.weights = None

        self._loss_normalization = self._get_loss_norm()

    def _set_experimental(
            self, x1_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
                  x2_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
                  y_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
                  device: torch.device, dtype: torch.dtype) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Set experimental given parameter.

        :param x1_exp: Experimental x1-axis data. It can be magnetic
            field (T), time (s)
        :param x2_exp: Experimental x1-axis data. It can be magnetic
            field (T), time (s)
        :param y_exp: Experimental y-axis data.
        :return: x1_exp, x2_exp, y_exp in appropriate format and flag
            that is multisample fitting data
        """
        if isinstance(x1_exp, list) and isinstance(x2_exp, list):
            if (len(x1_exp) != len(y_exp)) or (len(x2_exp) != len(y_exp)):
                raise ValueError("The number of x1 and x2 array and experimental arrays must be the same")
            else:
                x1_exp = [torch.tensor(b, dtype=dtype, device=device) for b in x1_exp]
                x2_exp = [torch.tensor(b, dtype=dtype, device=device) for b in x2_exp]
                y_exp = [torch.tensor(y, dtype=dtype, device=device) for y in y_exp]
                for idx, (b1, b2) in enumerate(zip(x1_exp, x2_exp)):
                    y_exp[idx] = normalize_spectrum2d(b1, b2, y_exp[idx], mode=self.norm_mode)
                multisample = True
        else:
            x1_exp = torch.tensor(x1_exp, dtype=dtype, device=device)
            x2_exp = torch.tensor(x2_exp, dtype=dtype, device=device)
            y_exp = torch.tensor(y_exp, dtype=dtype, device=device)
            y_exp = normalize_spectrum2d(x1_exp, x2_exp, y_exp, mode=self.norm_mode)
            multisample = False

        return x1_exp, x2_exp, y_exp, multisample

    def _simulate_single_spectrum(self, params: tp.Dict[str, float], **kwargs) -> torch.Tensor:
        return normalize_spectrum2d(self.x1_exp, self.x2_exp, self._simulate_callable(
            self.x1_exp, self.x2_exp, params, **kwargs), mode=self.norm_mode)

    def _simulate_spectral_set(self, params: tp.Dict[str, float], **kwargs) -> list[torch.Tensor]:
        models = self._simulate_callable(self.x1_exp, self.x2_exp, params, **kwargs)
        for idx in range(len(models)):
            models[idx] = normalize_spectrum2d(self.x1_exp[idx], self.x2_exp[idx], models[idx], mode=self.norm_mode)
        return models

    def save_state(self, path: tp.Union[str, pathlib.Path]) -> None:
        """Save the fitter state, including experimental 2D data."""
        super().save_state(path)
        torch.save(
            {
                "x1_exp": self.x1_exp,
                "x2_exp": self.x2_exp,
                "y_exp": self.y_exp,
            },
            path + "_exp.pt",
        )

    @classmethod
    def load_state(
        cls,
        path: tp.Union[str, pathlib.Path],
        spectra_simulator: tp.Callable,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> "Spectrum2DFitter":
        """
        Reconstruct a Spectrum2DFitter from saved state.

        :param path: Base path (without extension) used in `save_state`.
        :param spectra_simulator: The same (or equivalent) simulator callable.
        :param device: torch device (if None, use CPU).
        :param dtype: torch dtype.
        :return: A new Spectrum2DFitter instance.
        """
        param_space, objective, norm_mode, multisample, weights = cls._load_state_common(
            path, device, dtype
        )

        exp_data = torch.load(path + "_exp.pt", map_location=device)
        x1_exp = exp_data["x1_exp"]
        x2_exp = exp_data["x2_exp"]
        y_exp = exp_data["y_exp"]

        if isinstance(x1_exp, list):
            x1_exp = [t.to(device=device, dtype=dtype) for t in x1_exp]
            x2_exp = [t.to(device=device, dtype=dtype) for t in x2_exp]
            y_exp = [t.to(device=device, dtype=dtype) for t in y_exp]
        else:
            x1_exp = x1_exp.to(device=device, dtype=dtype)
            x2_exp = x2_exp.to(device=device, dtype=dtype)
            y_exp = y_exp.to(device=device, dtype=dtype)

        return cls(
            x1_exp=x1_exp,
            x2_exp=x2_exp,
            y_exp=y_exp,
            param_space=param_space,
            spectra_simulator=spectra_simulator,
            norm_mode=norm_mode,
            objective=objective,
            weights=weights.tolist() if weights is not None else None,
            device=device,
            dtype=dtype,
        )


class SpectrumCompositeFitter:
    """
    Composite fitter for simultaneously fitting multiple spectroscopic datasets
    with a shared parameter space.

    This class aggregates several :class:`BaseSpectrumFitter` instances, computes
    their individual losses, and combines them into a weighted sum. It provides a
    unified :meth:`fit` interface that delegates to the same backends (Optuna,
    Nevergrad) as the base fitter.

    All sub‑fitters must share the identical :class:`ParameterSpace` (the same
    varying parameter names, bounds, and fixed values). The combined loss is
    simply the weighted sum of the losses reported by each sub‑fitter.

    Attributes
    ----------
    param_space : ParameterSpace
        The common parameter space (extracted from the first fitter).
    fitters : List[BaseSpectrumFitter]
        The list of sub‑fitters.
    weights : List[float]
        The weights applied to each sub‑fitter's loss.
    """
    def __init__(
        self,
        fitters: tp.Sequence[BaseSpectrumFitter],
        weights: tp.Optional[tp.Sequence[float]] = None,
    ) -> None:
        """
        Initialise the composite fitter.

        :param fitters: Sequence of :class:`BaseSpectrumFitter` instances that
            will contribute to the combined loss.
        :param weights: Optional weights for each fitter. If ``None``, equal
            weights are used.
        :raises ValueError: If no fitters are provided, if their parameter
            spaces differ, or if the number of weights mismatches the number of
            fitters.
        """
        if not fitters:
            raise ValueError("At least one fitter must be provided.")

        first_ps = fitters[0].param_space
        for i, f in enumerate(fitters[1:], start=2):
            if f.param_space != first_ps:
                raise ValueError(
                    f"Fitter {i} has a different parameter space than the first fitter."
                )
        self.param_space = first_ps
        self.fitters = list(fitters)

        n = len(self.fitters)
        if weights is None:
            self.weights = [1.0 / n] * n
        else:
            if len(weights) != n:
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match number of fitters ({n})."
                )
            self.weights = list(weights)

    def _loss_from_params(self, params: tp.Dict[str, float], **kwargs) -> torch.Tensor:
        """
        Compute the combined loss from all sub‑fitters.

        The loss is a weighted sum of the individual losses returned by each
        fitter's :meth:`_loss_from_params` method.

        :param params: Full parameter dictionary (including fixed parameters).
        :param kwargs: Additional keyword arguments forwarded to each sub‑fitter.
        :return: Scalar loss tensor.
        """
        device = torch.device("cpu")
        dtype = torch.float32
        if hasattr(self.fitters[0], "x_exp"):
            x = self.fitters[0].x_exp
            if isinstance(x, list):
                device = x[0].device if x else device
                dtype = x[0].dtype if x else dtype
            elif isinstance(x, torch.Tensor):
                device = x.device
                dtype = x.dtype

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        for fitter, w in zip(self.fitters, self.weights):
            total_loss = total_loss + w * fitter._loss_from_params(params, **kwargs)
        return total_loss

    def _tracker_to_trials(self, trials_tracker: TrialsTracker) -> tp.List[NevergradTrial]:
        """
        Convert Nevergrad trial records to a list of :class:`NevergradTrial` objects.

        :param trials_tracker: The :class:`TrialsTracker` instance that collected
            trial data.
        :return: List of :class:`NevergradTrial` objects.
        """
        trials_all_results = trials_tracker.get_all_trials()
        ng_trials = [
            NevergradTrial(
                params=self.param_space.varying_vector_to_dict(trial["params"]),
                _trial_id=trial["_trial_id"],
                value=trial["value"],
            )
            for trial in trials_all_results
        ]
        return ng_trials

    def fit(
        self,
        backend: str = "optuna",
        seed: tp.Optional[int] = None,
        show_progress: bool = True,
        return_best_spectrum: bool = True,
        **backend_kwargs,
    ) -> FitResult:
        """
        Run the optimisation using the selected backend.

        The behaviour mirrors that of :meth:`BaseSpectrumFitter.fit`. The
        combined loss is minimised, and the best parameters are returned
        together with optional simulated spectra from each sub‑fitter.

        :param backend: Optimisation library to use. Allowed values are
            ``"optuna"`` or ``"nevergrad"`` (``"ng"`` is also accepted).
        :param seed: Random seed for reproducibility.
        :param show_progress: If ``True``, display a progress bar during
            optimisation.
        :param return_best_spectrum: If ``True``, the returned :class:`FitResult`
            will contain the simulated spectra for the best parameters. For a
            composite fitter this is a list of spectra, one per sub‑fitter.
        :param backend_kwargs: Additional arguments passed to the backend‑specific
            fit method. For Optuna these include ``n_trials``, ``timeout``,
            ``n_jobs``, ``sampler``, etc. For Nevergrad they include ``budget``,
            ``optimizer_name``, ``track_trials``.
        :return: A :class:`FitResult` object containing the best parameters, loss
            value, best spectra (if requested), and backend‑specific metadata.
        :raises ValueError: If an unknown backend is requested.
        :raises RuntimeError: If Nevergrad is selected but the library is not
            installed.
        """
        method = backend.lower()
        if method == "optuna":
            return self._fit_optuna(
                seed=seed,
                show_progress=show_progress,
                return_best_spectrum=return_best_spectrum,
                **backend_kwargs,
            )
        elif method in ("nevergrad", "ng"):
            return self._fit_nevergrad(
                seed=seed,
                show_progress=show_progress,
                return_best_spectrum=return_best_spectrum,
                **backend_kwargs,
            )
        else:
            raise ValueError(f"Unknown fit method: {method}")

    def _fit_optuna(
        self,
        show_progress: bool,
        seed: tp.Optional[int],
        return_best_spectrum: bool,
        n_trials: int = 300,
        timeout: tp.Optional[float] = None,
        n_jobs: int = 1,
        sampler: tp.Optional["optuna.samplers.BaseSampler"] = None,
        study_name: tp.Optional[str] = None,
        run_dashboard: bool = True,
        **kwargs,
    ) -> FitResult:
        """
        Internal method: fit using Optuna.

        :param show_progress: Show Optuna progress bar.
        :param seed: Random seed for sampler.
        :param return_best_spectrum: Whether to return best spectra.
        :param n_trials: Number of Optuna trials.
        :param timeout: Stop study after the given number of seconds.
        :param n_jobs: Number of parallel jobs.
        :param sampler: Optuna sampler instance.
        :param study_name: Name of the Optuna study.
        :param run_dashboard: If ``True``, launch an Optuna dashboard after
            optimisation.
        :param kwargs: Additional arguments passed to the loss function.
        :return: :class:`FitResult`
        """
        import optuna

        def loss_function(trial):
            p = self.param_space.suggest_optuna(trial)
            return self._loss_from_params(p, **kwargs).item()

        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if run_dashboard:
            storage = optuna.storages.InMemoryStorage()
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                study_name=study_name,
                load_if_exists=True,
                storage=storage,
            )
            study.optimize(
                loss_function,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=show_progress,
            )
            from optuna_dashboard import run_server
            run_server(storage)
        else:
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                study_name=study_name,
                load_if_exists=True,
            )
            study.optimize(
                loss_function,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=show_progress,
            )

        best_params = {k: float(v) for k, v in study.best_params.items()}
        best_spec = None
        if return_best_spectrum:
            full_params = {**self.param_space.fixed_params, **best_params}
            best_spec = [f.simulate_spectroscopic_data(full_params, **kwargs) for f in self.fitters]

        return FitResult(
            best_params,
            float(study.best_value),
            best_spec,
            {"backend": "optuna", "study": study},
        )

    def _fit_nevergrad(
        self,
        show_progress: bool,
        seed: tp.Optional[int],
        return_best_spectrum: bool,
        budget: int = 200,
        optimizer_name: str = "TwoPointsDE",
        track_trials: bool = True,
        **kwargs,
    ) -> FitResult:
        """
        Internal method: fit using Nevergrad.

        :param show_progress: Show Nevergrad progress bar.
        :param seed: Random seed.
        :param return_best_spectrum: Whether to return best spectra.
        :param budget: Number of function evaluations.
        :param optimizer_name: Name of the Nevergrad optimizer.
        :param track_trials: Whether to record trial history.
        :param kwargs: Additional arguments passed to the loss function.
        :return: :class:`FitResult`
        :raises RuntimeError: If Nevergrad is not installed.
        """
        try:
            import nevergrad as ng
        except ImportError:
            raise RuntimeError("Nevergrad is required for fit_nevergrad but not installed.")

        instr = self.param_space.instrument_nevergrad()
        if seed is not None:
            ng.optimizers.registry.seed(seed)
        opt = ng.optimizers.registry[optimizer_name](parametrization=instr, budget=budget)

        def _loss_from_tuple(*args):
            params = self.param_space.vector_to_dict(args)
            return self._loss_from_params(params, **kwargs).item()

        if show_progress:
            progress_bar = ng.callbacks.ProgressBar()
            progress_bar.update_frequency = 25
            opt.register_callback("tell", progress_bar)

        trials_tracker = None
        if track_trials:
            trials_tracker = TrialsTracker()
            opt.register_callback("tell", trials_tracker)

        recommendation = opt.minimize(_loss_from_tuple)
        x = recommendation.value
        best_params = self.param_space.varying_vector_to_dict(x[0])

        best_spec = None
        if return_best_spectrum:
            full_params = {**self.param_space.fixed_params, **best_params}
            best_spec = [f.simulate_spectroscopic_data(full_params, **kwargs) for f in self.fitters]

        trials = None
        if track_trials:
            trials = self._tracker_to_trials(trials_tracker)

        return FitResult(
            best_params,
            self._loss_from_params({**self.param_space.fixed_params, **best_params}, **kwargs).item(),
            best_spec,
            {"backend": "nevergrad", "optimizer": optimizer_name, "trials": trials},
        )

    def save_state(self, path: str) -> None:
        """
        Save the composite fitter state.

        Each sub‑fitter is saved in a subdirectory named `fitter_<i>`.
        """
        os.makedirs(path, exist_ok=True)

        meta = {
            "num_fitters": len(self.fitters),
            "weights": self.weights,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        for i, fitter in enumerate(self.fitters):
            sub_path = os.path.join(path, f"fitter_{i}")
            fitter.save_state(sub_path)

    @classmethod
    def load_state(
        cls,
        path: str,
        spectra_simulators: tp.List[tp.Callable],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> "SpectrumCompositeFitter":
        """
        Reconstruct a SpectrumCompositeFitter from saved state.

        :param path: Directory path used in `save_state`.
        :param spectra_simulators: List of simulator callables, one per sub‑fitter.
        :param device: torch device (if None, use CPU).
        :param dtype: torch dtype.
        :return: A new SpectrumCompositeFitter instance.
        """
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        fitters = []
        for i in range(meta["num_fitters"]):
            sub_path = os.path.join(path, f"fitter_{i}")
            exp_path = sub_path + "_exp.pt"
            exp_data = torch.load(exp_path, map_location="cpu")
            if "x1_exp" in exp_data:
                fitter_cls = Spectrum2DFitter
            else:
                fitter_cls = SpectrumFitter
            fitter = fitter_cls.load_state(
                sub_path,
                spectra_simulator=spectra_simulators[i],
                device=device,
                dtype=dtype,
            )
            fitters.append(fitter)

        return cls(fitters, weights=meta["weights"])


class SpaceSearcher:
    """For some cases not only the best fitting parameters are useful but all
    'good' parameters.

    Space searcher try to catch 'good' parameters that are far from the
    best fit parameters.
    """
    def __init__(
        self,
        loss_rel_tol: float = 1.0,
        top_k: int = 5,
        distance_fraction: float = 0.2,
    ):
        """
        :param loss_rel_tol: loss_trial / loss_best: cutoff parameter.

            that sets the acceptable loss of trial. Default is 1
        :param top_k: Returns only top_k lowest-loss trials.
        :param distance_fraction: Among all 'good' trials with low loss it
            accepts only trials with Euclidean distance in scaled (-1, 1) parameters > distance_fraction * max_distance

            To compute distance the parameters are scaled to (-1, 1)
        """
        self.loss_rel_tol = float(loss_rel_tol)
        self.top_k = int(top_k)
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

    def __call__(self, fit_result: FitResult, param_names: tp.Optional[tp.List[str]] = None) ->\
            tp.List[tp.Dict[str, tp.Any]]:
        """
        :param fit_result: The output of fitter.

        :param param_names: The names of parameters that should be included in search procedure.
        Default value is None means that all spec (varying) parameters should be included.
        :return: The results of fitting searching
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
                chosen_idx = within_thresh[: self.top_k]
            else:
                chosen_idx = sorted_idx[: self.top_k]
        else:
            chosen_idx = sorted_idx[: self.top_k]

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
