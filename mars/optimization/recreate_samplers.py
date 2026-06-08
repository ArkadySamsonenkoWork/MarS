import inspect
import optuna


def recreate_sampler_without_history(sampler: optuna.samplers.BaseSampler):
    """Recreate an Optuna sampler with same parameters but zero history."""

    sampler_class = type(sampler)
    init_params = inspect.signature(sampler_class.__init__).parameters
    kwargs = {}
    for param_name in init_params:
        if param_name == "self":
            continue
        for attr_name in [param_name, f"_{param_name}", f"__{param_name}"]:
            if hasattr(sampler, attr_name):
                kwargs[param_name] = getattr(sampler, attr_name)
                break
    return sampler_class(**kwargs)

