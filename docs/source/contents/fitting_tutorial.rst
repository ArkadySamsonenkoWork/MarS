.. _fitting_tutorial:

Fitting Spectroscopic Data
==========================

MarS provides a flexible framework for fitting experimental spectroscopic data using modern optimization libraries such as Optuna and Nevergrad.

Defining the Parameter Space
----------------------------

All fitting procedures start by defining which parameters are allowed to vary and within what bounds.
This is done using the :class:`mars.optimization.fitter.ParameterSpace` and :class:`mars.optimization.fitter.ParamSpec`.

.. code-block:: python

    from mars.optimization import ParameterSpace, ParamSpec

    param_specs = [
        ParamSpec(name="g", bounds=(2.0, 2.3), default=2.05),
        ParamSpec(name="D",  bounds=(100e6, 1000e6), default=200e6),
        ParamSpec(name="J",  bounds=(5e9, 40e9),   default=10e6),
        ParamSpec(name="ham_strain", bounds=(2e7, 1e8), default=5e7),
        ParamSpec(name="lorentz", bounds=(0.0, 1e-3), default=1e-4)
    ]

    param_space = ParameterSpace(
        specs=param_specs,
        fixed_params={"temperature": 10.0, "freq": 34e9}
    )

Creating a Spectra Simulator
----------------------------

The optimizer needs a callable that maps a parameter dictionary to a simulated spectrum:

.. code-block:: python

    class CWSpectraSimulator:
        def __init__(self, init_params):
            sample = create_sample(init_params)  # user-defined function
            self.spectra_creator = StationarySpectra(
                freq=init_params["freq"],
                sample=sample,
                temperature=init_params["temperature"]
            )

        def __call__(self, fields, params):
            sample = create_sample(params)  # user-defined function
            return self.spectra_creator(sample, fields)

For cases where only context-dependent quantities (e.g., populations or kinetic rates) change while spin Hamiltonian parameters remain fixed, you can cache resonance data to speed up evaluation:

.. code-block:: python

    class CWSpectraSimulator:
        def __init__(self, init_params):
            self.sample = create_sample(init_params)  # user-defined function
            self.spectra_creator = StationarySpectra(
                recompute_spin_parameters=False,  # resonance positions/vectors won't be recomputed
                freq=init_params["freq"],
                sample=self.sample,
                temperature=init_params["temperature"]
            )

        def __call__(self, fields, params):
            context = create_context(params)  # user-defined function
            self.spectra_creator.update_context(context)  # update context to new polarization parameters
            return self.spectra_creator(self.sample, fields)

Objective Functions
-------------------

MarS supports multiple objective functions for comparing simulated and experimental spectra.
All objectives inherit from :class:`mars.optimization.objectives.BaseObjectiveFunction` and return a scalar loss.

.. code-block:: python

    from mars.optimization.objectives import MSEObjective, CosineSimilarity

    fitter = SpectrumFitter(
        x_exp=fields_exp,
        y_exp=spectrum_exp,
        param_space=param_space,
        spectra_simulator=CWSpectraSimulator(),
        norm_mode="max",
        objective=MSEObjective()  # or CosineSimilarity(), etc.
    )

Fitting with Optuna Samplers
----------------------------

MarS supports two optimization libraries for parameter fitting: *Optuna* and *Nevergrad*. While both are powerful, they differ in design philosophy and ease of use:

- **Optuna** provides a smaller but well curated set of samplers that are easy to configure and highly effective for most spectroscopic fitting tasks.

- **Nevergrad** offers a much broader collection of optimization algorithms - from gradient-free methods to evolutionary and population-based strategies,
making it highly flexible but also more challenging to navigate without prior experience.

Optuna samplers (e.g., ``TPESampler``, ``CmaEsSampler``, ``NSGAIISampler``) offers several samplers suitable for different stages of optimization.

**Tree-structured Parzen Estimator (TPE)**

.. code-block:: python

    import optuna
    from mars.optimization import SpectrumFitter

    fitter = SpectrumFitter(
        x_exp=fields_exp,
        y_exp=spectrum_exp,
        param_space=param_space,
        spectra_simulator=CWSpectraSimulator(),
        norm_mode="max"
    )

    sampler_tpe = optuna.samplers.TPESampler(
        seed=42,
        multivariate=True,
        n_startup_trials=50
    )

    result_tpe = fitter.fit(
        backend="optuna",
        n_trials=300,
        sampler=sampler_tpe,
        study_name="tpe_fit"
    )

**Covariance Matrix Adaptation Evolution Strategy (CMA-ES)**

.. code-block:: python

    sampler_cmaes = optuna.samplers.CmaEsSampler(
        seed=42,
        n_startup_trials=30,
        restart_strategy="ipop"
    )

    result_cmaes = fitter.fit(
        backend="optuna",
        n_trials=300,
        sampler=sampler_cmaes,
        study_name="cmaes_fit"
    )

**Bayesian Optimization with BoTorch**

.. code-block:: python

    sampler_botorch = optuna.integration.BoTorchSampler(
        seed=42,
        n_startup_trials=20
    )

    result_botorch = fitter.fit(
        backend="optuna",
        n_trials=100,
        sampler=sampler_botorch,
        study_name="botorch_fit"
    )

In general, MarS can accept any additional arguments and pass them to ``optuna.create_study()``. For more possibilities, see the `Optuna documentation <https://optuna.readthedocs.io/en/stable/>`_.

Optuna Dashboard
----------------

MarS supports launching the Optuna dashboard during fitting for real-time monitoring. Set ``run_dashboard=True`` in the ``fit()`` call:

.. code-block:: python

    result = fitter.fit(backend="optuna", n_trials=300, run_dashboard=True)

This starts a local web server (typically at http://localhost:8080) where you can inspect trial history, parameter importance, and convergence behavior. The dashboard requires the ``optuna-dashboard`` package to be installed.

Fitting with Nevergrad
----------------------

`Nevergrad <https://github.com/facebookresearch/nevergrad>` is a gradient-free optimization library. It provides a unified interface to more than 100 derivative-free optimization algorithms, including evolutionary strategies (e.g., CMA-ES), Bayesian optimization variants, particle swarm methods, and classical direct search techniques like COBYLA and Powell.

Example: using the COBYLA optimizer (a constrained optimization algorithm based on linear approximations):

.. code-block:: python

    result_cobyla = fitter.fit(
        backend="ng",
        budget=300,          # maximum number of function evaluations
        optimizer_name="Cobyla"
    )

The full list of optimizers available through Nevergrad in MarS can be inspected at runtime:

.. code-block:: python

    print(SpectrumFitter.__available_optimizers__["nevergrad"])

This includes popular choices such as:
- ``"CMA"`` (Covariance Matrix Adaptation Evolution Strategy) — robust for continuous, non-convex problems,
- ``"DE"`` (Differential Evolution) — effective for multimodal landscapes,
- ``"PSO"`` (Particle Swarm Optimization),
- ``"OnePlusOne"`` — simple yet surprisingly effective for low-dimensional problems,
- and many more.

Weighted Multi-Spectrum Fitting
-------------------------------

When fitting multiple spectra simultaneously, you can assign different weights to each dataset using the ``weights`` argument. This is useful when some spectra are noisier or more critical than others:

.. code-block:: python

    fitter = SpectrumFitter(
        x_exp=[fields1, fields2],
        y_exp=[spectrum1, spectrum2],
        param_space=param_space,
        spectra_simulator=MultiSpectraSimulator(),
        weights=[1.0, 0.5],  # second spectrum contributes half as much to total loss
        norm_mode="max"
    )

2D Spectral Fitting with Spectrum2DFitter
-----------------------------------------

For time-resolved or other 2D spectroscopic data, it is possible to use :class:`mars.optimization.fitter.Spectrum2DFitter`. It accepts two independent axes (e.g., magnetic field and time) and a 2D intensity array.

The simulator must accept three arguments: ``x1``, ``x2``, and ``params``.

Example setup:

.. code-block:: python

    class TRSpectraSimulator:
        def __call__(self, fields, times, params):
            # fields: 1D tensor (magnetic field axis)
            # times: 1D tensor (time axis)
            # returns: 2D tensor of shape (len(times), len(fields))
            sample = create_sample(params)
            context = create_context(params)
            tr_creator = TimeResolvedSpectra(
                context=context,
                freq=params["freq"],
                sample=sample,
                temperature=params["temperature"]
            )
            return tr_creator(sample, fields, times)

    fitter_2d = Spectrum2DFitter(
        x1_exp=fields,      # magnetic field (T)
        x2_exp=times,       # time (s)
        y_exp=spectrum_2d,  # 2D experimental data
        param_space=param_space,
        spectra_simulator=TRSpectraSimulator(),
        norm_mode="integral"
    )

    result_2d = fitter_2d.fit(backend="optuna", n_trials=200)

Like its 1D counterpart, :class:`Spectrum2DFitter` supports multi-dataset fitting (list inputs), custom objectives, and weighted losses.

Exploring Alternative Minima with SpaceSearcher
-----------------------------------------------

In complex landscapes, multiple parameter sets may yield similarly good fits. The :class:`mars.optimization.fitter.SpaceSearcher` class identifies such alternatives that are distant from the best solution in parameter space.

.. code-block:: python

    from mars.optimization import SpaceSearcher

    searcher = SpaceSearcher(
        loss_rel_tol=0.2,      # accept trials with loss ≤ 1.2 × best_loss
        top_k=5,               # return up to 5 alternatives
        distance_fraction=0.2  # require minimum scaled Euclidean distance
    )

    alternatives = searcher(fit_result=result_tpe, param_names=["g", "D", "J"])
    print_trial_results(alternatives)

:class:`SpaceSearcher` works with results from both Optuna and Nevergrad backends and scales parameters before computing distances to ensure fair comparison across units.

For convenience, use :func:`mars.optimization.fitter.print_trial_results` to display results in a readable format:

.. code-block:: python

    from mars.optimization import print_trial_results
    print_trial_results(alternatives, max_params=6, precision=5)