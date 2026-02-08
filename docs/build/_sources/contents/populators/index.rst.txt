.. _population_description:

Population Computation
======================

Overview
--------

The population computation module provides flexible tools for modeling EPR transition intensities based on energy level populations and density matrix dynamics. This module supports both stationary (continuous wave) and time-resolved EPR spectroscopy.

The module implements several approaches to compute the population-dependent contribution to EPR signal intensity:

* **Stationary spectra**: Population differences computed from thermal equilibrium or context-defined initial states
* **Kinetic approach**: Time evolution of level populations following rate equations
* **Rotating wave approximation**: Efficient density matrix evolution for systems meeting specific constraints
* **Propagator method**: Full quantum evolution without approximations on Hamiltonian parameters

All methods support powder averaging for disordered samples through spherical grid integration over molecular orientations (α, β angles).

Core Concepts
-------------

Population-Based Intensity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

EPR transition intensity depends on the population difference between resonant energy levels. For a transition between lower level j and upper level i:

.. math::

   I \propto (n_j - n_i) |M_{ij}|^2

where n are populations and M is the transition matrix element.

Initial Populations
~~~~~~~~~~~~~~~~~~~

Populations can be initialized in two ways:

1. **Thermal equilibrium**: Boltzmann distribution at temperature T:

   .. math::

      n_i \propto \exp(-E_i / k_B T)

2. **Context-defined**: Explicitly specified populations, automatically transformed to the field-dependent eigenbasis. (see :ref:`context-general-information` and :class:`mars.population.context.Context`)

Time-Dependent Evolution
~~~~~~~~~~~~~~~~~~~~~~~~~

For time-resolved experiments, populations or the full density matrix evolve according to relaxation dynamics encoded in a Context object and solved numerically by the appropriate populator class.

Module Components
-----------------

.. toctree::
   :maxdepth: 2

   stationary_spectra
   level_based_kynetic
   rotating_wave_approximation
   propagator_computation
   detailed_balance
