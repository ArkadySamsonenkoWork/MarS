.. _detailed_balance:

Detailed Balance Enforcement
============================

Overview
--------

Spontaneous (free) relaxation transitions must satisfy the principle of detailed balance at thermal equilibrium. This ensures that at temperature T, the system reaches a stationary Boltzmann distribution where the rates of forward and backward transitions between any pair of energy levels satisfy:

.. math::

   \frac{w_{ij}}{w_{ji}} = \exp\left(\frac{E_j - E_i}{k_B T}\right)

MarS automatically enforces detailed balance for all free transition probabilities by applying Boltzmann corrections. Driven (induced) transitions are not subject to this constraint and are added to the kinetic matrix or relaxation superoperator without modification.

This document describes how MarS modifies free transitions to satisfy detailed balance in both the kinetic (population-based) and density matrix paradigms.


Detailed Balance in the Kinetic Approach
----------------------------------------

Input Convention
~~~~~~~~~~~~~~~~

The input matrix of free transition probabilities, denoted ``free_probs``, may be *arbitrary*. However, MarS provides two modes of processing this input, controlled by the flag ``symmetry_probs``:

1. **Symmetric mode** (``symmetry_probs=True``, default):  
   The class **symmetrizes** the input internally by computing:

   .. math::

      w'_{ij} = \frac{1}{2}(w_{ij} + w_{ji})

   This symmetric average is then used as the base rate for Boltzmann correction.  
   This mode is appropriate when the user provides raw or unstructured rates and wishes MarS to enforce physical symmetry before thermal scaling.

2. **Asymmetric mode** (``symmetry_probs=False``):  
   No symmetrization is performed. The input is interpreted as containing the **backward** rates :math:`w_{ji}` directly. Forward rates are derived via detailed balance:

   .. math::

      w_{ij} = w_{ji} \cdot \exp\left(\frac{E_j - E_i}{k_B T}\right)

   This mode is useful when the user already has a physically meaningful asymmetric rate structure (e.g., from microscopic modeling).


Boltzmann Correction
~~~~~~~~~~~~~~~~~~~~

For each pair of energy levels i and j with energy difference :math:`\Delta E_{ij} = E_i - E_j`, the corrected transition rates are:

.. math::

   w_{ij}^* = \frac{2w'_{ij}}{1 + \exp(-\Delta E_{ij} / k_B T)}

.. math::

   w_{ji}^* = \frac{2w'_{ij}}{1 + \exp(\Delta E_{ij} / k_B T)} = w_{ij}^* \exp(\Delta E_{ij} / k_B T)

where :math:`w'_{ij}` is the symmetric input rate.

**Verification**: The corrected probabilities satisfy:

.. math::

   \frac{w_{ij}^*}{w_{ji}^*} = \exp\left(\frac{E_j - E_i}{k_B T}\right) = \exp\left(\frac{-\Delta E_{ij}}{k_B T}\right)

and their sum is conserved:

.. math::

   w_{ij}^* + w_{ji}^* = 2w'_{ij}


Kinetic Matrix Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full kinetic matrix K is constructed as:

.. math::

   K = W^* + D - \text{diag}(O)

where:

- **W\*** is the matrix of Boltzmann-corrected free transitions
- **D** is the matrix of driven transitions (no Boltzmann correction)
- **O** is the vector of outgoing loss rates (e.g., phosphorescence)

The diagonal elements of W\* and D are set to enforce probability conservation:

.. math::

   W^*_{ii} = -\sum_{j \neq i} W^*_{ji}, \quad D_{ii} = -\sum_{j \neq i} D_{ji}

Thus the total kinetic matrix has:

.. math::

   K_{ii} = -\sum_{j \neq i} (W^*_{ji} + D_{ji}) - O_i

This ensures that in the absence of losses (O = 0), the column sums are zero and total population is conserved.


Detailed Balance in the Density Matrix Approach
-----------------------------------------------

Liouville Space Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the density matrix formalism, the evolution equation in Liouville space is:

.. math::

   \frac{d\hat{\rho}}{dt} = (-i\hat{H} + \hat{R}_{\text{free}} + \hat{R}_{\text{driv}})\hat{\rho}

where:

- :math:`\hat{H}` is the Hamiltonian superoperator: :math:`\hat{H} = H \otimes I - I \otimes H`
- :math:`\hat{R}_{\text{free}}` is the spontaneous relaxation superoperator (subject to detailed balance)
- :math:`\hat{R}_{\text{driv}}` is the driven relaxation superoperator (no detailed balance)

The density matrix :math:`\rho` (:math:`N \times N`) is vectorized into :math:`\hat{\rho}` (:math:`N^2 \times 1`), and superoperators are :math:`N^2 \times N^2` matrices.


Population Transfer Block
~~~~~~~~~~~~~~~~~~~~~~~~~

The relaxation superoperator couples elements of the density matrix. For population transfers, only the diagonal block matters:

.. math::

   \hat{R}[|i\rangle\langle i|, |j\rangle\langle j|] \equiv R_{iijj}

This represents the rate of transition from population :math:`\rho_{jj}` to population :math:`\rho_{ii}`.


Detailed Balance Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

MarS applies Boltzmann correction only to the population-population coupling block of the free relaxation superoperator.

**Algorithm**:

1. Extract the population block indices: for N-level system, the population elements of the vectorized density matrix are at positions {0, N+1, 2(N+1), ..., (N-1)(N+1)} corresponding to {:math:`\rho_{00}`, :math:`\rho_{11}`, ..., :math:`\rho_{N-1,N-1}`}.

2. Extract the :math:`N \times N` population transfer sub-matrix:

   .. math::

      P_{ij} = R_{iijj} \quad \text{for } i, j = 0, 1, ..., N-1

3. Store the total decay rate from each level (must be preserved):

   .. math::

      \text{decay}_i = \sum_j R_{iijj}

4. Symmetrize and apply Boltzmann factor:

   .. math::

      P'_{ij} = (P_{ij} + P_{ji}) \cdot \frac{1}{1 + \exp(-\Delta E_{ij}/k_B T)}

   for all :math:`i \neq j`.

5. Restore diagonal elements to preserve total decay:

   .. math::

      P'_{ii} = -\sum_{j \neq i} P'_{ji} + \text{decay}_i

6. Replace the population block in the full superoperator with P'.

**Result**: The corrected superoperator satisfies:

.. math::

   \frac{R_{iijj}^*}{R_{jjii}^*} = \exp\left(\frac{E_j - E_i}{k_B T}\right)

for population-population coupling elements, while all coherence-related elements (dephasing, coherence-population coupling if present) remain unchanged.


Total Decay Correction
~~~~~~~~~~~~~~~~~~~~~~

The diagonal element :math:`R_{iiii}` represents the total rate of depopulation from level i, including:

- Transfers to other levels: :math:`-\sum_{j \neq i} R_{jjii}`
- Pure losses (e.g., phosphorescence): part of the original :math:`R_{iiii}`

When Boltzmann correction is applied, the sum :math:`\sum_{j \neq i} R_{jjii}` changes. To maintain the correct total loss rate (which is a physical observable), we adjust :math:`R_{iiii}` so that:

.. math::

   \text{decay}_i = R_{iiii} + \sum_{j \neq i} R_{iijj} = \text{constant}

This ensures that spontaneous emission rates (if included in the model) are not artificially altered by the detailed balance correction.


Implementation Notes
--------------------

In MarS, relaxation parameters are organized into distinct categories:

**For kinetic (population-based) computations**:

- **free_probs**: Spontaneous transition rates between energy levels (subject to detailed balance)
- **driven_probs**: Induced transition rates from external perturbations (no detailed balance)
- **out_probs**: Irreversible loss rates from individual levels (no detailed balance)

**For density matrix computations**:

- **free_superop** (:math:`\hat{R}_{\text{free}}`): Combines free_probs, out_probs, and dephasing rates into a single spontaneous relaxation superoperator (subject to detailed balance correction on the population block)
- **driven_superop** (:math:`\hat{R}_{\text{driv}}`): Induced relaxation processes (no detailed balance)

The enforcement of detailed balance for spontaneous (free) relaxation processes is implemented in two core utility classes within the "MarS" library:

- :class:`mars.population.tr_utils.EvolutionMatrix`:  
  Handles detailed balance correction in the *kinetic (population-based)* framework.
  It symmetrizes the input free transition probabilities, applies Boltzmann weighting to satisfy thermal equilibrium, and constructs a column-conservative kinetic matrix.

- :class:`mars.population.tr_utils.EvolutionSuper`:  
  Performs analogous corrections in the *density matrix (Liouville space)* formalism.