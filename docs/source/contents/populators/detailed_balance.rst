.. _detailed_balance:

Detailed Balance Enforcement
============================

Overview
--------

Spontaneous (free) relaxation transitions must satisfy the principle of detailed balance at thermal equilibrium. This ensures that at temperature :math:`T`, the system reaches a stationary Boltzmann distribution where the rates of forward and backward transitions between any pair of energy levels satisfy:

.. math::

   \frac{w_{j \to i}}{w_{i \to j}} = \exp\left(-\frac{E_i - E_j}{k_B T}\right)

In MarS, the kinetic matrix element :math:`w_{ij}` is defined as the physical transition rate **from state :math:`j` to state :math:`i`**, i.e.,

.. math::

   w_{ij} \equiv w_{j \to i}

Thus, the detailed balance condition can be equivalently written as:

.. math::

   \frac{w_{ij}}{w_{ji}} = \exp\left(-\frac{E_i - E_j}{k_B T}\right)

MarS automatically enforces detailed balance for all free transition probabilities by applying Boltzmann corrections.
Driven (induced) transitions are not subject to this constraint and are added to the kinetic matrix or relaxation superoperator without modification.

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
   The input matrix is interpreted directly as physical rates :math:`w_{ij} = w_{j \to i}`.  
   Missing backward rates (where :math:`w_{ji} = 0` but :math:`w_{ij} > 0`) are inferred via detailed balance:

   .. math::

      w_{ij} = w_{ji} \cdot \exp\left(-\frac{E_i - E_j}{k_B T}\right)

Boltzmann Correction
~~~~~~~~~~~~~~~~~~~~
Since the symmetry mode is more convenient for discussion, we will use it in the following.
Under this mode for each pair of energy levels i and j with energy difference :math:`\Delta E_{ij} = E_j - E_i`, the corrected transition rates are:

.. math::

   w_{ij}^* = \frac{2w'_{ij}}{1 + \exp(-\Delta E_{ij} / k_B T)}

.. math::

   w_{ji}^* = \frac{2w'_{ij}}{1 + \exp(\Delta E_{ij} / k_B T)} = w_{ij}^* \exp(-\Delta E_{ij} / k_B T)

where :math:`w'_{ij}` is the symmetric input rate equal to :math:`w'_{j->i}`

**Verification**: The corrected probabilities satisfy:

.. math::

   \frac{w_{ij}^*}{w_{ji}^*} = \exp\left(-\frac{E_i - E_j}{k_B T}\right) = \exp\left(\frac{\Delta E_{ij}}{k_B T}\right)

and their sum is conserved:

.. math::

   w_{ij}^* + w_{ji}^* = 2w'_{ij}

Kinetic Matrix Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full kinetic matrix K is constructed as:

.. math::

   K = W^* + D - \text{diag}(O)

where:

- **W** is the matrix of Boltzmann-corrected free transitions
- **D** is the matrix of driven transitions (no Boltzmann correction)
- **O** is the vector of outgoing loss rates (e.g., phosphorescence)

The diagonal elements of W\* and D are set to enforce probability conservation:

.. math::

   W_{ii} = -\sum_{j \neq i} W_{ji}, \quad D_{ii} = -\sum_{j \neq i} D_{ji}

Thus the total kinetic matrix has:

.. math::

   K_{ii} = -\sum_{j \neq i} (W_{ji} + D_{ji}) - O_i

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

This element represents the rate of transition from population :math:`\rho_{jj}` **to population** :math:`\rho_{ii}`. In other words, :math:`R_{iijj}` is the rate :math:`j \to i`.

Detailed Balance Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

MarS applies Boltzmann correction only to the population‑population coupling block of the free relaxation superoperator.

**Algorithm**:

1. **Identify population indices** – For an :math:`N`-level system, the population elements of the vectorized density matrix are at positions  
   :math:`\{0, N+1, 2(N+1), \dots, (N-1)(N+1)\}`, corresponding to :math:`\rho_{00}, \rho_{11}, \dots, \rho_{N-1,N-1}`.

2. **Extract the population submatrix** – Define the :math:`N \times N` matrix :math:`\mathbf{P}` by  

   .. math::

      P_{ij} = R_{iijj} \qquad (\text{rate from } j \text{ to } i),\quad i,j = 0,\dots,N-1.

3. **Store the original column sums** – For each level :math:`j`, compute the total outflow (including any irreversible losses) as the negative of the column sum:

   .. math::

      \text{colsum}_j = \sum_{i} P_{ij}.

   In a closed system :math:`\text{colsum}_j = 0`; with losses :math:`\text{colsum}_j = -\text{loss}_j`.  
   The values :math:`\text{colsum}_j` must be preserved after correction.

4. **Symmetrize and apply Boltzmann factor** – For every pair :math:`i \neq j` with energy difference :math:`\Delta E_{ij} = E_j - E_i`,  
   compute the symmetric average :math:`s_{ij} = (P_{ij} + P_{ji})/2`. Then define the new rates that satisfy detailed balance:

   .. math::

      P'_{ij} = \frac{2 s_{ij}}{1 + \exp(-\Delta E_{ij}/k_B T)}, \qquad
      P'_{ji} = \frac{2 s_{ij}}{1 + \exp(\Delta E_{ij}/k_B T)}.

   These expressions preserve the sum :math:`P'_{ij} + P'_{ji} = 2 s_{ij}` and guarantee  

   .. math::

      \frac{P'_{ij}}{P'_{ji}} = \exp\!\left(-\frac{E_i - E_j}{k_B T}\right).

5. **Restore the original column sums** – For each column :math:`j`, compute the sum of the new off‑diagonals:

   .. math::

      S'_j = \sum_{i \neq j} P'_{ij}.

   Then set the diagonal element so that the column sum remains unchanged:

   .. math::

      P'_{jj} = \text{colsum}_j - S'_j.

   This is equivalent to :math:`P'_{jj} = P_{jj} + \bigl(\sum_{i \neq j} P_{ij} - \sum_{i \neq j} P'_{ij}\bigr)`, ensuring that the total decay rate (including losses) from level :math:`j` is preserved.

6. **Replace the population block** – Insert the corrected :math:`\mathbf{P}'` back into the full superoperator at the population indices. All coherence‑related elements (dephasing, coherence‑population coupling) remain untouched.

**Result**: The corrected superoperator satisfies detailed balance for population transfers:

.. math::

   \frac{R_{iijj}^*}{R_{jjii}^*} = \exp\!\left(-\frac{E_i - E_j}{k_B T}\right),

while the total outflow from each level (the column sum) is identical to that of the input superoperator. This guarantees that observable decay rates (e.g., spontaneous emission or phosphorescence) are not artificially altered by the thermal correction.

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
  It symmetrizes the input free transition probabilities, applies Boltzmann weighting to satisfy thermal equilibrium, and constructs a column‑conservative kinetic matrix.

- :class:`mars.population.tr_utils.EvolutionSuper`:  
  Performs analogous corrections in the *density matrix (Liouville space)* formalism, following the algorithm described above.