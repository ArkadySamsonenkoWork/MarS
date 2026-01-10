# MarS

**A toolkit for researchers to simulate, analyze, and explore EPR systems efficiently.**

---

## 🚀 Overview

**MarS** is a Python library for constructing spin systems (electrons and nuclei), defining their magnetic interactions, and simulating Electron Paramagnetic Resonance (EPR) spectra.

It supports a wide range of interaction models, efficient batched computations on CPU and GPU, flexible numerical precision (`float32` / `float64`), and tools for both stationary and time-resolved EPR experiments.

MarS is designed for research-oriented workflows, including rapid prototyping, large-scale parameter scans, and automated parameter fitting.

---

## 🔑 Core Capabilities

### Interaction Support
MarS allows users to construct spin systems with the most widely used magnetic interactions:
- Zeeman interaction  
- Exchange interaction  
- Dipolar interaction  
- Zero-field splitting (ZFS)  
- Hyperfine interaction  

Both **isotropic** and **anisotropic** parameters are supported.

---

### Orientation Support
- Arbitrary orientation of interaction tensors using **Euler angles**

---

### Broadening Support
MarS provides several mechanisms to model experimental linewidths:
- Gaussian and Lorentzian line broadening  
- Hamiltonian broadening  
- Broadening due to distributions of Hamiltonian parameters  (so-called strains)

---

### EPR Spectroscopy Simulation
- Simulation of **continuous-wave (CW) EPR spectra**
- Support for **powder** and **single-crystal** samples
- Field-domain and frequency-domain simulations

---

### Radiation Polarization Support
- Simulation of spectra under **polarized microwave radiation**
- Polarization-dependent transition probabilities

---

### Numerical Precision Control
- Support for `float64` and `float32` precision

---

### CPU / CUDA Support
- Support execution on **CPU** and **CUDA-enabled GPUs**
---

### Optimization Framework
- Parameter fitting using **Optuna** and **Nevergrad** libraries
---

### Post-Fitting Analysis
- Tools for analyzing alternative solutions
- Exploration of parameter correlations and degeneracies

---

# ⏱️ Time-Resolved Capabilities

MarS is a comprehensive framework for modeling **time-resolved EPR experiments** with two complementary computational paradigms.

### Relaxation Paradigms
- **Population relaxation (Kinetic approach)**: Evolution of diagonal density matrix elements (population vectors)
- **Density matrix relaxation**: Full quantum evolution of all density matrix elements. It includes two methods of computations:
  - Rotating frame approximation method
  - Direct propagator calculation method

---

### Flexible Relaxation Factrors Definition
MarS provides powerful tools for defining complex relaxation processes:
- **Population losses** (e.g., phosphorescence from triplet states)
- **Spontaneous transitions** (free transitions satisfying detailed balance)
- **Induced transitions** (driven transitions)
- **Decoherence** (for density matrix formalism)

All mechanisms can be specified in any of several predefined bases or custom transformation matrices.

---

### Basis Transformation Framework
Comprehensive support for relaxation parameter specification in multiple bases:
- **Eigenbasis** (`eigen`): Hamiltonian eigenstates in magnetic field
- **Zero-field splitting basis** (`zfs`): Eigenstates of the ZFS operator
- **Multiplet basis** (`multiplet`): Total spin and projection states |S, M⟩
- **Product basis** (`product`): Individual spin projections
- **Custom bases**: User-defined transformation matrices

Automatic transformation of kinetic matrices and relaxation superoperators between bases.

---

### Relaxation Algebra
- **Summation**: Combine multiple relaxation mechanisms defined in different bases
- **Tensor product**: Construct composite quantum systems with independent subsystem dynamics
- Enables modeling of complex multi-component systems and multi-pathway relaxation

---

### Liouville Space Formalism
- Full support for Liouvillian relaxation superoperators
- Implementation via **Lindblad equation** for general Markovian evolution
- Automatic enforcement of detailed balance for spontaneous transitions
---

### Numerical Solvers
Multiple solution strategies optimized for different scenarios:

**Population kinetics:**
- Stationary solution via matrix exponentiation (for time-independent systems)
- Quasi-stationary iterative solution (for time-dependent rates)
- Adaptive ODE integration (via `torchdiffeq`, for general time dependence)

**Density matrix evolution:**
- Rotating frame approximation (computationally efficient, limited to isotropic g-factors)
- Propagator computation approach (fully general, supports arbitrary anisotropy and relaxation)

---

## ▶️ Getting Started

### Installation

```bash
git clone https://github.com/ArkadySamsonenkoWork/MarS.git
cd mars
pip install -e <folder>

```
### Code Example
```bash


import torch
import matplotlib.pyplot as plt
from mars import spin_system, spectra_manager

# Select device and precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

# Define a simple electron spin system
g_tensor = spin_system.Interaction((2.02, 2.04, 2.06), dtype=dtype, device=device)

system = spin_system.SpinSystem(
    electrons=[0.5],
    g_tensors=[g_tensor],
    dtype=dtype,
    device=device
)

# Create a powder sample
sample = spin_system.MultiOrientedSample(
    spin_system=system,
    gauss=0.001,
    lorentz=0.001,
    dtype=dtype,
    device=device
)

# Create spectrum calculator
spectra = spectra_manager.StationarySpectra(
    freq=9.8e9,
    sample=sample,
    dtype=dtype,
    device=device
)

# Magnetic field range
fields = torch.linspace(0.3, 0.4, 1000, device=device, dtype=dtype)

# Compute spectrum
intensity = spectra(sample, fields)

# Plot result
plt.plot(fields.cpu(), intensity.cpu())
plt.xlabel("Magnetic field (T)")
plt.ylabel("Intensity (a.u.)")
plt.title("Simulated CW EPR Spectrum")
plt.show()

```