# [PYSEQM: PYtorch-based Semi-Empirical Quantum Mechanics]
[![Docs](https://img.shields.io/badge/docs-user%20guide-blue)](https://lanl.github.io/PYSEQM/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

[PYSEQM](https://lanl.github.io/PYSEQM/) is a semi-empirical quantum mechanics package implemented in [PyTorch](http://pytorch.org). It runs batched ground- and excited-state calculations on CPU or GPU, provides built-in interfaces for machine learning, and includes molecular dynamics engines (BOMD, XL-BOMD, excited-state MD) and nonadiabatic surface-hopping dynamics.

> **New here?** The [**user guide**](https://lanl.github.io/PYSEQM/) has copy-paste inputs for every workflow, with all `seqm_parameters` options explained. Start with the [Quickstart](https://lanl.github.io/PYSEQM/quick_start.html), or jump to:
> [Excited states](https://lanl.github.io/PYSEQM/excited_states.html) · [Molecular dynamics](https://lanl.github.io/PYSEQM/bomd.html) · [XL-BOMD](https://lanl.github.io/PYSEQM/Feature-XLBOMD.html) · [Surface hopping](https://lanl.github.io/PYSEQM/nonadiabatic_dynamics.html) · [Geometry optimization](https://lanl.github.io/PYSEQM/geometry_optimization.html)

<hr/>

## Features

**Methods**
* Semi-empirical Hamiltonians: MNDO, AM1, PM3, PM6 (including d orbitals, and the `PM6_SP` s/p-only variant), and the orthogonalization-corrected OM1, OM2 and OM3
* Restricted and unrestricted (open-shell) SCF
* D2 and D3 dispersion corrections (OM2/OM3 and others; see the user guide for availability per method)
* Batched calculations: many molecules, including different sizes, are processed together as tensors

**Ground state** ([docs](https://lanl.github.io/PYSEQM/single_point_scf.html))
* Single-point energies, analytic forces, dipoles, heats of formation
* Geometry optimization (through [geomeTRIC](https://github.com/leeping/geomeTRIC)) and normal-mode analysis
* Density matrix by diagonalization or by the [SP2](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.66.155115) expansion algorithm
* Born-Oppenheimer MD (NVE and Langevin) with checkpoint/restart
* Extended Lagrangian BOMD ([XL-BOMD](https://aip.scitation.org/doi/full/10.1063/1.3148075)), including the Krylov subspace approximation (KSA-XL-BOMD) for small HOMO-LUMO gaps

**Excited states** ([docs](https://lanl.github.io/PYSEQM/excited_states.html))
* Configuration Interaction Singles (CIS) and Time-Dependent Hartree-Fock (TDHF/RPA) with a batched Davidson solver
* Analytic excited-state forces, excited-state MD (including XL-ESMD)
* Fewest-switches surface hopping (FSSH) nonadiabatic dynamics over CIS states, with batched trajectories and restart
* CIS minimum-energy crossing point (MECI) optimization
* Absorption spectra and transition-density cube files (see `scripts/`)

**Machine learning and performance** ([docs](https://lanl.github.io/PYSEQM/Initialization.html))
* Differentiable end to end: learned Hamiltonian parameters can be trained through the SCF with implicit backward passes
* Interface with ML frameworks such as [HIPNN](https://aip.scitation.org/doi/abs/10.1063/1.5011181) (`examples/hipnn_training`, `examples/hippynn_namd`)
* GPU support through PyTorch, with optional `torch.compile` acceleration of selected kernels for long dynamics runs


## Installation

```bash
git clone https://github.com/lanl/PYSEQM.git
cd PYSEQM
pip install .
```
or
```bash
pip install git+https://github.com/lanl/PYSEQM.git
```

Optional extras:

```bash
pip install ".[geomopt]"   # geomeTRIC, for geometry optimization and MECI
```

## Quick start

The full walkthrough, with expected output, is in the [online Quickstart](https://lanl.github.io/PYSEQM/quick_start.html).

```python
import torch
from seqm.api import Constants, Electronic_Structure, Molecule

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

species = torch.as_tensor([[8, 1, 1]], dtype=torch.int64, device=device)
coordinates = torch.tensor(
    [[[0.00, 0.00, 0.00], [0.96, 0.00, 0.00], [-0.24, 0.93, 0.00]]], device=device
)

seqm_parameters = {
    "method": "AM1",  # MNDO, AM1, PM3, PM6, OM1, OM2, OM3
    "scf_eps": 1e-8,
    "scf_converger": [0, 0.1],
    # "excited_states": {"n_states": 5, "method": "cis"},  # uncomment for CIS
}

molecule = Molecule(Constants().to(device), seqm_parameters, coordinates, species).to(device)
Electronic_Structure(seqm_parameters).to(device)(molecule)

print(molecule.Etot, molecule.Hf, molecule.force)
```

## Documentation and examples

* [User guide](https://lanl.github.io/PYSEQM/) (start here): installation, SCF options, excited states, MD, XL-BOMD, geometry optimization, nonadiabatic dynamics. Sources are in `docs/source`.
* `examples/`: Jupyter notebooks (`Test1`-`Test9`) covering single points, geometry optimization, BOMD, XL-BOMD, KSA-XL-BOMD, normal modes, open-shell, PM6 and excited states, plus scripts for geometry optimization and `torch.compile` benchmarks.
* `scripts/`: absorption and IR spectra, transition-density cubes, and a batched NAMD workflow (see `scripts/NAMD_WORKFLOW.md`), plus Tully-model surface hopping tests.

## Tests

```bash
pip install ".[test]"
pytest tests
```

## Semi-empirical methods implemented

| Method | Notes |
|--------|-------|
| MNDO, AM1, PM3 | NDDO methods, s/p basis |
| PM6 | Includes d orbitals, used only by atoms that have them (molecules without such atoms run on the faster s/p kernels); `PM6_SP` is the s/p-only variant |
| OM1, OM2, OM3 | Orthogonalization-corrected methods; D2/D3 dispersion supported for OM2/OM3 |

A trained model file (`model.pt`) is included in the repository root.

<hr/>

## Authors:
[Vishikh Athavale](mailto:vishikh@lanl.gov), [Maksim Kulichenko](mailto:maxim@lanl.gov), [Guoqing Zhou](mailto:guoqingz@usc.edu), [Benjamin Nebgen](mailto:bnebgen@lanl.gov), , [Nikita Fedik](https://www.nikitafedik.xyz/), Nicholas Lubbers, Walter Malone, Anders M. N. Niklasson and Sergei Tretiak

## Citation:
1. [Athavale, Vishikh, et al. "PYSEQM 2.0: Accelerated Semiempirical Excited-State Calculations on Graphical Processing Units." *Journal of Chemical Theory and Computation* 21, 9498–9510 (2025).
](https://doi.org/10.1021/acs.jctc.5c00980)
2. [Zhou, Guoqing, et al. "Graphics processing unit-accelerated semiempirical Born Oppenheimer molecular dynamics using PyTorch." *Journal of Chemical Theory and Computation* 16.8 (2020): 4951-4962.](https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00243)
3. [Kulichenko, Maksim, et al. "Semi-Empirical Shadow Molecular Dynamics: A PyTorch implementation." *Journal of Chemical Theory and Computation* 19.11 (2023): 3209–3222.](https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c00234)
4. [Zhou, Guoqing, et al. "Deep learning of dynamically responsive chemical Hamiltonians with semiempirical quantum mechanics." *Proceedings of the National Academy of Sciences* 119.27 (2022): e2120333119.](https://www.pnas.org/doi/10.1073/pnas.2120333119)

## Acknowledgments:
Los Alamos National Lab (LANL), Center for Nonlinear Studies (CNLS), T-1

## Copyright Notice:

© (or copyright) 2020. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

## License:

This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
