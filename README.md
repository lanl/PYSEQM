# [PYSEQM: PYtorch-based Semi-Empirical Quantum Mechanics](https://github.com/lanl/PYSEQM)
[PYSEQM](https://github.com/lanl/PYSEQM) is a Semi-Empirical Quantum Mechanics package implemented in [PyTorch](http://pytorch.org). It provides built-in interfaces for machine learning and efficient molecular dynamic engines with GPU supported. Several molecular dynamics algorithms are implemented for facilitating dynamic simulations, inlcuding orginal and Extended Lagrangian Born-Oppenheimer Molecular Dynamics, geometric optimization and  several thermostats. 

<hr/>

## Features:

* Interface with machine learning (ML) framework like [HIPNN](https://aip.scitation.org/doi/abs/10.1063/1.5011181) for ML applications and development.
* GPU-supported Molecular Dynamics Engine
* Stable and Efficient Extended Lagrangian Born Oppenheimer Molecular Dynamics ([XL-BOMD](https://aip.scitation.org/doi/full/10.1063/1.3148075))
* Includes
* Efficient expansion algorithm [SP2](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.66.155115) for generating density matrix


## Installation:

```bash
git clone https://github.com/kulichenko-LANL/PYSEQM_dev.git
cd PYSEQM
python setup.py install
```
or
```bash
pip install git+https://github.com/kulichenko-LANL/PYSEQM_dev.git
```

To enable GPU with CUDA, please refer to the Installation Guide on [PyTorch website](https://pytorch.org/)

## Prerequisites:
* PyTorch>=1.9

## Usage:
see [```./doc/documentation.md```](./doc/documentation.md)

## trained model file
model.pt

## Semi-Empirical Methods Implemented:
1. MNDO
2. AM1
3. PM3

<hr/>

## Authors:

TBD

## Citation:
[Zhou, Guoqing, et al. "Graphics processing unit-accelerated semiempirical Born Oppenheimer molecular dynamics using PyTorch." *Journal of Chemical Theory and Computation* 16.8 (2020): 4951-4962.](https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00243)

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
