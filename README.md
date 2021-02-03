# PYSEQM: Semi-Empirical Quantum Mechanics with PYtorch

PYSEQM is a module for  Born Oppenheimer Molecular Dynamics (BOMD) and Extended Lagrangian  BOMD based on Semi-Empirical Quantum Mechanics. It has efficient molecular dynamic engines with GPU supported and it is implemented in Python 3 with machine learning framework [PyTorch](http://pytorch.org). Several geometric optimization and thermostat algoritms are included for facilitating simulations.

<hr/>

## Authors:

[Guoqing Zhou](guoqingz@usc.edu), Benjamin Nebgen, Nicholas Lubbers, Walter Malone, Anders M. N. Niklasson and Sergei Tretiak

<hr/>

## Installation:

```bash
git clone https://github.com/lanl/PYSEQM.git
cd PYSEQM
python setup install
```
or
```bash
pip install git+https://github.com/lanl/PYSEQM.git
```

## Usage:
see [```./doc/documentation.md```](./doc/documentation.md)

### Semi-Empirical Methods Implemented:
1. MNDO
2. AM1
3. PM3

### Prerequisites
* PyTorch>=1.2

## Copyright Notice

© (or copyright) 2020. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

## License

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