# PYSEQM: Semi-Empirical Quantum Mechanics with PYtorch

PYSEQM is a module for  Born Oppenheimer Molecular Dynamics (BOMD) and Extended Lagrangian  BOMD based on Semi-Empirical Quantum Mechanics. It has efficient molecular dynamic engines with GPU supported and it is implemented in Python 3 with machine learning framework [PyTorch](http://pytorch.org). Several geometric optimization and thermostat algoritms are included for facilitating the simulations.

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

## Usage
see ```./doc/documentation.pdf```

### Semi-Empirical Methods Implemented:
1. MNDO
2. AM1
3. PM3

### Prerequisites
* PyTorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment. (tested with 1.2 and 1.5)




## Structure:

# ./test/test*.py : testing example
1. ./test1.py : example to get Fock matrix, energy etc, and backward
2. ./test2.py : example to get force
3. ./test3.py : example for geometry optimization
4. ./test4.py : example for molecular dynamcies (NVE, NVT)
5. ./test5.py : example for how to add trained parameters for the calculation
6. ./test6/test6.py : example to verify the force computed from the code
==> autograd.grad and backward are working now
7. ./test7/test7.py : example to verify the gradient on parameters
8. ./test8.py : XL-BOMD
9. ./test9.py : NVE, compared with test8.py
10. ./test10  : Nanostar 884 atoms, XL-BOMD
11. ./test11.py : test scf_backward recursive formula
12. ./test12.py : second order gradient
13. ./test13 : check gradient of orbital energy on parameters
14. ./test14 : check gradient of orbital energy on coordinates
15. ./test15 : check the second order gradients, test with gradient of force on coordinates

./params/MOPAC : MNDO/AM1/PM3 parameters from MOPAC7.1

./seqm : seqm module  
├── basics.py                                  : collections of classes for perform basic operations  
├── MolecularDynamics.py                       : geometry optimization and NVE and Langevin Molecular Dyanmics  
├── XLBOMD.py                                  : XL-BOMD  
└── seqm_functions                             :  
    ├── cal_par.py                             : compute dipole/qutrupole charge separation and additive terms rho1 and rho2  
    ├── constants.py                           : store some constant parameters  
    ├── data_loader.py                         : load and prepare dataset from numpy array, not updated  
    ├── diag.py                                : diagonalization functions, where pseudo_diag is not used  
    ├── diat_overlap.py                        : get overlap integrals  
    ├── energy.py                              : compute various energies  
    ├── fock.py                                : construct Fockian  
    ├── hcore.py                               : construct Hcore  
    ├── pack.py                                : functions to deal with the padding in batch of matrix  
    ├── parameters.py                          : load parameters from structured csv files as in ./params/MOPAC  
    ├── scf_loop.py                            : perform SCF procedure  
    ├── SP2.py                                 : single particle density matrix expansion algorithm SP2  
    ├── two_elec_two_center_int_local_frame.py : compute two electron two center integrals in local frame for each pair of atoms  
    └── two_elec_two_center_int.py             : rotate and get two electron two center integrals in global frame  

basics.py  
  class Parser : prepare data in the form for other parts, similar to ./seqm/seqm_functions/data_loader  
  class Pack_Parameters : combine parameters provided (like from ML) with other required ones loaded using ./seqm/seqm_functions/parameters.py from ./params/MOPAC  
  class Hamiltonian : assemble functions in seqm/seqm_functions, perform SCF and construct Fockian  
  class Energy : get energies based on Hamiltonian (total energy, heat of formation, nuclear energies, etc)  
  class Force : use torch.autograd.grad to get gradient of total energy on coordinates to get force  

MolecularDynamics.py  
  class Geometry_Optimization_SD : geometry optimization with steepest descend  
  class Molecular_Dynamics_Basic : NVE MD  
  class Molecular_Dynamics_Langevin : NVT with Langevin thermostat  
  class Geometry_Optimization_SD_LS : geometry optimization using linear search based on steepest descend, not finished  
  class Molecular_Dynamics_Nose_Hoover : NVT with Nose Hoover, not finished  

XLBOMD.py  
  class EnergyXL : get energies based on XL-BOMD  
  class ForceXL : get force with XL-BOMD  
  class XL_BOMD : module to perform XL-BOMD  

./huckel : extended Huckel theory, still need parameters for computing overlap and construct Fockian

## Caution:

1. atoms in molecules are sorted in descending order based on atomic number
2. 0 padding is used for atoms in molecules
```
   NH3 ==> [7,1,1,1], H2O ==> [8,1,1,0]
```
3. indexing for atoms works with or without padding atom
```
   [[7,1,1,1],[[7,1,1,0],[7,1,1,1]]  ==> index [0,1,2,3,4,5,6,8,9,10,11]
   [[7,1,1,1],[[7,1,1,0],[7,1,1,1]]  ==> index [0,1,2,3,4,5,6,7,8,9,10]
```

## Suggestions:
1. Benchmark convergence criteria eps for SCF and SP2, eps_sp2 will affect number of iterations for SCF
2. in general set converger=[2], combining adaptive mixing and pulay seems takes fewest iterations
3. when on GPU, use SP2, on CPU don't use SP2, but benchmark to check


## Citation:
Zhou, Guoqing, et al. "Graphics processing unit-accelerated semiempirical Born Oppenheimer molecular dynamics using PyTorch." *Journal of Chemical Theory and Computation* 16.8 (2020): 4951-4962.

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