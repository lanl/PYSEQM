# PYSEQM

## Semi-Empirical Methods Implemented:
1. MNDO
2. AM1
3. PM3

## Usage:

### Units:

**Length**: Å
**Energy**: eV
**Temperature**: Kelvin
**Time**: femtosecond


### Runing script:
```python
# Perform Molecular Dynamics with batch of systems, see test8.py
# For furthur usage, see examples in tests/test*
import torch
from seqm.seqm_functions.constants import Constants
from seqm.XLBOMD import XL_BOMD

#torch setting
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Specify SEQM parameters
seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM
                   'scf_eps' : 1.0e-6,  # unit eV, SCF converging threshold
                   'scf_converger' : [2,0.0], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [True, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : [0,1,6,8], # element atomic number, ascending order
                   'learned' : [], # learned parameters name list, e.g ['U_ss'], 
                        # name lists in seqm.basics.parameterlist  
                        # if empty, loaded from ./seqm/params, 
                        # otherwise provide dictionary for modules, see test5.py
                        # or callable for modules, f(species, coordinates), which return a dictionary 
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   }

# prepare configurations
# Note configurations are prepared in batch mode, i.e. first dimension: molecules
#example with two molecule systems
species = torch.as_tensor([[8,6,1,1],[8,6,1,1]],dtype=torch.int64, device=device)
coordinates = torch.tensor([
                  [
                   [0.014497970281389074, 3.208059797520201e-05, -1.0697192468126102e-07],
                   [1.3364260161590171, -3.26283382508022e-05, 8.510168803526663e-07],
                   [1.7576599286132542, 1.0395080227523756, -5.348699492766755e-07],
                   [1.757558154681721, -1.039614513603968, 2.8473584469483316e-06]
                  ],
                  [
                   [0.014497970281389074, 3.208059797520201e-05, -1.0697192468126102e-07],
                   [1.3364260161590171, -3.26283382508022e-05, 8.510168803526663e-07],
                   [1.7576599286132542, 1.0395080227523756, -5.348699492766755e-07],
                   [1.757558154681721, -1.039614513603968, 2.8473584469483316e-06]
                  ]
                 ], device=device)

# Prepare some constant parameters
const = Constants().to(device)


# Setup Molecular Dynamics Enginee
#md =  Molecular_Dynamics_Basic(seqm_parameters, timestep=1.0).to(device)
#md =  Molecular_Dynamics_Langevin(seqm_parameters, timestep=1.0, damp=100.0, T=300.0, output={'molid':[0, 1], 'thermo':1, 'dump':10, 'prefix':'md'}).to(device)
md = XL_BOMD(seqm_parameters, timestep=1.0, k=9).to(device)

# Initialize velocities, (can also be provided)
velocities = md.initialize_velocity(const, coordinates, species, Temp=300.0)

# NVE or NVT
#coordinates, velocities, accelaration =  md.run(const, 20, coordinates, velocities, species)

# XL-BOMD
coordinates, velocities, accelaration, P, Pt =  md.run(const, 100, coordinates, velocities, species)
```

## Testing Example: /tests/test*.py
1. ./test1.py : example to get Fock matrix, energy etc, and backward
2. ./test2.py : example to get force
3. ./test3.py : example for geometry optimization
4. ./test4.py : example for molecular dynamcies (NVE, NVT)
5. ./test5.py : example for how to add trained parameters for the calculation
6. ./test6/test6.py : example to verify the force computed from the code
7. ./test7/test7.py : example to verify the gradient on parameters
8. ./test8.py : XL-BOMD
9. ./test9.py : NVE, compared with test8.py
10. ./test10  : Nanostar 884 atoms, XL-BOMD
11. ./test11.py : test scf_backward recursive formula
12. ./test12.py : second order gradient
13. ./test13 : check gradient of orbital energy on parameters
14. ./test14 : check gradient of orbital energy on coordinates
15. ./test15 : check the second order gradients, test with gradient of force on coordinates

## Structure:

seqm : seqm module  
├── basics.py : collections of classes for perform basic operations  
├── MolecularDynamics.py : geometry optimization and NVE and Langevin Molecular Dyanmics  
├── XLBOMD.py : XL-BOMD  
├── params : MNDO/AM1/PM3 parameters  
└── seqm_functions  
    ├── cal_par.py : compute dipole/qutrupole charge separation and additive terms rho1 and rho2  
    ├── constants.py : store some constant parameters  
    ├── data_loader.py : load and prepare dataset from numpy array, not updated  
    ├── diag.py : diagonalization functions, where pseudo_diag is not used  
    ├── diat_overlap.py : get overlap integrals  
    ├── energy.py : compute various energy terms  
    ├── fock.py : construct Fockian matrix  
    ├── hcore.py : construct Hcore  
    ├── pack.py : functions to deal with the padding in batch of matrix  
    ├── parameters.py : load parameters from structured csv files as in ./params  
    ├── scf_loop.py : perform SCF procedure  
    ├── SP2.py : single particle density matrix expansion algorithm SP2  
    ├── two_elec_two_center_int_local_frame.py : compute two electron two center integrals in local frame for each pair of atoms  
    └── two_elec_two_center_int.py : rotate and get two electron two center integrals in global frame  

basics.py  
  class Parser : prepare data in the form for other parts, similar to ./seqm/seqm_functions/data_loader  
  class Pack_Parameters : combine parameters provided (like from ML) with other required ones loaded from ./params/  
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