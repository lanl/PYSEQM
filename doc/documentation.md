# [PYSEQM](https://github.com/lanl/PYSEQM)
awdwadwad
## Semi-Empirical Methods Implemented:
1. MNDO
2. AM1
3. PM3

## Usage:

### Units:

* **Length**: Å  
* **Energy**: eV  
* **Temperature**: Kelvin  
* **Time**: femtosecond  
* **Force**: eV/Å  
* **Velocity**: Å/femtosecond  

### Runing script:
```python
# Perform Single Point  calculation with batch of systems, see Test1.py
# For furthur usage, see examples in tests/test*
import torch
from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.ElectronicStructure import Electronic_Structure

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Specify SEQM parameters and create molecule object:
species = torch.as_tensor([[8,6,1,1],
                           [8,6,1,1],
                           [8,8,6,0]], # zero-padding for batching
                          dtype=torch.int64, device=device)

coordinates = torch.tensor([
                              [
                               [0.00,    0.00,    0.00],
                               [1.22,    0.00,    0.00],
                               [1.82,    0.94,    0.00],
                               [1.82,   -0.94,    0.00]
                              ],
                              [
                               [0.00,    0.00,    0.00],
                               [1.22,    0.00,    0.00],
                               [1.82,    0.94,    0.00],
                               [1.82,   -0.94,    0.00]
                              ],
                              [
                               [0.00,    0.00,    0.00],
                               [1.23,    0.00,    0.00],
                               [1.82,    0.94,    0.00],
                               [0.0,0.0,0.0]            # zero-padding for batching
                              ]
                            ], device=device)

const = Constants().to(device)

elements = [0]+sorted(set(species.reshape(-1).tolist()))

seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [2,0.0], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [False, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : elements, #[0,1,6,8],
                   'learned' : [], # learned parameters name list, e.g ['U_ss']
                   #'parameter_file_dir' : '../seqm/params/', # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   'eig' : True
                   }

molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)

### Create electronic structure driver:
esdriver = Electronic_Structure(seqm_parameters).to(device)

### Run esdriver on molecules:
esdriver(molecules)

print(' Total Energy (eV):\n', molecules.Etot)
print('\n Electronic Energy (eV): ', molecules.Eelec)
print('\n Nuclear Energy (eV):\n', molecules.Enuc)
print('\n Heat of Formation (ev):\n', molecules.Hf)
print('\n HOMO-LUMO gap (eV):\n', molecules.e_gap)
print('\n Orbital energies (eV):\n', molecules.e_mo)
print('\n Charges:\n', molecules.q)
print('\n Dipole:\n', molecules.d)
print('\n Forces (eV/A):\n', molecules.force)
```

## Testing Examples: /examples/Test*.py
1. ./Test1_SinglePointProp.ipynb : single point calculation
2. ./Test2_GeomOpt.ipynb : geometry optimization
3. ./Test3_BOMD.ipynb : BOMD (NVE, NVT, Langevin)
4. ./Test4_XL-BOMD.ipynb : XL-BOMD
5. ./Test5_KSA-XL-BOMD.ipynb : KSA-XL-BOMD
6. ./Fullerenes : fullerenes test set
6. ./GramicidinS/KSA-XL-BOMD_GS_dp.ipynb : Langevin MD of doubly protonated Gramicidin S ( [GS + H<sub>2</sub>]<sup>2+</sup> )


## Output files
1. *filename.id*.xyz : single point calculation
2. ./Test2_GeomOpt.ipynb : geometry optimization



## Structure:

seqm : seqm module  
├── basics.py : collections of classes for perform basic operations  
├── MolecularDynamics.py : geometry optimization and NVE and Langevin Molecular Dyanmics  
├── XLBOMD.py : XL-BOMD  
├── params : MNDO/AM1/PM3 parameters  
└── seqm_functions  
    ├── cal_par.py : compute dipole/qutrupole charge separation and additive terms rho1 and rho2  
    ├── canon_dm_prt.py : canonical density matrix perturbation algorithm for first-order response (for KSA-XL-BOMD)  
    ├── constants.py : store some constant parameters  
    ├── data_loader.py : load and prepare dataset from numpy array, not updated  
    ├── diag.py : diagonalization functions, where pseudo_diag is not used  
    ├── diat_overlap.py : get overlap integrals  
    ├── energy.py : compute various energy terms  
    ├── fermi_q.py : Fermi operator expansion (for KSA-XL-BOMD)  
    ├── fock.py : construct Fockian matrix  
    ├── G_XL_LR.py : generates two-electron part of Fockian (for KSA-XL-BOMD)  
    ├── hcore.py : construct Hcore  
    ├── pack.py : functions to deal with the padding in batch of matrix  
    ├── parameters.py : load parameters from structured csv files as in ./params  
    ├── read_xyz.py  : read structures .xyz files  
    ├── save_xyz.py  : save structures to .xyz files  
    ├── scf_loop.py : perform SCF procedure  
    ├── SP2.py : single particle density matrix expansion algorithm SP2  
    ├── spherical_pot_force.py : apply spherical potential (plane bottom with parabolic walls)  
    ├── two_elec_two_center_int_local_frame.py : compute two electron two center integrals in local frame for each pair of atoms  
    └── two_elec_two_center_int.py : rotate and get two electron two center integrals in global frame  

basics.py  
  class Parser : prepare data in the form for other parts, similar to ./seqm/seqm_functions/data_loader  
  class Pack_Parameters : combine parameters provided (like from ML) with other required ones loaded from ./params/  
  class Hamiltonian : assemble functions in seqm/seqm_functions, perform SCF and construct Fockian  
  class Energy : get energies based on Hamiltonian (total energy, heat of formation, nuclear energies, etc)  
  class Force : use torch.autograd.grad to get gradient of total energy on coordinates to get force  

ElectronicStructure.py  
  class Electronic_Structure : calls SCF/XL-BOMD drivers and updates molecular properties  

Molecule.py  
  class Molecule : molecule objects for storing electronic/structural properties  

MolecularDynamics.py  
  class Geometry_Optimization_SD : geometry optimization with steepest descend  
  class KSA_XL_BOMD : NVE/NVT with Langevin thermostat  for KSA-XL-BOMD
  class Molecular_Dynamics_Basic : NVE MD  
  class Molecular_Dynamics_Langevin : NVT with Langevin thermostat  
  class XL_BOMD : NVE/NVT with Langevin thermostat  for XL-BOMD

XLBOMD.py  
  class EnergyXL : get energies based on XL-BOMD  
  class ForceXL : get force with XL-BOMD  

## Caution:

1. Atoms in molecules are sorted in descending order based on atomic number
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
4. For molecules with degeneracy, use convergers=0 or 2, and turn on ```seqm.seqm_functions.diag.CHECK_DEGENERACY = True```

## Citation:
[Zhou, Guoqing, et al. "Graphics processing unit-accelerated semiempirical Born Oppenheimer molecular dynamics using PyTorch." *Journal of Chemical Theory and Computation* 16.8 (2020): 4951-4962.](https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00243)