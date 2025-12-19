import torch
from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.ElectronicStructure import Electronic_Structure
from seqm.seqm_functions.read_xyz import read_xyz

import warnings
warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

### create molecule object:
species = torch.as_tensor([
                            # [8,1,1],
                            [6,1,1,1,1],
                           # [1,1,],
                           # [1,1,0,0],
                           # [8,6,],
                          # [8,6,1,1],
                          # [8,8,6,0]
                          ], # zero-padding for batching
                          dtype=torch.int64, device=device)
   
coordinates = torch.tensor([
                           #  [
                           # [  0.000000000 ,    0.000000000  ,   0.000000],
                           # [  0.943153505 ,    0.000000000  ,   0.000000],
                           # [ -0.272586078 ,    0.903190425  ,   0.000000],
                           # ]
                           [
                             [ -2.7878725,    1.3475389,    0.0000000,], 
                             [ -2.2590725,    1.5085389,    0.9359000,],
                             [ -2.5827725,    2.1715389,   -0.6786000,],
                             [ -2.4533725,    0.4161389,   -0.4496000,],
                             [ -3.8563725,    1.2938389,    0.1921000,],]
                              # [
                              #  [0.82,    0.00,    0.00],
                              #  [1.82,    0.94,    0.00],
                              #  [1.81,   -0.94,    0.10],
                              # # [0.0,0.0,0.0],            # zero-padding for batching
                              # # [0.0,0.0,0.0],            # zero-padding for batching
                              # ],
                             # [
                             #  [0.00,    0.00,    0.00],
                             #  [1.23,    0.00,    0.00],
                             #  [1.82,    0.94,    0.00],
                             #  [1.81,    -0.94,    0.00],
                             #  # [0.0,0.0,0.0],            # zero-padding for batching
                             # ]
                            ], device=device)
species, coordinates = read_xyz(['/Users/vishikh/onedrive/calculations/CNT_10.xyz'])
# species, coordinates = read_xyz(['./examples/methane.xyz'])
# species, coordinates = read_xyz(['./methane.xyz'])
species = torch.as_tensor(species,dtype=torch.int64,device=device)
coordinates = torch.as_tensor(coordinates, device=device)

const = Constants().to(device)
active_state = 1

seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'scf_eps' : 1.0e-8,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [2], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   # 'uhf' : True,
                   'excited_states': {'n_states':9, 'method':'cis'},
                   'active_state': active_state,
                   # 'scf_backward': 1,
                   'analytical_gradient':[True]
                   }

molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)

### Create electronic structure driver:
esdriver = Electronic_Structure(seqm_parameters).to(device)

esdriver(molecules)
anal = molecules.force.detach().cpu()
import seqm
# for deb in (50.0,5.0,0.5,5e-2,5e-4,5e-5,5e-6,5e-7,5e-8,5e-9,5e-16):
for deb in (5.0,5e-16):
    seqm.seqm_functions.diag.DEGEN_THRESHOLD = deb
    print(f"degen thresh is {deb}")
    # species, coordinates = read_xyz(['h2o.xyz','./XYZ.0.xyz'])
    species, coordinates = read_xyz(['/Users/vishikh/onedrive/calculations/CNT_10.xyz'])
    # species, coordinates = read_xyz(['./examples/methane.xyz'])
    # species, coordinates = read_xyz(['./methane.xyz'])
    species = torch.as_tensor(species,dtype=torch.int64,device=device)
    coordinates = torch.as_tensor(coordinates, device=device)

    const = Constants().to(device)

    seqm_parameters = {
                       'method' : 'AM1',  # AM1, MNDO, PM#
                       'scf_eps' : 1.0e-8,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                       'scf_converger' : [2], # converger used for scf loop
                                             # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                             # [1], adaptive mixing
                                             # [2], adaptive mixing, then pulay
                       # 'uhf' : True,
                       'excited_states': {'n_states':9, 'method':'cis'},
                       'active_state': active_state,
                       'scf_backward': 1,
                       # 'analytical_gradient':[True]
                       }

    molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)

    ### Create electronic structure driver:
    esdriver = Electronic_Structure(seqm_parameters).to(device)

    esdriver(molecules)
    print(f'Force norm is\n{torch.linalg.norm(anal-molecules.force)}')
### Run esdriver on molecules:
# for i in range(1):
#     # esdriver(molecules,analytical_gradient=[True,'analytical'])
#     esdriver(molecules,analytical_gradient=[True,'numerical'])
# analyt_time = molecules.const.timing["Force"]
# molecules.const.timing["Force"] = []
# for i in range(1):
#     esdriver(molecules)
# backprop_time = molecules.const.timing["Force"]
# import os
# import numpy as np
# print(f'{os.path.basename(__file__)} {np.average(backprop_time)} {np.average(analyt_time)})')
print(f'Force is\n{molecules.force}')
#
# print(' Total Energy (eV):\n', molecules.Etot)
# print(f"Dipoles\n{molecules.dipole}")
# print(f"Charges:\n{molecules.q}")
# print('\n Electronic Energy (eV): ', molecules.Eelec)
# print('\n Nuclear Energy (eV):\n', molecules.Enuc)
# print('\n Heat of Formation (ev):\n', molecules.Hf)
# print('\n Orbital energies (eV):\n', molecules.e_mo)
