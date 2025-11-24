import os

import torch
from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.read_xyz import read_xyz

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# species, coordinates = read_xyz([os.path.join(os.path.dirname(__file__),'toluene.xyz')])
species, coordinates = read_xyz([os.path.join(os.path.dirname(__file__),'benzene.xyz')])
species = torch.as_tensor(species,dtype=torch.int64, device=device)[:]
coordinates = torch.tensor(coordinates, device=device)[:]

const = Constants().to(device)

elements = [0]+sorted(set(species.reshape(-1).tolist()))

seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'dispersion': True,
                   'scf_eps' : 1.0e-8,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [1], # converger used for scf loop
                   # 'excited_states': {'n_states':5},
                   # 'active_state': 2,
                   # 'scf_backward': 2,
                   'analytical_gradient':[True]#,'numerical']
                   }

molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)

### Create electronic structure driver:
esdriver = Electronic_Structure(seqm_parameters).to(device)

### Run esdriver on molecules:
# for i in range(1):
#     esdriver(molecules,analytical_gradient=[True,'analytical'])
esdriver(molecules)
# import os
# import numpy as np
# print(f'{os.path.basename(__file__)} {np.average(backprop_time)} {np.average(analyt_time)})')
# print(f'Force is\n{molecules.force}')

print(' Total Energy (eV):\n', molecules.Etot)
print('\n Electronic Energy (eV): ', molecules.Eelec)
print('\n Nuclear Energy (eV):\n', molecules.Enuc)
print('\n Heat of Formation (kcal/mol):\n', 23.0609*molecules.Hf)
print(f"Force is \n{molecules.force}")
