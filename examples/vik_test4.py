import torch
from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.ElectronicStructure import Electronic_Structure

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
                           # [1,1,],
                           # [8,6,],
                          [8,6,1,1],
                          # [8,8,6,0]
                          ], # zero-padding for batching
                          dtype=torch.int64, device=device)

coordinates = torch.tensor([
                              # [
                              #  [0.82,    0.00,    0.00],
                              #  [1.82,    0.94,    0.00],
                              #  [1.82,   -0.94,    0.00],
                              # ],
                              # [
                              #  [1.82,    0.0,    0.00],
                              #  [1.82,   -0.6633,    0.00],
                              # ],
                              # [
                              #  [1.82,    0.94,    0.00],
                              #  [1.82,   -0.94,    0.00],
                              # ],
                             [
                              [0.00,    0.00,    0.00],
                              [1.22,    0.00,    0.20],
                              [1.82,    0.94,    0.00],
                              [1.81,   -0.93,    -0.20]
                             ],
                            #  [
                            #   [0.00,    0.00,    0.00],
                            #   [1.23,    0.00,    0.00],
                            #   [1.82,    0.94,    0.00],
                            #   [0.0,0.0,0.0]            # zero-padding for batching
                            #  ]
                            ], device=device)

const = Constants().to(device)

elements = [0]+sorted(set(species.reshape(-1).tolist()))

seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'scf_eps' : 1.0e-9,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
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
                   'eig' : True,
                   # 'uhf' : True,
                   # 'analytical_grad':True
                   # 'do_scf_grad':[True, 'analytical'],  # [Want to calc SCF gradients:True/False, Which type: 'analytical,numerical']
                  'excited_states': [True,4]
                   }

molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)

### Create electronic structure driver:
esdriver = Electronic_Structure(seqm_parameters).to(device)

### Run esdriver on molecules:
# for i in range(1):
#     esdriver(molecules,analytical_gradient=[True,'analytical'])
# analyt_time = molecules.const.timing["Force"]
# molecules.const.timing["Force"] = []
for i in range(1):
    esdriver(molecules)
# backprop_time = molecules.const.timing["Force"]
# import os
# import numpy as np
# print(f'{os.path.basename(__file__)} {np.average(backprop_time)} {np.average(analyt_time)})')
# print(f'Force is\n{molecules.force}')

print(' Total Energy (eV):\n', molecules.Etot)
# print('\n Electronic Energy (eV): ', molecules.Eelec)
# print('\n Nuclear Energy (eV):\n', molecules.Enuc)
# print('\n Heat of Formation (ev):\n', molecules.Hf)
print('\n Orbital energies (eV):\n', molecules.e_mo)
