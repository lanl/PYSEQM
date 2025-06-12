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
species = torch.as_tensor([[6,     6,     6,     6,     6,     6,     1,     1,     1,     1,     1,     1]] ,                           
                          dtype=torch.int64, device=device)

coordinates = torch.tensor([
                             [
                               [ -35.894866 ,   9.288626 ,  10.054420], 
                               [ -36.369609 ,   8.136872 ,   9.487425],
                               [ -34.536331 ,   9.569570 ,   9.955502],
                               [ -35.492308 ,   7.159723 ,   8.983842],
                               [ -33.712643 ,   8.697458 ,   9.304412],
                               [ -34.183340 ,   7.494548 ,   8.762036],
                               [ -36.585465 ,   9.939934 ,  10.537961],
                               [ -37.450824 ,   8.064607 ,   9.436016],                            
                               [ -34.070057 ,  10.502494 ,  10.500065],                            
                               [ -35.950748 ,   6.201235 ,   8.765212],                            
                               [ -33.438373 ,   6.826374 ,   8.246956],                            
                               [ -32.654911 ,   8.872607 ,   9.123019]
                              ],
                           ], device=device)

const = Constants().to(device)


seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [2,0.0], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [False, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'learned' : [], # learned parameters name list, e.g ['U_ss']
                   #'parameter_file_dir' : '../seqm/params/', # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   'eig' : True,
                   'analytical_gradient':[True]#,'numerical']
                   # 'do_scf_grad':[True, 'analytical'],  # [Want to calc SCF gradients:True/False, Which type: 'analytical,numerical']
                   # 'excited_states': {'n_states':3},
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
print('\n Heat of Formation (ev):\n', molecules.Hf)
# print('\n Orbital energies (eV):\n', molecules.e_mo)
