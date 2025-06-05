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

torch.set_printoptions(precision=7)
### create molecule object:
species = torch.as_tensor([
                            # [8,1,1],
                           # [1,1,],
                           # [1,1,1,1],
                           # [8,8,],
                          # [8,6,1,1],
                          [8,6,1,1,1,1],
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
                              #  [1.82,    0.0,    0.00],
                              #  [1.82,   -0.6633,    0.00],
                              #  [0.0,    0.0,    0.00],
                              #  [0.1,   -0.3633,    0.00],
                              # ],
                              # [
                              #  [1.82,    0.94,    0.00],
                              #  [1.82,   -0.94,    0.00],
                              # ],
                             # [
                             #  [0.00,    0.00,    0.00],
                             #  [1.22,    0.00,    0.20],
                             #  [1.82,    0.94,    0.00],
                             #  [1.81,   -0.93,    -0.20]
                             # ],
                            [
                              [0.702557224724283, 0.009082218612640677, 0.0032799875195913897],
                              [-0.707614145212714, -0.016445490928756433, -0.0053160302976097504],
                              [-1.0976860826353918, -0.7955775292486926, 0.6962719690854523],
                              [-1.097755234761741, -0.2082662697782238, -1.0360554595036433],
                              [-1.020222647310211, 1.001104652073342, 0.33967650723822534],
                              [1.0046208851957734, -0.8579975807303093, -0.29065697404201524]
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
                   'scf_eps' : 1.0e-10,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [0,0.1], # converger used for scf loop
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
                   # 'analytical_gradient':[True],
                   # 'do_scf_grad':[True, 'analytical'],  # [Want to calc SCF gradients:True/False, Which type: 'analytical,numerical']
                   # 'excited_states': {'n_states':6, 'method': 'rpa', 'cis_tolerance': 1e-10},
                   # 'active_state' : 2,
                   'scf_backward' : 2,
                   'normal modes': True,
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
    # esdriver(molecules,cis_gradient=[True])
    esdriver(molecules)
# backprop_time = molecules.const.timing["Force"]
# import os
# import numpy as np
# print(f'{os.path.basename(__file__)} {np.average(backprop_time)} {np.average(analyt_time)})')
print(f'Force is\n{molecules.force}')

print(' Total Energy (eV):\n', molecules.Etot)
# print('\n Electronic Energy (eV): ', molecules.Eelec)
# print('\n Nuclear Energy (eV):\n', molecules.Enuc)
# print('\n Heat of Formation (ev):\n', molecules.Hf)
# print('\n Orbital energies (eV):\n', molecules.e_mo)
exit()
import io
import contextlib
fd_gradient = torch.zeros_like(coordinates)
delta = 1e-5
natoms = coordinates.shape[1]
esdriver = Electronic_Structure(seqm_parameters).to(device)
for atom in range(natoms):
    for x in range(3):
        mol_coord_plus = coordinates.clone()
        mol_coord_minus = coordinates.clone()

        mol_coord_plus[0,atom,x] += delta
        mol_coord_minus[0,atom,x] -= delta

        with contextlib.redirect_stdout(io.StringIO()):
            molecules = Molecule(const, seqm_parameters, mol_coord_plus, species).to(device)
            esdriver(molecules)
            # energy_plus = molecules.cis_energies[:,seqm_parameters['active_state']-1]
            energy_plus = molecules.Etot

            molecules = Molecule(const, seqm_parameters, mol_coord_minus, species).to(device)
            esdriver(molecules)
            # energy_minus = molecules.cis_energies[:,seqm_parameters['active_state']-1]
            energy_minus = molecules.Etot

        fd_gradient[0,atom, x] = (energy_plus-energy_minus)/(2.0*delta)

print(f'The finite difference gradient of the quantity is:\n{fd_gradient}')
