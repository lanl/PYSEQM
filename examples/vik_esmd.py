import torch
from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.MolecularDynamics import Molecular_Dynamics_Basic

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


species = torch.as_tensor([[8,6,1,1],
                            [8,6,1,1],
                           # [8,8,6,0],
                           ],
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
                              [1.21,    0.00,    0.00],
                              [1.62,    0.94,    0.00],
                              [1.82,   -0.84,    0.00]
                              ],
                           ], device=device)
one = False
# one = True
if one:
    species = torch.as_tensor([[8,8,6],
                               ],
                            dtype=torch.int64, device=device)

    coordinates = torch.tensor([
                                  [
                                  [0.00,    0.00,    0.00],
                                  [1.21,    0.00,    0.00],
                                  [1.62,    0.94,    0.00],
                                  ],
                               ], device=device)

const = Constants().to(device)

seqm_parameters = {
   'method': 'AM1',
   'scf_eps': 1.0e-8,
   'scf_converger': [0,0.2],
   # 'UHF': True,
   # 'excited_states': {'n_states':3},
   # 'active_state': 1,
}

# timestep = 1.0
timestep = 0.5

output = {
# 'molid': [0,1],
'molid': [0],
'thermo': 1,
'dump': 1,
'prefix': f'./examples/Outputs/vik_esmd.step_{timestep:.1f}'
}

torch.manual_seed(42)
molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
from seqm.MolecularDynamics import Molecular_Dynamics_Langevin, XL_BOMD

md = Molecular_Dynamics_Langevin( damp=50.0, seqm_parameters=seqm_parameters,
                                           Temp=300.0, timestep=timestep,
                                           output=output).to(device)
xl_bomd_params={'k':6}

md =  XL_BOMD(xl_bomd_params=xl_bomd_params, Temp = 400.0,
              seqm_parameters=seqm_parameters, timestep=0.4, output=output).to(device)
# md = Molecular_Dynamics_Basic(seqm_parameters=seqm_parameters, Temp=300.0, timestep=timestep, output=output).to(device)
# molecule.velocities = torch.tensor([[[-20.9603, -0.5334,  0.4491], 
#                                     [-21.2032, -0.2912, -1.8145],
#                                     [ 20.0957, -2.0413,  7.2473],
#                                     [ 20.4869, 13.9784,  7.2473],
#                                     ]],device=device)*1e-3
# _ = md.run(molecule, 1000, remove_com=('angular',10))
_ = md.run(molecule, 20, remove_com=('angular', 1), Info_log=True, h5_write_mo=True)
# _ = md.run(molecule, 10,Info_log=True)
