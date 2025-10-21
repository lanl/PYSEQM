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
   'scf_converger': [1],
   # 'UHF': True,
   'excited_states': {'n_states':3},
   'active_state': 1,
}

# timestep = 1.0
timestep = 0.4

output = {
# 'molid': [0,1],
'molid': [0],
'prefix': f'./examples/Outputs/vik_esmd.step_{timestep:.1f}',
'print every': 1,
"xyz": 1,
"h5": {
    "data": 1,      # write T/Ek/Ep, excitations, MO, etc.; 0 disables
    "velocities": 1,   # write vel/forces/coords; 0 disables
    "write_mo": True,
    },
}

torch.manual_seed(42)
molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
from seqm.MolecularDynamics import Molecular_Dynamics_Langevin, XL_BOMD

# md = Molecular_Dynamics_Langevin( damp=50.0, seqm_parameters=seqm_parameters,
#                                            Temp=300.0, timestep=timestep,
#                                            output=output).to(device)
xl_bomd_params={'k':6}

# md =  XL_BOMD(xl_bomd_params=xl_bomd_params, Temp = 400.0,
#               seqm_parameters=seqm_parameters, timestep=0.4, output=output).to(device)
md = Molecular_Dynamics_Basic(seqm_parameters=seqm_parameters, Temp=400.0, timestep=timestep, output=output).to(device)
_ = md.run(molecule, 5, remove_com=None,reuse_P=False)
