import torch
from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.MolecularDynamics import Molecular_Dynamics_Basic

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


species = torch.as_tensor([[8,6,1,1],
                           [8,6,1,1],
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

const = Constants().to(device)

seqm_parameters = {
   'method': 'AM1',
   'scf_eps': 1.0e-8,
   'scf_converger': [0,0.2],
   # 'UHF': True,
   'excited_states': {'n_states':3},
   'active_state': 1,
}

timestep = 1.0
# timestep = 0.5

output = {
'molid': [0,1],
'thermo': 1,
'dump': 1,
'prefix': f'./examples/Outputs/vik_esmd.step_{timestep:.1f}'
}

molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
md = Molecular_Dynamics_Basic(seqm_parameters=seqm_parameters, timestep=timestep, output=output).to(device)
md.initialize_velocity(molecule,Temperature=0.0)
_ = md.run(molecule, 500, remove_com=[True, 1], Info_log=True)
