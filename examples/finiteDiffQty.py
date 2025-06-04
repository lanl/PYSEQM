import torch
from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.ElectronicStructure import Electronic_Structure
import io
import contextlib

import warnings
warnings.filterwarnings("ignore")

def get_qty_from_output(geometry):
    molecules = Molecule(const, seqm_parameters, geometry, species).to(device)
    # Capture the printed output of calculate_energy
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        esdriver(molecules)
    
    # Retrieve the captured output and split it into lines
    output_lines = f.getvalue().strip().split('\n')

    # Search for the line containing 'Total energy = '
    searchStr = 'State   1:' # 'Dummy = '
    for line in output_lines:
        if searchStr in line:
            # Extract the energy value from the line
            energy_str = line.split()[2]
            energy = float(energy_str)
            return energy

    # Raise an error if the energy line is not found
    raise ValueError(f"No line with '{searchStr}' found in output")


torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

### create molecule object:
species = torch.as_tensor([
                            # [6,1,1,1,1],
                            # [1,1,],
                           [8,6,1,1],
                          ], # zero-padding for batching
                          dtype=torch.int64, device=device)

mol_coord = torch.tensor([
                                # [ 0.00000 ,   0.00000 ,   0.00000], 
                                # [ 0.00000 ,   0.00000 ,   1.08900],
                                # [ 1.02672 ,   0.00000 ,  -0.36300],
                                # [-0.51336 ,  -0.88916 ,  -0.36300],
                                # [-0.51336 ,   0.88916 ,  -0.36300],
                            # [1.82, 0.94, 0.00],
                            # [1.82, -0.94, 0.00],
                               # [0.00,    0.00,    0.00],
                               # [1.22,    0.00,    0.20],
                               # [1.82,    0.94,    0.00],
                               # [1.81,   -0.93,    -0.20]
                               [0.00,    0.00,    0.00],
                               [1.22,    0.00,    0.00],
                               [1.82,    0.94,    0.00],
                               [1.82,   -0.94,    0.00]
                         ],device=device)
fd_gradient = torch.zeros_like(mol_coord)
const = Constants().to(device)
elements = [0]+sorted(set(species.reshape(-1).tolist()))

seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'scf_eps' : 1.0e-12,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
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
                   'excited_states': {'n_states':4,'tolerance':1e-6,'method':'cis'},
                   # 'cis_tolerance' : 1e-8,
                   }

delta = 1e-5
natoms = mol_coord.shape[0]
esdriver = Electronic_Structure(seqm_parameters).to(device)
for atom in range(natoms):
    for x in range(3):
        mol_coord_plus = mol_coord.clone()
        mol_coord_minus = mol_coord.clone()

        mol_coord_plus[atom,x] += delta
        mol_coord_minus[atom,x] -= delta

        energy_plus = get_qty_from_output(torch.unsqueeze(mol_coord_plus,0))
        energy_minus = get_qty_from_output(torch.unsqueeze(mol_coord_minus,0))

        fd_gradient[atom, x] = (energy_plus-energy_minus)/(2.0*delta)
torch.set_printoptions(precision=15)
print(f'The finite difference gradient of the quantity is:\n{fd_gradient}')
