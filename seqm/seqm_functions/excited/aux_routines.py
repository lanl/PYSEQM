'''
Auxiliary routine to get molecule object and pass it to Davidson algorithm
Ideally, should not exist in PYSEQM, and should be rewritten as interafce modeule to run calculations
Leaving here for Davidson prototype only 

'''

import torch
from ase.io import read as ase_read
import seqm
from seqm.seqm_functions.constants import Constants
from seqm.ElectronicStructure import Electronic_Structure

def run_seqm_1mol(xyz, device, dtype):
    """
    run_seqm_1mol : run PYSEQM for a single molecule

    Args:
        xyz (str): path to xyz file

    Returns:
        Molecule object: PYSEQM object with molecule data
    """    
    
    atoms = ase_read(xyz)
    species = torch.tensor([atoms.get_atomic_numbers()], dtype=torch.long, device=device)
    coordinates = torch.tensor([atoms.get_positions()], dtype=dtype, device=device)
    
    const = Constants().to(device)

    elements = [0]+sorted(set(species.reshape(-1).tolist()))

    seqm_parameters = {
                    'method' : 'PM3',  # AM1, MNDO, PM#
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
                    'eig' : True,
                    'excited' : True,
                    }

    mol = seqm.Molecule.Molecule(const, seqm_parameters, coordinates, species).to(device)

    ### Create electronic structure driver:
    esdriver = Electronic_Structure(seqm_parameters).to(device)

    ### Run esdriver on m:
    esdriver(mol)
    
    return mol
