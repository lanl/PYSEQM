import torch

from seqm.seqm_functions.constants import Constants, ev_kcalpmol
from seqm.basics import  Parser, Hamiltonian, Pack_Parameters, Energy
from seqm.seqm_functions.parameters import params
from os import path
import sys
import seqm
#seqm.seqm_functions.scf_loop.MAX_ITER=10000
seqm.seqm_functions.scf_loop.debug = False
seqm.seqm_functions.diag.CHECK_DEGENERACY = False

here = path.abspath(path.dirname(__file__))

#check code to produce energy terms for each molecule
# with a 'learned' given parameters


torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


species = torch.as_tensor([[1,1],[6,6],[7,7],[8,8]],dtype=torch.int64, device=device) #[:2]
charges = torch.as_tensor([0.0,0.0,0.0,0.0], dtype=torch.double, device=device) #[:2]

DX = float(sys.argv[1])
coordinates = torch.tensor([
                  [[0.0000,    0.0000,    0.0000],
                   [DX,     0.0000,    0.0000]],
                 ], device=device).repeat(4, 1, 1).contiguous() #[:2]

elements = [0]+sorted(set(species.reshape(-1).tolist()))
seqm_parameters = {
                   'method' : 'PM3',  # AM1, MNDO, PM#
                   'scf_eps' : 1.0e-14,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [1,0.9], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [False, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : elements, #[0,1,6,8],
                   'learned' : [], # learned parameters name list, e.g ['U_ss']
                   'parameter_file_dir' : here+'/../seqm/params/', # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e3, # consistent with the unit on coordinates
                   'Hf_flag': False, # true: Hf, false: Etot-Eiso
                   'eig': True,
                   #'scf_backward':0,
                   }


const = Constants().to(device)

eng = Energy(seqm_parameters).to(device)
Etot_m_Eiso, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = eng(const, coordinates, species, all_terms=True, charges=charges)

print([DX,]+Etot_m_Eiso.numpy().tolist())
#print([DX,]+Etot_m_Eiso.numpy().tolist()+Etot.numpy().tolist()+Enuc.numpy().tolist())
#print(Etot_m_Eiso)
print(e)
print(torch.diagonal(P, dim1=1, dim2=2))
