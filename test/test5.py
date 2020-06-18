import torch
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.parameters import params
from seqm.basics import  Parser, Hamiltonian, Pack_Parameters, Energy,Force

#check computing force


torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

species = torch.as_tensor([[8,6,1,1]],dtype=torch.int64, device=device)

coordinates = torch.tensor([
                  [
                   [0.0000,    0.0000,    0.0000],
                   [1.22732374,    0.0000,    0.0000],
                   [1.8194841064614802,    0.93941263319067747,    0.0000],
                   [1.8193342232738994,    -0.93951967178254525,    3.0565334533430606e-006]
                  ]
                 ], device=device)


const = Constants().to(device)

#may need to add scaling factor for length and energy on const, check constants.py

elements = [0]+sorted(set(species.reshape(-1).tolist()))
seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [2,0.0], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [True, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : elements, #[0,1,6,8],
                   'learned' : ['g_ss'], # learned parameters name list, e.g ['U_ss']
                   'parameter_file_dir' : '../params/MOPAC/', # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   }
#


#parser is not needed here, just use it to get Z and create "learned" parameters
#prepare a fake learned parameters: learnedpar
parser = Parser(seqm_parameters).to(device)
nmol, molsize, \
nHeavy, nHydro, nocc, \
Z, maskd, atom_molid, \
mask, pair_molid, ni, nj, idxi, idxj, xij, rij = parser(const, species, coordinates)
#add learned parameters
#here use the data from mopac as example

p=params(method=seqm_parameters['method'],
         elements=seqm_parameters['elements'],
         root_dir=seqm_parameters['parameter_file_dir'],
         parameters=seqm_parameters['learned']).to(device)
p, =p[Z].transpose(0,1).contiguous()
p.requires_grad_(True)
learnedpar = {'g_ss':p}




force =  Force(seqm_parameters).to(device)
#coordinates.requires_grad_(True)

#######################################
#require grad on p
f, P, L = force(const, coordinates, species, learned_parameters=learnedpar, par_grad=True)[:3]
#print(f)
print(p.grad)



if const.do_timing:
    print(const.timing)








#
