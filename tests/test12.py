import torch
from seqm.seqm_functions.constants import Constants
from seqm.basics import Force


#check computing force


torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

species = torch.as_tensor([[8,6,1,1],[8,6,1,1]],dtype=torch.int64, device=device)

coordinates = torch.tensor([
                  [
                   [0.0000,    0.0000,    0.0000],
                   [1.22732374,    0.0000,    0.0000],
                   [1.8194841064614802,    0.93941263319067747,    0.0000],
                   [1.8193342232738994,    -0.93951967178254525,    3.0565334533430606e-006]
                  ],
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
                   'learned' : [], # learned parameters name list, e.g ['U_ss']
                   #'parameter_file_dir' : '../params/MOPAC/', # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   '2nd_grad' : True,
                   }
#


force =  Force(seqm_parameters).to(device)
#coordinates.requires_grad_(True)
f, P, L = force(const, coordinates, species, learned_parameters=dict())[:3]
#print(f)
#print(P)
#print(L)
print(coordinates.requires_grad)
print(f.requires_grad)
#torch.autograd.grad(f,coordinates,torch.ones_like(f),create_graph=True) #this gives zeros as net force is always zero
g2 = torch.autograd.grad(f**2,coordinates,torch.ones_like(f),create_graph=True)
print(f)
print(g2)
"""
print(coordinates.shape)
print(f.shape)
print(g2.shape)
"""

"""
with torch.no_grad():
    coordinates += 0.01*f

f, P, L = force(const, coordinates, species, learned_parameters=dict(), P0=P)
print(f)

"""
if const.do_timing:
    print(const.timing)







#
