#get force y component
#move atom 0 along x direction, get a function fy(x)
#get dfy/dx

import torch

from seqm.seqm_functions.constants import Constants
from seqm.basics import Energy
import seqm
seqm.seqm_functions.scf_loop.debug = True
seqm.seqm_functions.scf_loop.SCF_BACKWARD_MAX_ITER = 1000
seqm.seqm_functions.scf_loop.RAISE_ERROR_IF_SCF_BACKWARD_FAILS = False
seqm.seqm_functions.MAX_ITER_TO_STOP_IF_SCF_BACKWARD_DIVERGE = 90

#check code to produce energy terms for each molecule
# with a 'learned' given parameters

#torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

N = 100

#"""
#pertubation direction
dir1 = torch.randn(3).to(device)
dir1 /= torch.norm(dir1)

#force direction
dir2 = torch.randn(3).to(device)
dir2 /= torch.norm(dir2)

#dir2 = dir2 - (dir2*dir1).sum()*dir1
#dir2 /= torch.norm(dir2)
#dir2 = dir1
#"""
"""
dir1 = torch.tensor([1.0,0.0,0.0])
dir2 = torch.tensor([0.0,1.0,0.0])
#"""

dxmin = -0.5
dxmax = 0.5

N += 1
dx = torch.arange(N+0.0,device=device)*(dxmax-dxmin)/(N-1.0)+dxmin

const = Constants().to(device)

species = torch.as_tensor([[8,6,1,1]],dtype=torch.int64, device=device) \
               .expand(N,4)

coordinates_op = torch.tensor([
             [
              [0.014497983896917479, 3.208059775069048e-05, -1.0697192017402962e-07],
              [1.3364260303072648, -3.2628339194439124e-05, 8.51016890853131e-07],
              [1.757659914731728, 1.03950803854101, -5.348699815983099e-07],
              [1.7575581407994696, -1.039614529391432, 2.84735846426227e-06]
             ],
             ], device=device)
#
coordinates = coordinates_op.expand(N,4,3).clone()
coordinates[...,0,:] += dx.unsqueeze(1)*dir1.unsqueeze(0)






elements = [0]+sorted(set(species.reshape(-1).tolist()))
seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'scf_eps' : 1.0e-7,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [1,0.0], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [True, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : elements, #[0,1,6,8],
                   'learned' : [], # learned parameters name list, e.g ['U_ss']
                   #'parameter_file_dir' : '../../seqm/params/', # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   'eig' : True,
                   'scf_backward': 2, #scf_backward==0: ignore the gradient on density matrix
                                      #scf_backward==1: use recursive formu
                                      #scf_backward==2: go backward scf loop directly
                   'scf_backward_eps' : 1.0e-7,
                   }

coordinates.requires_grad_(True)
eng = Energy(seqm_parameters).to(device)
Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = eng(const, coordinates, species, learned_parameters=dict(), all_terms=True)
#fy =  dE/dy
force, = torch.autograd.grad(Etot.sum(),coordinates, create_graph=True)

Fdir2 = (force[:,0,:]*dir2.unsqueeze(0)).sum(dim=1)
Fdir2.sum().backward()
#
dFdir2_ddir1 = (coordinates.grad[:,0,:]*dir1.unsqueeze(0)).sum(dim=1)



f=open('log.dat', 'w')
f.write("#index, dx (Angstrom), Fdir2 (eV/Angstrom), dFdir2_ddir1\n")
for i in range(N):
    f.write("%d %12.8e %12.8e %12.8e\n" % (i,dx[i],Fdir2[i], dFdir2_ddir1[i] ))
f.close()
